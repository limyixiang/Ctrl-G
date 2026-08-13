"""A hidden Markov model over token sequences, viewed as a probabilistic circuit.

The model is parameterized by three tensors (see HMM.__init__):
    alpha_exp[i, j] = P(z_{t+1} = j | z_t = i)  -- transitions, in probability space
    beta[i, w]      = log P(x_t = w | z_t = i)  -- emissions, in log space
    gamma[i]        = log P(z_0 = i)            -- initial distribution, in log space

`forward` runs the backward (bottom-up) message pass to get the sequence
log-likelihood; `backward` runs the top-down pass that turns those messages into
the expected counts ("flows") consumed by an EM update. Both are written in log
space with explicit max-subtraction for numerical stability, since a sequence's
probability underflows float32 within a few dozen tokens.
"""

import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def matmul(A, B):
    """Thin wrapper around torch.matmul, kept as a single seam for swapping in a custom kernel."""
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(pf, pp, cp):
    """Accumulate transition flows over a batch; the name spells out the index signature.

    Computes af[i, j] = sum_b pf[i, b] * exp(cp[b, j] - pp[i, b]), i.e. for every
    (parent state i, child state j) pair it sums, over the batch, the parent's flow
    scaled by the child/parent probability ratio.
    pf: parent flow,        hidden_states * batch_size ("ib")
    pp: log parent probs,   hidden_states * batch_size ("ib")
    cp: log child probs,    batch_size * hidden_states ("bj")
    Returns hidden_states * hidden_states ("ij").
    Note the transition probability alpha_exp[i, j] is NOT applied here -- the caller
    multiplies it in once at the end of the EM step.
    """
    # shift both operands by the same per-example constant before exponentiating;
    # it cancels in the ratio but keeps exp() away from underflow
    ll = torch.amax(cp, dim=-1) # batch_size
    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0 # a parent with zero probability carries no flow; avoids 0/0 -> NaN
    af = torch.matmul(ratio, cp)

    return af


class HMM(nn.Module, PyTorchModelHubMixin):
    """HMM with fixed-size hidden state space, distilled from a base LM and used to
    guide constrained generation. Parameters are buffers rather than gradient-trained
    weights: they are re-estimated wholesale by EM (see distillation/train_hmm.py).
    """
    def __init__(self, hidden_states: int, vocab_size: int, eos_token_id: int):
        super().__init__()

        # random init; in practice these are immediately overwritten by update_params
        # with latent variable distillation estimates (see distillation/lvd_hmm.py)
        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states), dim=1) # rows sum to 1
        beta = torch.log_softmax(torch.randn(hidden_states, vocab_size), dim=1) # rows sum to 1 in prob space
        gamma = torch.log_softmax(torch.randn(hidden_states), dim=0)

        # Parameters (not plain tensors) so they move with .to(device) and are saved by
        # save_pretrained, but requires_grad=False since EM updates them directly
        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id


    def update_params(self, alpha_exp, beta, gamma):
        """Overwrite the parameters in place, e.g. with the result of one EM step."""
        self.alpha_exp.data = alpha_exp
        self.beta.data = beta
        self.gamma.data = gamma


    # bottom-up circuit pass
    def forward(self, input_ids):
        """Backward (in HMM terms) message pass, run bottom-up over the circuit.

        input_ids: batch_size * seq_len; an id of -1 marks a MISSING token, which
            contributes no evidence (used by the token dropout in train_hmm.py).
        Returns a list `ys` of length seq_len + 1 where
            ys[k][i, b] = log P(x_k: | z_k = i) for k = seq_len-1-... , i.e. index k
            corresponds to time step t = seq_len - 1 - k, and
            ys[-1][b] = log P(x_0:) -- the log-likelihood of the whole sequence.
        """
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape

        # time-major layout so each step of the recursion below is a contiguous slice
        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()

        # gather the emission log-prob of every observed token under every hidden state;
        # broadcasting (1, H, 1) state indices against (seq_len, 1, B) token indices
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        # -1 indexes the last vocab entry above, so zero those out instead:
        # log-prob 0.0 == probability 1.0, i.e. the token is unobserved
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1) # 0.0 for MISSING token

        ys = []
        y = torch.zeros((hidden_states, batch_size), device=device)
        for t in range(seq_len-1, -1, -1):
            # y holds log P(x_{t+1}: | z_{t+1} = j); push it back through the transitions
            # to get log P(x_{t+1}: | z_t = i). Skipped at the last position, where the
            # message starts out empty (all-zero == probability 1).
            if t != seq_len - 1:
                y_max = torch.amax(y, dim=0, keepdim=True) # 1 * batch_size; log-sum-exp shift
                y = torch.exp(y - y_max)
                y = matmul(alpha_exp, y) # sum_j P(z_{t+1}=j | z_t=i) * P(x_{t+1}: | z_{t+1}=j)
                y = torch.log(y) + y_max
            y += input_probs[t, :, :] # multiply in P(x_t | z_t=i); hidden_states * batch_size
            ys.append(y) # ys is appended in reverse time order: ys[k] is time seq_len-1-k

        # finally marginalize the first hidden state against the initial distribution
        y_max = torch.amax(y, dim=0) # batch_size
        y = torch.exp(y - y_max.unsqueeze(0))
        y = matmul(gamma_exp.unsqueeze(0), y).squeeze() # (1,H) @ (H,B) -> batch_size
        y = torch.log(y) + y_max

        ys.append(y) # ys[-1][b] = log P(x_0:) for example b

        return ys


    # top-down circuit pass
    def backward(self, input_ids, probs,
        alpha_flow, beta_flow, gamma_flow):
        """E-step: turn the messages from forward() into expected counts, accumulated
        in place into the three flow accumulators so batches can be summed up.

        input_ids: batch_size * seq_len, same tensor that was passed to forward()
        probs: the list returned by forward() for that batch
        alpha_flow: hidden_states * hidden_states; expected transition counts, but
            WITHOUT the alpha_exp factor -- the caller multiplies it in once at the end
        beta_flow: (vocab_size + 1) * hidden_states; expected emission counts, with the
            extra row vocab_size collecting the mass of MISSING tokens
        gamma_flow: hidden_states; expected counts of starting in each state

        The "flow" of a node is its posterior probability given the observed sequence,
        so this pass is the standard forward-backward E-step written as a circuit
        traversal: it walks top-down from z_0, computing P(z_t | x) at each step.
        """
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape

        # recompute the emission log-probs exactly as in forward(); they are needed to
        # strip the emission factor back out of each message below
        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous() # seq_len * batch_size
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        flows = []
        # flow of the first hidden state:
        #   P(z_0=i | x) = P(z_0=i) * P(x_0: | z_0=i) / P(x_0:)
        # probs[-2] is the message at t=0, probs[-1] the sequence log-likelihood
        pf = gamma_exp.unsqueeze(0) * torch.exp(
            torch.permute(probs[-2], (1, 0)).contiguous() - probs[-1][:, None]) # batch_size * hidden_states
        flows.append(pf)

        # update gamma_flow
        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len-1):
            # forward() appended in reverse time order, so time t lives at this index
            layer_idx = seq_len - t - 1
            # parent: the message at time t with the emission of x_t divided out,
            # leaving log P(x_{t+1}: | z_t = i)
            pp = probs[layer_idx] - input_probs[t, :, :] # parent probs; hidden_states * batch_size
            # child: log P(x_{t+1}: | z_{t+1} = j)
            cp = probs[layer_idx-1] # child probs; hidden_states * batch_size

            # transition flow P(z_t=i, z_{t+1}=j | x), summed over the batch:
            #   P(z_t=i | x) * alpha_exp[i,j] * P(x_{t+1}: | z_{t+1}=j) / P(x_{t+1}: | z_t=i)
            # the alpha_exp[i,j] factor is applied by the caller after all batches
            alpha_flow.add_(ib_ib_bj_to_ij(torch.permute(pf, (1, 0)).contiguous(),
                pp,
                torch.permute(cp, (1, 0)).contiguous()))

            # same quantity summed over i instead of b, giving the next layer's flow
            # P(z_{t+1}=j | x); pp_max cancels between the two exp() terms
            pp = torch.permute(pp, (1, 0)) # batch_size * hidden_states
            cp = torch.permute(cp, (1, 0)) # batch_size * hidden_states
            pp_max = torch.amax(pp, dim=1, keepdim=True) # batch_size * 1
            pp_ = torch.exp(pp - pp_max)

            ratio = pf / pp_
            ratio[pp_ == 0.0] = 0.0 # zero-probability parents carry no flow; avoids 0/0
            pf = matmul(ratio, alpha_exp) * torch.exp(cp - pp_max)

            flows.append(pf)

        # update beta_flow: every position adds its state posterior to the row of the
        # token it emitted, so beta_flow[w, i] ends up as the expected count of state i
        # emitting token w
        flows = torch.stack(flows, dim=0) # seq_len * batch_size * hidden_states
        input_ids_[input_ids_ == -1] = vocab_size # route MISSING tokens to the spare row
        input_ids_ = input_ids_[:, :, None].expand(-1, -1, hidden_states).view(seq_len * batch_size, hidden_states)
        beta_flow.scatter_add_(0, input_ids_, flows.view(seq_len * batch_size, hidden_states))


    def loglikelihood(self, input_ids, batch_size):
        """Total (not averaged) log-likelihood of input_ids, evaluated in mini-batches.

        input_ids lives on CPU here and is moved over one batch at a time, so a dev set
        larger than GPU memory can still be scored.
        """
        device = self.alpha_exp.device
        data_size, seq_len = input_ids.shape

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1]) # probs_[-1] is the per-example log-likelihood

        return ll