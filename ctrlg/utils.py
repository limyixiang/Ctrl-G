"""Constrained decoding: a LogitsProcessor that steers a base LLM with an HMM + DFA,
plus the log-space linear algebra helpers it needs.

Naming conventions used throughout this file:

HMM parameters (see ctrlg/hmm.py). The HMM is a cheap approximation of the base LLM;
z_t denotes its latent state at position t:
    alpha_exp -- transition matrix, *exp*onentiated (i.e. in probability space, not log):
        alpha_exp[i, j] = P(z_{t+1} = j | z_t = i); hidden_states * hidden_states.
    beta      -- log emission matrix: beta[i, w] = log P(token w | z_t = i);
        hidden_states * vocab_size.
    gamma     -- log initial state distribution: gamma[i] = log P(z_0 = i).
    hidden_states -- the number of HMM latent states.

DFA tensors (see ctrlg/dfa.py). The DFA encodes the logical constraint; its states are
the "vertices" V and its transitions are the "edges" E:
    VE_mask -- Vertex-to-Edge incidence, num_states * num_transitions;
        VE_mask[u, e] = 1 iff edge e leaves state u.
    EV_mask -- Edge-to-Vertex incidence, num_transitions * num_states;
        EV_mask[e, v] = 1 iff edge e enters state v.
    T_mask  -- Token mask, num_transitions * vocab_size; T_mask[e, w] = 1 iff token w
        is a valid label for edge e.
    E2Src / E2Dst -- Edge-to-Source / Edge-to-Destination state index, num_transitions.
    T_weights -- per-edge emission weight derived from T_mask and beta,
        num_transitions * hidden_states (see ConstraintLogitsProcessor.__init__).

The four caches of ConstraintLogitsProcessor (A/B/C/D name the order in which __init__
builds them; A and B are the usual HMM forward/backward messages):
    A_cache -- forward pass over everything generated so far (seeded with prefix_ids).
    B_cache -- backward pass over the fixed suffix.
    C_cache -- backward pass over the DFA, i.e. the constrained "future" weights.
    D_cache -- the DFA state each prefix leads to.

Everything is in log space unless the name says otherwise (the _exp suffix), and
neginf = -1e30 stands in for log(0) so that arithmetic stays finite.
"""

import torch
from transformers import LogitsProcessor

torch.set_float32_matmul_precision('high')


@torch.compile
def logsumexp(A, dim):
    """torch.logsumexp, wrapped so that it is compiled once and reused at every decoding step.
    logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    x = (1, 2, 3) -> ln(e^1 + e^2 + e^3)
    logsumexp is the log-space version of 'add up probabilities'.
    """
    return torch.logsumexp(A, dim)


@torch.compile
def matmul_log(A, B):
    """Matrix multiply where both operands are in log space: returns log(exp(A) @ exp(B)).

    The row/column maxima are subtracted before exponentiating and added back afterwards,
    which keeps exp() from underflowing/overflowing (the log-sum-exp trick).
    bd -- the axis of B that is contracted (its second-to-last), so B may carry batch dims.
    """
    bd = len(B.shape) - 2
    A_max = torch.amax(A, dim=-1, keepdim=True)
    B_max = torch.amax(B, dim=bd, keepdim=True)
    A = A - A_max
    B = B - B_max
    A.exp_()
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max + B_max)

    return C


@torch.compile
def matmul_loga_b(A, B):
    """Mixed matmul, A in log space and B in probability space: returns log(exp(A) @ B)."""
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A = A - A_max
    A.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max)

    return C


@torch.compile
def matmul_a_logb(A, B):
    """Mixed matmul, A in probability space and B in log space: returns log(A @ exp(B))."""
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)

    return C


@torch.compile
def distribute_state_weights(E2D, y):
    """Copy a per-state weight vector out onto the edges: out[e, :] = y[E2D[e], :].

    E2D -- an Edge-to-state index map (E2Src or E2Dst); with E2Dst every edge picks up
        the weights of the state it points at, which is how a backward pass over the DFA
        moves information from states to the edges feeding into them.
    y: num_states * hidden_states -> out: num_transitions * hidden_states.
    """
    device = y.device
    _, hidden_states = y.shape
    return y[E2D[:, None],
            torch.arange(0, hidden_states, device=device)[None, :]]


@torch.compile
def aggregate_edge_weights(E2S, y, num_states):
    """Sum per-edge weights back into their states, in log space:
    out[u, :] = logsumexp over every edge e with E2S[e] == u of y[e, :].

    E2S -- an Edge-to-state index map (here E2Src, so each edge reports to the state it
        leaves from). scatter_reduce has no log-sum-exp mode, so this is done manually:
        take the per-state max with 'amax', subtract it, scatter_reduce with 'sum', log,
        and add the max back. States that no edge maps to end up at neginf (-1e30).
    y: num_transitions * hidden_states -> out: num_states * hidden_states.
    """
    device = y.device
    _, hidden_states = y.shape
    num_edges = E2S.shape[0]
    E2S_ = E2S[:, None].expand(-1, hidden_states)

    y_out = torch.zeros(num_states, hidden_states, device=device)

    y_out_max = -1e30 * torch.ones(num_states, hidden_states, device=device)
    y_out_max.scatter_reduce_(0, E2S_, y, reduce='amax')
    y_max = y_out_max[E2S[:, None],
        torch.arange(0, hidden_states, device=device)[None, :]]

    y = torch.exp(y - y_max)
    y_out.scatter_reduce_(0, E2S_, y, reduce='sum')
    y_out.log_()
    y_out.nan_to_num(neginf=-1e30)
    y_out += y_out_max

    return y_out

# Original one
# def ends_at(prefix, suffix,
#     offset_min, D_cache, dfa_model):
#     ans = []
#     for s in range(0, len(suffix)):
#         offset = len(prefix) - s
#         if offset < offset_min:
#             break
#         state = D_cache[tuple(prefix[:-s])] if s != 0 else D_cache[tuple(prefix)]
#         if dfa_model.is_accept(state):
#             if s == 0 or tuple(suffix[:s]) == prefix[-s:]:
#                 ans.append(s)
#     return ans
def ends_at(prefix, suffix,
    offset_min, D_cache, dfa_model):
    """Find the ways `prefix` could already have finished generating and started emitting `suffix`.

    Returns every overlap length s (0 <= s < len(suffix)) such that
      - the last s tokens of `prefix` are exactly the first s tokens of `suffix`,
      - the DFA accepts prefix[:-s], i.e. the constraint is already satisfied at the point
        where the suffix would begin, and
      - prefix[:-s] is at least offset_min tokens long (the minimum-length requirement).
    s = 0 means generation could stop right here, with none of the suffix emitted yet;
    the caller then knows suffix[s] is a legal next token and adds its probability mass.

    D_cache -- prefix tuple -> DFA state reached after reading it. Since _cleanup_cache
        only keeps the live prefixes, the states needed here are recomputed on demand.
    """
    ans = []
    # Ensure we can get the DFA state for any target prefix by recomputing from the nearest cached ancestor.
    def ensure_state(target):
        if target in D_cache:
            return D_cache[target]
        # Find the longest cached ancestor of `target`.
        k = len(target)
        while k > 0 and target[:k] not in D_cache:
            k -= 1
        if k == 0:
            # Fall back to a shortest cached base (should include the initial prompt tuple).
            if not D_cache:
                raise KeyError("D_cache is empty; cannot recompute DFA state.")
            shortest_len = min(len(p) for p in D_cache.keys())
            base = None
            for p in D_cache.keys():
                if len(p) == shortest_len and (len(p) == 0 or (len(p) <= len(target) and p == target[:len(p)])):
                    base = p
                    break
            if base is None:
                base = min(D_cache.keys(), key=len)
            state = D_cache[base]
            start_idx = len(base)
        else:
            base = target[:k]
            state = D_cache[base]
            start_idx = k
        # Walk remaining tokens (don’t cache intermediates).
        for tok in target[start_idx:]:
            state = dfa_model.next_state(state, tok)
        # Cache only the final state for `target`.
        D_cache[target] = state
        return state

    for s in range(0, len(suffix)):
        offset = len(prefix) - s
        if offset < offset_min:
            break
        target = tuple(prefix) if s == 0 else tuple(prefix[:-s])
        state = ensure_state(target)
        if dfa_model.is_accept(state):
            if s == 0 or tuple(suffix[:s]) == prefix[-s:]:
                ans.append(s)
    return ans



class ConstraintLogitsProcessor(LogitsProcessor):
    """Reweights a base LLM's next-token distribution so that the completion satisfies a DFA.

    A generation is laid out as   prompt_ids | prefix_ids + generated tokens | suffix_ids,
    where the DFA constraint applies to the middle part, whose length must land inside one
    of `token_ranges` (defaults to a single [min_new_tokens, max_new_tokens] range).

    At each decoding step, __call__ adds `alpha` * log P(the generation can still be
    completed into something the DFA accepts and that ends with suffix_ids | prefix so far,
    next token = w) to the base model's log-probabilities. That probability is unavailable
    from the LLM itself, so it is estimated with the HMM, which -- unlike the LLM -- can be
    marginalized over all possible futures in closed form. alpha = 1.0 is the exact
    posterior reweighting and smaller values soften the constraint; `temperature` rescales
    the final distribution.
    """
    def __init__(self, hmm_model, dfa_model,
        min_new_tokens, max_new_tokens, prompt_ids, prefix_ids=[], suffix_ids=[],
        temperature=1.0, alpha=1.0, token_ranges=None, hmm_batch_size=None):
        """Precompute everything that does not depend on the tokens sampled at run time.

        This is the expensive part: caches A (prefix forward pass), B (suffix backward
        pass), C (backward pass over the DFA for every allowed number of remaining tokens)
        and D (DFA states) are all built here, so that each decoding step only has to
        extend A/D by one token and read C off the shelf.

        hmm_batch_size -- caps how many sequences are pushed through the
            num_states * vocab_size computation at once; None means the whole batch.
        """
        device = hmm_model.alpha_exp.device
        hidden_states, vocab_size = hmm_model.hidden_states, hmm_model.vocab_size
        num_states = dfa_model.num_states

        neginf = -1e30
        neginf_cuda = neginf * torch.ones(1, device=device)
        alpha_exp, beta, gamma = hmm_model.alpha_exp, hmm_model.beta, hmm_model.gamma
        alpha_exp_t = torch.transpose(alpha_exp, 0, 1)

        if token_ranges is None:
            token_ranges = [[min_new_tokens, max_new_tokens]]
        min_tokens = min_new_tokens
        max_tokens = max([x[1] for x in token_ranges])

        # initialize cache A -- the HMM forward pass over the prefix.
        # A_cache[p][i] = log P(the HMM emits the tokens of p, and z = i at the next step),
        # so it summarizes everything already generated. Each step emits a token (+ beta)
        # and then takes one transition (matmul with alpha_exp).
        A_cache = {}
        y = gamma.clone()
        for t in range(0, len(prefix_ids)):
            y = y + beta[:, prefix_ids[t]]
            y = matmul_loga_b(y[None, :], alpha_exp).squeeze(0)
        A_cache[tuple(prefix_ids)] = y

        # initialize cache B -- the HMM backward pass over the fixed suffix.
        # B_cache[t, i] = log P(the HMM emits suffix_ids[t:] | z = i at that step), i.e. the
        # cost of finishing with the suffix. Walked right to left; y is left holding row 0
        # (all zeros, meaning "no cost", when there is no suffix) for cache C below.
        B_cache = torch.empty(len(suffix_ids), hidden_states, device=device)
        y = torch.zeros(hidden_states, device=device)
        for t in range(len(suffix_ids)-1, -1, -1):
            if t != len(suffix_ids) - 1:
                y = matmul_a_logb(alpha_exp, y[:, None]).squeeze(-1)
            y = y + beta[:, suffix_ids[t]]
            B_cache[t, :] = y

        # compute T_weights -- collapse each DFA edge's set of allowed tokens into a single
        # emission weight: T_weights[e, i] = log P(the HMM emits some token labelling edge e
        # | z = i) = log sum over those tokens of exp(beta[i, w]). Edges labelled with no
        # token log to -inf, hence the nan_to_num_.
        T_mask = dfa_model.T_mask
        VE_mask = dfa_model.VE_mask
        EV_mask = dfa_model.EV_mask
        E2Src, E2Dst = dfa_model.E2Src, dfa_model.E2Dst

        T_weights = matmul_a_logb(T_mask, torch.transpose(beta, 0, 1)) # num_transitions * hidden_states
        T_weights.nan_to_num_(neginf=neginf)

        # initialize cache C -- the backward pass over the DFA, one layer per remaining token.
        # C[t, v, i] = log P(the generation runs for exactly t more tokens, walking the DFA
        # from state v into an accept state, and is then followed by the suffix | the HMM was
        # in state i when it emitted the token that moved the DFA into v).
        # C[0] is "stop now": only accept states are live,
        # and they inherit the suffix weight y from cache B (after one transition, since the
        # suffix starts at the following step).
        y_ = torch.full((num_states, hidden_states), neginf, device=device)
        y_[list(dfa_model.accept_states), :] = y
        y = matmul_loga_b(y_, alpha_exp_t) # num_states * hidden_states

        # Each iteration prepends one more token: push the state weights out onto the edges
        # that lead there (E2Dst), pay that edge's emission weight, sum the edges back into
        # their source states (E2Src), then take one HMM transition backwards (alpha_exp_t).
        C = torch.empty(max_tokens+1, num_states, hidden_states, device=device)
        C[0, :, :] = y
        for t in range(1, max_tokens+1):
            y = distribute_state_weights(E2Dst, y) # num_transitions * hidden_states
            y = aggregate_edge_weights(E2Src, T_weights + y, num_states=num_states) # num_states * hidden_states
            y = matmul_loga_b(y, alpha_exp_t) # num_states * hidden_states
            C[t, :, :] = y

        # precompute ranges for C_cache -- at decoding time the generation may still run for
        # anywhere between `remaining_min` and `remaining_max` tokens, so what is actually
        # needed is C summed (in log space) over a contiguous range of layers. Enumerate the
        # (lo, hi) pairs reachable from each token_range: (i, i + max_ - min_) while fewer
        # than min_ tokens have been generated, then (0, i) once the minimum is met.
        ranges = set()
        for token_range in token_ranges:
            min_tokens_, max_tokens_ = token_range
            for i in range(min_tokens_, -1, -1):
                ranges.add((i, i + max_tokens_ - min_tokens_))
            for i in range(max_tokens_ - min_tokens_ - 1, -1, -1):
                ranges.add((0, i))
        ranges = list(ranges)
        range_mask = torch.zeros(len(ranges), max_tokens+1, device=device)
        for idx, r in enumerate(ranges):
            range_mask[idx, torch.arange(r[0], r[1]+1)] = 1.0

        # One 0/1 row per range selects the layers of C it covers, so a single matmul in log
        # space performs all the logsumexps at once: C_cache[(lo, hi)] = logsumexp of C[lo:hi+1].
        C_shape = C.shape
        C = matmul_a_logb(range_mask, torch.flatten(C, start_dim=1, end_dim=2)) # num_ranges * (num_states * hidden_states)
        C = C.view(-1, C_shape[1], C_shape[2])
        C.nan_to_num_(neginf=neginf)

        C_cache = {}
        for idx, r in enumerate(ranges):
            C_cache[r] = C[idx]

        # initialize cache D -- prefix tuple -> the DFA state it leads to; extended one
        # token at a time as generation proceeds.
        D_cache = {tuple(prefix_ids): dfa_model.initial_state}

        self.A_cache = A_cache
        self.B_cache = B_cache
        self.C_cache = C_cache
        self.D_cache = D_cache

        self.prompt_ids = prompt_ids
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids

        self.temperature = temperature
        self.alpha = alpha
        self.token_ranges = token_ranges
        self.hmm_batch_size = hmm_batch_size

        self.dfa_model = dfa_model
        self.hmm_model = hmm_model


    def __call__(self, input_ids, scores):
        """Apply the constraint to one decoding step and return log-probabilities (not raw logits).

        input_ids: batch_size * seq_len, including the prompt -- which is stripped off here,
        since only the generated part is constrained. Sequences that have already emitted
        eos are finished, so they keep the base model's distribution untouched.
        """
        input_ids = input_ids[:,len(self.prompt_ids):].tolist()
        prefixes = [tuple(self.prefix_ids + x) for x in input_ids]

        if len(prefixes[0]) > 0:
            selected_idx = [i for i, prefix in enumerate(prefixes)
                if prefix[-1] != self.hmm_model.eos_token_id]
        else:
            selected_idx = [i for i, _ in enumerate(prefixes)]

        logits = torch.log_softmax(scores, dim=-1)

        if len(selected_idx) > 0:
            selected_prefixes = [prefixes[i] for i in selected_idx]
            if len(self.token_ranges) == 1:
                selected_token_ranges = [self.token_ranges[0] for _ in selected_idx]
            else:
                selected_token_ranges = [self.token_ranges[i] for i in selected_idx]

            hmm_batch_size = len(selected_idx) if self.hmm_batch_size is None else min(len(selected_idx), self.hmm_batch_size)
            # Dividing the constrained score by the unconstrained one (a subtraction in log
            # space) cancels the HMM's own estimate of P(prefix, next token), leaving just
            # log P(the constraint can still be satisfied | prefix, next token).
            hmm_logits, hmm_logits_ = self.compute_logits(selected_prefixes, selected_token_ranges, hmm_batch_size)
            hmm_logits -= hmm_logits_

            # ban special tokens that are not in the HMM
            if hmm_logits.shape[1] < logits.shape[1]:
                neginf = torch.full((hmm_logits.shape[0], logits.shape[1]-hmm_logits.shape[1]), -1e30, device=hmm_logits.device)
                hmm_logits = torch.cat((hmm_logits, neginf), dim=1)
            logits[selected_idx, :] += self.alpha * hmm_logits
            logits = torch.log_softmax(logits, dim=-1)

        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        return logits


    # compute logits for next_token
    def compute_logits(self, prefixes, token_ranges, batch_size):
        """Score every possible next token under the HMM, with and without the constraint.

        Returns (logits, logits_), both prefix_num * vocab_size and in log space:
            logits[p, w]  = log P(prefix p, next token = w, and the generation then walks
                the DFA into an accept state within p's token range and ends with the suffix)
            logits_[p, w] = log P(prefix p, next token = w)   -- the plain HMM marginal,
                used by the caller as the normalizing constant.

        batch_size -- how many prefixes to score at once; the intermediate tensor is
            batch_size * num_states * vocab_size, which is what makes chunking worthwhile.
        """
        device = self.hmm_model.alpha_exp.device
        neginf = -1e30
        neginf_cuda = neginf * torch.ones(1, device=device)

        suffix = self.suffix_ids
        generation_offset = len(self.prefix_ids)
        prefix_num = len(prefixes)
        prefix_lens = [len(prefix) for prefix in prefixes]

        VE_mask, EV_mask, T_mask = self.dfa_model.VE_mask, self.dfa_model.EV_mask, self.dfa_model.T_mask
        A_cache, B_cache, C_cache, D_cache = self.A_cache, self.B_cache, self.C_cache, self.D_cache
        alpha_exp, beta, gamma = self.hmm_model.alpha_exp, self.hmm_model.beta, self.hmm_model.gamma
        hidden_states, vocab_size = self.hmm_model.hidden_states, self.hmm_model.vocab_size

        # update prefix hidden states
        # Process prefixes that need cache updates
        
        # generation_offset -- where the generated tokens start, i.e. everything before it is
        # the fixed prefix, whose forward pass was already done in __init__.
        prefixes_to_update = [i for i, prefix_len in enumerate(prefix_lens) if prefix_len > generation_offset]
        if prefixes_to_update:
            # update A_cache for prefixes that need it -- extend the forward pass by the one
            # token sampled since the last step (emit with beta, then transition with alpha_exp)
            for prefix_idx in prefixes_to_update:
                prefix = prefixes[prefix_idx]
                if prefix not in A_cache:
                    if prefix[:-1] in A_cache:
                        A_prev = A_cache[prefix[:-1]]
                        log_prob = beta[:, prefix[-1]]
                        A_new = A_prev + log_prob
                        A_new = matmul_loga_b(A_new[None, :], alpha_exp).squeeze(0)
                        A_cache[prefix] = A_new
                    else: 
                        # For vllm, there might be output resumed_from_preemption. (temp remove from batch for by scheduler)
                        # For a few of those cases, we recalculate all missing prefixes
                        # Iteratively compute from the initial prefix
                        A_current = A_cache[tuple(self.prefix_ids)]
                        for token_idx in range(len(self.prefix_ids), len(prefix)):
                            token = prefix[token_idx]
                            log_prob = beta[:, token]
                            A_current = A_current + log_prob
                            A_current = matmul_loga_b(A_current[None, :], alpha_exp).squeeze(0)
                            # Cache intermediate results
                        A_cache[prefix] = A_current # only store the last one

            # update D_cache -- likewise advance the DFA by the newly sampled token
            for prefix_idx in prefixes_to_update:
                prefix = prefixes[prefix_idx]
                if prefix not in D_cache:
                    # Check if parent prefix exists in cache
                    parent_prefix = prefix[:-1]
                    if parent_prefix not in D_cache:
                        # Recalculate all missing intermediate prefixes from initial state
                        current_state = D_cache[tuple(self.prefix_ids)]
                        for token_idx in range(len(self.prefix_ids), len(parent_prefix)):
                            token = parent_prefix[token_idx]
                            current_state = self.dfa_model.next_state(current_state, token)
                            # Cache all intermediate results
                            intermediate_prefix = parent_prefix[:token_idx + 1]
                            D_cache[intermediate_prefix] = current_state
                    
                    next_state = self.dfa_model.next_state(D_cache[parent_prefix], prefix[-1])
                    D_cache[prefix] = next_state

        # Get A values for all prefixes
        A = torch.stack([A_cache[prefix] for prefix in prefixes], dim=0) # prefix_num * hidden_states

        # Clean up cache to limit its size
        self._cleanup_cache(prefixes)

        logits = torch.full((prefix_num, vocab_size), neginf, device=device)

        # gather the list of indices that has at least one more token left before suffix
        # Calculate generated tokens for each prefix individually
        generated_tokens_list = [prefix_len - generation_offset for prefix_len in prefix_lens]
        selected_idx = [prefix_idx for prefix_idx, generated_tokens in enumerate(generated_tokens_list)
            if token_ranges[prefix_idx][1] - generated_tokens > 0]
        selected_num = len(selected_idx)
        if len(selected_idx) > 0:
            for batch_idx in range(0, selected_num, batch_size):
                batch_size_ = min(batch_size, selected_num - batch_idx)
                selected_batch = selected_idx[batch_idx: batch_idx+batch_size_]

                A_batch = A[selected_batch] # batch_size_ * hidden_states

                prefixes_batch = [prefixes[i] for i in selected_batch]

                # pick the precomputed C layer-range matching how many tokens each prefix has
                # left; the -1s are because the token being scored right now consumes one.
                C_batch = []
                for prefix_idx in selected_batch:
                    min_tokens, max_tokens = token_ranges[prefix_idx]
                    generated_tokens = generated_tokens_list[prefix_idx]
                    remaining_tokens_max = max_tokens - generated_tokens
                    remaining_tokens_min = max(1, min_tokens - generated_tokens)
                    C_batch.append(C_cache[(remaining_tokens_min-1, remaining_tokens_max-1)])
                C_batch = torch.stack(C_batch, dim=0) # batch_size_ * num_states * hidden_states

                # past (A) meets future (C), still indexed by HMM state; then contract that
                # state away against the emission matrix to score each token of the vocabulary.
                # The result is indexed by the DFA state the next token would move us *into*.
                C = A_batch[:, None, :] + C_batch # batch_size_ * num_states * hidden_states

                C_shape = C.shape
                C = matmul_log(torch.flatten(C, start_dim=0, end_dim=1), beta) # (batch_size_ * num_states) * vocab_size
                C = C.view(C_shape[0], C_shape[1], -1) # batch_size_ * num_states * vocab_size

                # build the 0/1 legality mask for that same (state, token) grid, by chaining
                # the DFA's three incidence matrices: the edges leaving the prefix's current
                # state (VE_mask), where each of those edges lands (EV_mask), and which tokens
                # label it (T_mask). Entry (v, w) is 1 iff token w moves the DFA to state v.
                mask = torch.stack([VE_mask[D_cache[prefix]] for prefix in prefixes_batch], dim=0) # prefix_mask, batch_size_ * num_transitions
                mask = mask[:, :, None] * EV_mask[None, :, :] # batch_size_ * num_transitions * num_states
                mask = torch.transpose(mask, 1, 2) # batch_size_ * num_states * num_transitions

                mask_shape = mask.shape
                mask = torch.matmul(torch.flatten(mask, start_dim=0, end_dim=1), T_mask) # (batch_size_ * num_states) * vocab_size
                mask = mask.view(mask_shape[0], mask_shape[1], -1) # batch_size_ * num_states * vocab_size
                mask = torch.nan_to_num(torch.log(mask), neginf=neginf) # 0.0 for legal, neginf for illegal

                # zero out the illegal (state, token) pairs and sum over the DFA states, since
                # a token is legal along at most one edge out of the current state
                logits_batch = logsumexp(C + mask, dim=1) # batch_size_ * vocab_size

                logits[selected_batch, :] = logits_batch

        # if current prefix already ends with part/none of the suffix;
        # the loop above only covers futures that keep walking the DFA, so add the mass of
        # the "stop generating and emit the suffix" continuations on top of it. ends_at gives
        # the overlaps s at which that is legal, and A + B_cache[s] is the log-probability of
        # the prefix followed by the rest of the suffix.
        for prefix_idx, prefix in enumerate(prefixes):
            min_tokens, max_tokens = token_ranges[prefix_idx]
            offset_min = min_tokens + generation_offset
            offset_max = max_tokens + generation_offset
            offsets = ends_at(prefix, suffix,
                offset_min, D_cache, self.dfa_model)
            for offset in offsets:
                log_prob = logsumexp(A[prefix_idx] + B_cache[offset], dim=0)
                logits[prefix_idx, suffix[offset]] = torch.logaddexp(logits[prefix_idx, suffix[offset]], log_prob)

        # compute normalizing constant; no hmm mini-batch here
        # logits_[p, w] = log P(prefix p, next token = w) with no constraint attached; this
        # one is cheap (no num_states axis), so it is done for the whole batch at once.
        logits_ = matmul_log(A, beta)
        
        return logits, logits_
    
    def _cleanup_cache(self, prefixes):
        # print("Cleanup cache!")
        """Clean up cache that is not recently used to limit its size.

        A_cache holds one hidden_states vector per prefix and D_cache one DFA state, and both
        would otherwise grow by one entry per sequence per decoding step. Only the prefixes
        currently being decoded (plus the initial one) are ever read again, so everything else
        is dropped; ends_at recomputes any D_cache entry it still needs.
        """
        keys_to_keep_cache = [tuple(self.prefix_ids)] # always keep the initial prefix
        keys_to_keep_cache += [prefix for prefix in prefixes] # keep all current prefixes
        
        A_cache_size_limit = 0 # Always clear cache
        if len(self.A_cache) > A_cache_size_limit:
            for key in list(self.A_cache.keys()):
                if not (key in keys_to_keep_cache):
                    del self.A_cache[key]
        
        # We allow D_cache to grow very large since each entry is small
        D_cache_size_limit = 0
        if len(self.D_cache) > D_cache_size_limit:
            for key in list(self.D_cache.keys()):
                if not (key in keys_to_keep_cache):
                    del self.D_cache[key]
        # print(f"Deleted {del_num} entries from A_cache ({len(existing_keys)}) to {len(self.A_cache)}")

def extract_generated_ids(outputs, prompt_ids, suffix_ids, eos_token_id):
    """Strip the parts that were not generated from each sequence in `outputs`.

    Removes the prompt from the front and any trailing eos tokens, then removes the longest
    prefix of suffix_ids that the sequence ends with -- generation may stop anywhere inside
    the suffix, so the overlap length varies per sequence. Returns a list of token id tuples.
    """
    processed_outputs = []

    suffix_ids = tuple(suffix_ids)
    while len(suffix_ids) > 0 and suffix_ids[-1] == eos_token_id:
        suffix_ids = suffix_ids[:-1]
    prompt_ids = tuple(prompt_ids)

    for output_ids in outputs:
        output_ids = tuple(output_ids)
        while output_ids[-1] == eos_token_id:
            output_ids = output_ids[:-1]
        output_ids = output_ids[len(prompt_ids):]

        l = 0
        for k in range(1, min(len(output_ids), len(suffix_ids))+1):
            if output_ids[-k:] == suffix_ids[:k]:
                l = k
        end = None if l == 0 else -l

        output_ids = output_ids[:end]

        processed_outputs.append(output_ids)

    return processed_outputs


# suffix_logits_only: only use logits from the suffix_ids
# suffix_length_cap: set suffix_ids to suffix_ids[:suffix_length_cap] for ranking
# length_penalty: 0.0 --> rank by log-likelihood, 1.0 --> rank by perplexity
def rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids,
    suffix_logits_only=False, suffix_length_cap=None, length_penalty=1.0):
    """Sort candidate generations best-first by the base model's score for prompt + candidate + suffix.

    The HMM only approximates the base LLM, so the constrained samples are re-scored by the
    LLM itself. logits_mask picks which positions count towards the score (see the flags
    above); norms applies the length penalty, and the candidates come back sorted descending.
    """
    device = base_model.device
    suffix_ids = suffix_ids[:suffix_length_cap]

    # preprocessing input_ids
    input_ids, logits_mask = [], []
    for generated in generated_ids:
        input_ids.append(list(prompt_ids) + list(generated) + list(suffix_ids))
        if suffix_logits_only:
            logits_mask.append([0.0] * len(prompt_ids) + [0.0] * len(generated) + [1.0] * len(suffix_ids))
        else:
            logits_mask.append([0.0] * len(prompt_ids) + [1.0] * len(generated) + [1.0] * len(suffix_ids))

    max_len = max([len(x) for x in input_ids])
    input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
    input_ids = torch.tensor(input_ids, device=device)
    logits_mask = [x + [0.0] * (max_len - len(x)) for x in logits_mask]
    logits_mask = torch.tensor(logits_mask, device=device)

    # llm forward
    n, d = input_ids.shape
    with torch.no_grad():
        logits = base_model(input_ids).logits[:, :-1, :]
        logits = torch.log_softmax(logits, dim=-1)
        log_probs = logits[
            torch.arange(n)[:, None],
            torch.arange(d-1)[None, :],
            input_ids[:, 1:]]

    norms = torch.sum(logits_mask[:, 1:], dim=-1) ** length_penalty
    log_probs = torch.sum(log_probs * logits_mask[:, 1:], dim=-1) / norms

    generated_ids_sorted = [a for a, b in 
        sorted([(x, y) for x,y in zip(generated_ids, log_probs.tolist())], key=lambda x: x[1], reverse=True)]

    return generated_ids_sorted