"""
hmm_lab.py — understand ctrlg/hmm.py by running it small and checking it exactly.

Put this next to the real hmm.py (or `pip install -e .` the repo) and run:
    python hmm_lab.py

Everything below uses the ACTUAL HMM class from the repo. Nothing is reimplemented
except the brute-force ground truth, which enumerates all h**n state paths.

Exercises are numbered. Run them in order; each one is ~15 lines and answers one
question you probably have. Change the numbers at the top and re-run.
"""

import itertools
import torch

from hmm import HMM  # the real thing, unmodified

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

H, V, N, B = 3, 5, 4, 2          # hidden states, vocab, seq len, batch
EOS = V - 1


# --------------------------------------------------------------------------
# ground truth: enumerate every state path. only tractable because H**N = 81.
# --------------------------------------------------------------------------
def brute(model, x):
    """x: list of token ids, -1 = missing. Returns everything, exactly."""
    a = model.alpha_exp.double()
    b = model.beta.double().exp()
    g = torch.softmax(model.gamma.double(), dim=0)

    def emit(i, t):
        return 1.0 if x[t] == -1 else b[i, x[t]].item()   # missing -> factor 1

    weights = {}
    for path in itertools.product(range(H), repeat=len(x)):
        w = g[path[0]].item() * emit(path[0], 0)
        for t in range(1, len(x)):
            w *= a[path[t - 1], path[t]].item() * emit(path[t], t)
        weights[path] = w

    Z = sum(weights.values())                                   # p(x)
    post = torch.zeros(len(x), H, dtype=torch.float64)          # p(z_t | x)
    xi = torch.zeros(len(x) - 1, H, H, dtype=torch.float64)     # p(z_t,z_t+1 | x)
    for path, w in weights.items():
        for t in range(len(x)):
            post[t, path[t]] += w / Z
        for t in range(len(x) - 1):
            xi[t, path[t], path[t + 1]] += w / Z
    return Z, post, xi


def banner(msg):
    print("\n" + "=" * 68 + "\n" + msg + "\n" + "=" * 68)


model = HMM(hidden_states=H, vocab_size=V, eos_token_id=EOS)
x = torch.tensor([[1, 3, 0, 2],
                  [4, 4, 1, 0]])


# --------------------------------------------------------------------------
# 1. WHAT SHAPE IS EVERYTHING?  Print the ys list and read the ordering off it.
# --------------------------------------------------------------------------
banner("1. forward() returns a list. What is in it?")
ys = model.forward(x)
print(f"len(ys) = {len(ys)}   (seq_len={N}, so one layer per t, plus the root)")
for k, y in enumerate(ys):
    t = N - 1 - k if k < N else "root"
    print(f"  ys[{k}]  shape {tuple(y.shape)}   <- t = {t}")
print("\nys[-1] (log p(x) per sequence):", ys[-1])
print("ys[-2] is the t=0 layer -> this is what backward() calls probs[-2]")


# --------------------------------------------------------------------------
# 2. IS THE INVARIANT TRUE?  ys[k][i,b] should be log p(x_>=t | z_t = i).
#    Check the easiest one: the base case is just the last emission.
# --------------------------------------------------------------------------
banner("2. Check the invariant at the base case (t = n-1)")
base = ys[0]                       # t = n-1
expect = model.beta[:, x[:, -1]]   # log p(x_n | z_n = i), h x b
print("ys[0]           :\n", base)
print("beta[:, x[:,-1]]:\n", expect)
print("match:", torch.allclose(base, expect))


# --------------------------------------------------------------------------
# 3. IS log p(x) RIGHT?  Compare against enumerating all 81 paths.
# --------------------------------------------------------------------------
banner("3. forward() vs brute force: does log p(x) match?")
for row in range(B):
    Z, _, _ = brute(model, x[row].tolist())
    print(f"  seq {row}: forward={ys[-1][row].item():+.6f}   "
          f"brute={torch.log(torch.tensor(Z)).item():+.6f}")


# --------------------------------------------------------------------------
# 4. WHAT IS A FLOW?  Run backward() and compare every accumulator to the
#    brute-force posterior. This is the exercise that makes it click.
# --------------------------------------------------------------------------
banner("4. backward() vs brute force: are the flows really EM counts?")
alpha_flow = torch.zeros(H, H)
beta_flow = torch.zeros(V + 1, H)
gamma_flow = torch.zeros(H)
model.backward(x, model.forward(x), alpha_flow, beta_flow, gamma_flow)

bf = [brute(model, x_.tolist()) for x_ in x]

# gamma_flow = sum over batch of p(z_1 | x)
g_true = sum(post[0] for _, post, _ in bf)
print("gamma_flow :", gamma_flow)
print("brute      :", g_true.float())

# alpha_flow is xi WITHOUT its alpha factor -> multiply it back in
a_true = sum(xi.sum(0) for _, _, xi in bf)
print("\nalpha_flow * alpha_exp:\n", alpha_flow * model.alpha_exp)
print("brute sum_t xi:\n", a_true.float())
print("max abs diff:", (alpha_flow * model.alpha_exp - a_true.float()).abs().max().item())

# beta_flow[token, i] = expected times state i emitted that token
b_true = torch.zeros(V + 1, H)
for row, (_, post, _) in enumerate(bf):
    for t in range(N):
        b_true[x[row, t]] += post[t].float()
print("\nbeta_flow:\n", beta_flow)
print("brute:\n", b_true)


# --------------------------------------------------------------------------
# 5. WHAT DOES -1 ACTUALLY DO?  A missing token should give exactly the
#    marginal probability of the observed positions.
# --------------------------------------------------------------------------
banner("5. The MISSING token (-1) marginalizes, it does not pad")
x_miss = torch.tensor([[1, -1, 0, 2]])
ll_miss = model.forward(x_miss)[-1].item()

# marginalize by hand: sum p(x) over every value the middle token could take
total = 0.0
for v in range(V):
    x_v = torch.tensor([[1, v, 0, 2]])
    total += torch.exp(model.forward(x_v)[-1]).item()
print(f"forward with -1        : {ll_miss:+.6f}")
print(f"log sum_v forward(v)   : {torch.log(torch.tensor(total)).item():+.6f}")
print("-> identical, because sum_x p(x|z) = 1 for every z")


# --------------------------------------------------------------------------
# 6. DOES EM ACTUALLY WORK?  The only end-to-end test that matters:
#    log-likelihood must increase monotonically. This is the M-step from
#    distillation/train_hmm.py, stripped of the distributed-training noise.
# --------------------------------------------------------------------------
banner("6. Five EM steps. Log-likelihood must go up, every time.")
data = torch.randint(0, V, (64, N))
m = HMM(hidden_states=H, vocab_size=V, eos_token_id=EOS)

for step in range(6):
    ll = m.loglikelihood(data, batch_size=16).item()
    print(f"  step {step}: total log-likelihood = {ll:11.4f}")

    a_f = torch.zeros(H, H); b_f = torch.zeros(V + 1, H); g_f = torch.zeros(H)
    m.backward(data, m.forward(data), a_f, b_f, g_f)

    a_f.mul_(m.alpha_exp)                 # <- the factor ib_ib_bj_to_ij left out
    b_f = b_f[:V].t().contiguous()        # drop the MISSING row, -> h x V
    eps = 1e-8                            # the trainer's pseudocount
    m.update_params(
        (a_f + eps) / (a_f + eps).sum(-1, keepdim=True),
        torch.log((b_f + eps) / (b_f + eps).sum(-1, keepdim=True)),
        torch.log((g_f + eps) / (g_f + eps).sum()),
    )

print("\nIf that column is monotonically increasing, you have understood the file.")