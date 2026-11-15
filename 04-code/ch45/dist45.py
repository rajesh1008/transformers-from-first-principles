"""Chapter 45: three ways to split training, run for real on one CPU with four processes (torch.distributed, gloo backend).
Model: Chapter 44's adder (add.py, two digits, 4 blocks, width 128, 801,552 learned numbers), 32-bit, seed 1865.
1. Data parallel: 4 ranks x 32 examples, gradients averaged by all-reduce, against 1 process x the same 128 examples.
   Compares gradients after one step and learned numbers after 20 AdamW steps.
2. Tensor parallel: block 0's feed-forward layer split four ways (first table by columns, second by rows); each rank
   computes a quarter and one all-reduce sums them.  Compares with the unsplit layer.
3. Pipeline: the four blocks on four ranks, activations sent forward rank to rank; compares the logits with one process.
Also counts the bytes each rank sends in each case.  Writes outputs/dist45.json."""
import os, sys, json, numpy as np, torch, torch.distributed as dist, torch.multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ch44"))
os.environ.update(DIGITS="2", FORMAT="pad"); import add
W = 4; STEPS = 20
def make():
    torch.manual_seed(1865); return add.Net()
def batches():
    tr, _ = add.pairs(); rng = np.random.default_rng(1865); return [add.batch(tr, rng) for _ in range(STEPS)]
def loss_of(m, X, M):
    lg = m(X[:, :-1]); y = X[:, 1:].clone(); y[~M[:, 1:]] = -100
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum"), int(M[:, 1:].sum())
def reference():
    m = make(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); B = batches(); g0 = None
    for s, (X, M) in enumerate(B):
        L, n = loss_of(m, X, M); opt.zero_grad(); (L / n).backward()
        if s == 0: g0 = torch.cat([p.grad.flatten() for p in m.parameters()]).clone()
        opt.step()
    return m, g0
def worker(rank, q):
    torch.set_num_threads(1); os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29545")
    dist.init_process_group("gloo", rank=rank, world_size=W); R = {}
    # 1. data parallel: each rank takes its quarter of every batch; the token count is summed so the mean matches exactly
    m = make(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); sent = 0
    for s, (X, M) in enumerate(batches()):
        Xr, Mr = X[rank * 32:(rank + 1) * 32], M[rank * 32:(rank + 1) * 32]
        L, n = loss_of(m, Xr, Mr); nt = torch.tensor([float(n)]); dist.all_reduce(nt)
        opt.zero_grad(); (L / nt.item()).backward()
        for p in m.parameters():
            dist.all_reduce(p.grad); sent += p.grad.numel() * 4 * 2 * (W - 1) / W   # ring all-reduce: 2(W-1)/W of the buffer per rank
        if s == 0: g = torch.cat([p.grad.flatten() for p in m.parameters()])
        opt.step()
    R["dp_grad0"] = g; R["dp_params"] = torch.cat([p.detach().flatten() for p in m.parameters()]); R["dp_bytes_per_step"] = sent / STEPS
    # 2. tensor parallel: block 0's feed-forward layer, a quarter of the hidden units per rank
    m = make(); ff = m.h[0].ff; D, H = 128, 512; h = H // W; x = torch.randn(64, D, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        W1 = ff[0].weight[rank * h:(rank + 1) * h]; b1 = ff[0].bias[rank * h:(rank + 1) * h]; W2 = ff[2].weight[:, rank * h:(rank + 1) * h]
        part = torch.nn.functional.gelu(x @ W1.T + b1, approximate="tanh") @ W2.T
        if rank == 0: part = part + ff[2].bias
        dist.all_reduce(part); R["tp_out"] = part; R["tp_bytes"] = part.numel() * 4 * 2 * (W - 1) / W
        R["tp_numbers_held"] = W1.numel() + b1.numel() + W2.numel() + (ff[2].bias.numel() if rank == 0 else 0)
    # 3. pipeline (forward only): rank r runs block r; rank 0 also embeds, rank 3 also normalises and reads out
    m = make(); X, M = batches()[0]; ids = X[:, :-1]
    with torch.no_grad():
        if rank == 0: a = m.E(ids) + m.P[:, :ids.shape[1]]
        else: a = torch.empty(ids.shape[0], ids.shape[1], 128); dist.recv(a, src=rank - 1)
        a = m.h[rank](a)
        if rank < W - 1: dist.send(a, dst=rank + 1); R["pp_bytes"] = a.numel() * 4
        else: R["pp_logits"] = m.out(m.lnf(a)); R["pp_bytes"] = 0
    dist.barrier(); q.put((rank, {k: (v.numpy().copy() if torch.is_tensor(v) else v) for k, v in R.items()})); dist.destroy_process_group()
if __name__ == "__main__":
    ref, g_ref = reference(); p_ref = torch.cat([p.detach().flatten() for p in ref.parameters()])
    q = mp.get_context("spawn").SimpleQueue(); ps = [mp.get_context("spawn").Process(target=worker, args=(r, q)) for r in range(W)]
    [p.start() for p in ps]; out = dict(q.get() for _ in range(W)); [p.join() for p in ps]
    out = {r: {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()} for r, d in out.items()}
    m = make(); X, M = batches()[0]; x = torch.randn(64, 128, generator=torch.Generator().manual_seed(7))
    with torch.no_grad(): full_ff = m.h[0].ff(x); full_logits = m(X[:, :-1])
    n = sum(p.numel() for p in m.parameters())
    res = dict(learned=n,
        dp_grad_maxdiff=float((out[0]["dp_grad0"] - g_ref).abs().max()), dp_grad_scale=float(g_ref.abs().max()),
        dp_ranks_agree=float(max((out[r]["dp_params"] - out[0]["dp_params"]).abs().max() for r in range(W))),
        dp_param_maxdiff_20=float((out[0]["dp_params"] - p_ref).abs().max()), dp_bytes_per_rank_per_step=out[0]["dp_bytes_per_step"],
        tp_maxdiff=float((out[0]["tp_out"] - full_ff).abs().max()), tp_bytes_per_rank=out[0]["tp_bytes"],
        tp_numbers_per_rank=[out[r]["tp_numbers_held"] for r in range(W)], ff_numbers=sum(p.numel() for p in m.h[0].ff.parameters()),
        pp_maxdiff=float((out[W - 1]["pp_logits"] - full_logits).abs().max()), pp_bytes_per_send=out[0]["pp_bytes"])
    print(f"learned numbers {n:,}; four processes, gloo, one thread each")
    print(f"1. data parallel: first-step gradient, 4 x 32 vs 1 x 128: largest difference {res['dp_grad_maxdiff']:.2e} (largest gradient {res['dp_grad_scale']:.2e})")
    print(f"   after {STEPS} AdamW steps: ranks agree to {res['dp_ranks_agree']:.1e}; against one process {res['dp_param_maxdiff_20']:.2e}")
    print(f"   bytes sent per rank per step (ring all-reduce of every gradient): {res['dp_bytes_per_rank_per_step']:,.0f}")
    print(f"2. tensor parallel, block 0 feed-forward ({res['ff_numbers']:,} numbers; per rank {res['tp_numbers_per_rank']}): largest difference {res['tp_maxdiff']:.2e}; bytes per rank per all-reduce {res['tp_bytes_per_rank']:,.0f} for 64 rows")
    print(f"3. pipeline, one block per rank: largest logit difference {res['pp_maxdiff']:.2e}; bytes per hand-off {res['pp_bytes_per_send']:,} (128 sequences x 31 positions x 128 x 4)")
    json.dump(res, open(os.path.join(HERE, "outputs", "dist45.json"), "w"), indent=1)
