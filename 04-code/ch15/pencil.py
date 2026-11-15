"""Chapter 15: the forward pass of the hand-sized Tiny as a person with a pencil would do it.
Plain Python floats and math.sqrt / math.exp only: no NumPy, no eps inside the root, no subtraction
of the largest score before exp, and (if rounded=True) every number written down to three decimal
places before it is used again, as on paper.  trace() returns every intermediate by name."""
import math

def rnd(x, on):
    return round(x, 3) if on else x

def rms_norm(row, on):
    r = math.sqrt(sum(a * a for a in row) / len(row))
    return [rnd(a / r, on) for a in row], rnd(r, on)

def times(row, W, on):              # row (d) times table W (d x k)
    return [rnd(sum(row[i] * W[i][j] for i in range(len(row))), on) for j in range(len(W[0]))]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def trace(E, P, ids, tables, n_layers=2, rounded=False, positions=True):
    """tables: dict with W_q, W_k, W_v, W_o, W_1, b_1, W_2, b_2 as nested lists; E: token rows; P: seat rows."""
    on = rounded
    T = {}
    n = len(ids)
    tok = [list(E[i]) for i in ids]
    pos = [list(P[s]) if positions else [0.0, 0.0] for s in range(n)]
    X = [[rnd(a + b, on) for a, b in zip(t, p)] for t, p in zip(tok, pos)]
    T["tok"], T["pos"], T["x0"] = tok, pos, X
    for l in range(n_layers):
        L = {}
        N1 = [rms_norm(x, on) for x in X]
        L["rms_1"] = [r for _, r in N1]; Xn = [v for v, _ in N1]; L["ln_1"] = Xn
        Q = [times(x, tables["W_q"], on) for x in Xn]
        K = [times(x, tables["W_k"], on) for x in Xn]
        V = [times(x, tables["W_v"], on) for x in Xn]
        L["q"], L["k"], L["v"] = Q, K, V
        root = math.sqrt(len(Q[0]))
        S = [[rnd(dot(Q[i], K[j]), on) for j in range(n)] for i in range(n)]
        Ss = [[rnd(S[i][j] / root, on) for j in range(n)] for i in range(n)]
        A = []
        for i in range(n):
            e = [rnd(math.exp(Ss[i][j]), on) if j <= i else 0.0 for j in range(n)]
            tot = sum(e)
            A.append([rnd(x / tot, on) for x in e])
        L["scores"], L["scaled"], L["shares"] = S, Ss, A
        B = [[rnd(sum(A[i][j] * V[j][c] for j in range(n)), on) for c in range(2)] for i in range(n)]
        O = [times(b, tables["W_o"], on) for b in B]
        L["blend"], L["att"] = B, O
        A1 = [[rnd(x + o, on) for x, o in zip(xr, orow)] for xr, orow in zip(X, O)]
        L["add_1"] = A1
        N2 = [rms_norm(x, on) for x in A1]
        L["rms_2"] = [r for _, r in N2]; An = [v for v, _ in N2]; L["ln_2"] = An
        U = [[rnd(u + b, on) for u, b in zip(times(x, tables["W_1"], on), tables["b_1"])] for x in An]
        H = [[max(0.0, u) for u in row] for row in U]
        F = [[rnd(f + b, on) for f, b in zip(times(h, tables["W_2"], on), tables["b_2"])] for h in H]
        Y = [[rnd(a + f, on) for a, f in zip(ar, fr)] for ar, fr in zip(A1, F)]
        L["match"], L["active"], L["ff"], L["add_2"] = U, H, F, Y
        T[f"layer{l}"] = L
        X = Y
    T["top"] = X
    NF = [rms_norm(x, on) for x in X]
    T["rms_f"] = [r for _, r in NF]; Z = [v for v, _ in NF]; T["ln_f"] = Z
    SC = [[rnd(dot(z, e), on) for e in E] for z in Z]
    SH = []
    for row in SC:
        e = [math.exp(s) for s in row]
        tot = sum(e)
        SH.append([rnd(x / tot, on) for x in e])
    T["out_scores"], T["out_shares"] = SC, SH
    return T
