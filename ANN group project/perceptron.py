"""Single-layer perceptron written from scratch (NumPy only)."""
import numpy as np

class Perceptron:
    def __init__(s, n, lr=0.1):
        s.w = np.zeros(n); s.b = 0.0; s.lr = lr
    def z(s, X): return X @ s.w + s.b                       # weighted sum
    def predict(s, X): return (s.z(X) >= 0).astype(int)     # step activation
    def update(s, x, t):
        """Perceptron learning rule: w <- w + lr*(t-y)*x ; b <- b + lr*(t-y)"""
        y = int(x @ s.w + s.b >= 0); e = int(t) - y
        s.w = s.w + s.lr * e * x; s.b += s.lr * e
        return y, e
    def fit(s, X, t, epochs=30, seed=0):
        rng = np.random.default_rng(seed); hist = []; best = (-1, None, None)
        for _ in range(epochs):
            for i in rng.permutation(len(X)): s.update(X[i], t[i])
            acc = float((s.predict(X) == t).mean()); hist.append(acc)
            if acc > best[0]: best = (acc, s.w.copy(), s.b)    # "pocket" variant: keep best weights
            if acc == 1.0: break
        if best[1] is not None:
            s.w, s.b = best[1], best[2]
        return hist

def metrics(t, p):
    tp = int(((t == 1) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / len(t), precision=pr,
                recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9))

def logreg(X, y, it=800, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        g = 1 / (1 + np.exp(-(X @ w + b))) - y
        w -= lr * X.T @ g / len(y); b -= lr * g.mean()
    return w, b
