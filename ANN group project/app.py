import os, numpy as np
from flask import Flask, jsonify, request, send_from_directory
from data import NAMES, extract, make_dataset, xor_data
from perceptron import Perceptron, metrics, logreg

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
Xtr, ytr, Xte, yte = make_dataset()

def train(lr=.1, epochs=30, cols=None):
    cols = cols or list(range(5)); m = Perceptron(len(cols), lr)
    return m, m.fit(Xtr[:, cols], ytr, epochs)
M, HIST = train()
LR = {"lr": .1}

@app.get("/")
def index():
    if os.path.exists(os.path.join(HERE, "index.html")):
        return send_from_directory(HERE, "index.html")
    return send_from_directory(os.path.join(HERE, "..", "frontend"), "index.html")

@app.get("/api/info")
def info(): return jsonify(names=NAMES, w=M.w.tolist(), b=float(M.b), hist=HIST, lr=LR["lr"])

@app.post("/api/train")
def retrain():
    global M, HIST
    d = request.get_json(silent=True) or {}
    LR["lr"] = float(d.get("lr", .1))
    M, HIST = train(LR["lr"], int(d.get("epochs", 30)))
    return info()

@app.post("/api/classify")
def classify():
    d = request.get_json(silent=True) or {}
    x = extract(d.get("text", ""), d.get("sender", ""))
    z = float(M.z(x)); return jsonify(x=x.tolist(), z=z, spam=bool(z >= 0), contrib=(M.w * x).tolist())

@app.post("/api/evade")
def evade():
    """Adversarial spammer: greedily nudge features in the 'legit' direction until the filter flips."""
    d = request.get_json(silent=True) or {}
    x = np.array(d.get("x", [0, 0, 0, 0, 0]), float); path = []
    for _ in range(40):
        if M.z(x) < 0: break
        room = np.where(M.w > 0, x, 1 - x) * np.abs(M.w)
        j = int(room.argmax())
        if room[j] <= 1e-9: break
        x[j] = max(0, x[j] - .1) if M.w[j] > 0 else min(1, x[j] + .1)
        path.append(dict(j=j, v=float(x[j]), z=float(M.z(x))))
    return jsonify(path=path, evaded=bool(M.z(x) < 0))

@app.get("/api/trace")
def trace():
    """Step-by-step learning on 2 features (keywords, reputation) so the boundary can be drawn."""
    lr = float(request.args.get("lr", .2)); idx = np.random.default_rng(3).choice(len(Xtr), 30, replace=False)
    P, T = Xtr[idx][:, [1, 2]], ytr[idx]; m = Perceptron(2, lr); steps = []
    for ep in range(4):
        for i in range(30):
            wb, bb = m.w.copy(), m.b; y, e = m.update(P[i], T[i])
            steps.append(dict(i=i, y=y, e=e, wb=wb.tolist(), bb=float(bb), wa=m.w.tolist(), ba=float(m.b)))
    return jsonify(points=P.tolist(), labels=T.tolist(), steps=steps)

@app.get("/api/insights")
def insights():
    z = M.z(Xte); mt = metrics(yte, M.predict(Xte)); mtr = metrics(ytr, M.predict(Xtr))
    th = np.r_[z.min() - 1, np.quantile(z, np.linspace(0, 1, 101)), z.max() + 1]
    roc = sorted([(float(((z >= t) & (yte == 0)).sum() / (yte == 0).sum()),
                   float(((z >= t) & (yte == 1)).sum() / (yte == 1).sum())) for t in th])
    auc = sum((roc[i+1][0] - roc[i][0]) * (roc[i+1][1] + roc[i][1]) / 2 for i in range(len(roc) - 1))
    abl = []
    for j in range(5):
        cols = [c for c in range(5) if c != j]; m2, _ = train(LR["lr"], 30, cols)
        abl.append(float((m2.predict(Xte[:, cols]) == yte).mean()))
    lw, lb = logreg(Xtr, ytr); lacc = float(((Xte @ lw + lb >= 0) == yte).mean())
    X, y = xor_data(); pm = Perceptron(2, .1); pm.fit(X, y, 50); xw, xb = logreg(X, y)
    out = dict(test=mt, train=mtr, roc=roc, auc=auc, ablation=abl, logreg=lacc, names=NAMES, w=M.w.tolist(),
               xor=dict(X=X.tolist(), y=y.tolist(), w=pm.w.tolist(), b=float(pm.b),
                        pacc=float((pm.predict(X) == y).mean()), lacc=float(((X @ xw + xb >= 0) == y).mean())))
    sb = os.path.join(HERE, "data", "spambase.data")          # optional: drop UCI Spambase here
    if os.path.exists(sb):
        D = np.loadtxt(sb, delimiter=","); A, t = D[:, :-1], D[:, -1].astype(int)
        rng = np.random.default_rng(42); p = rng.permutation(len(A)); A, t = A[p], t[p]
        A = (A - A.mean(0)) / (A.std(0) + 1e-9); k = int(.8 * len(A)); sm = Perceptron(A.shape[1], .01)
        sm.fit(A[:k], t[:k], 40); out["spambase"] = metrics(t[k:], sm.predict(A[k:]))
    return jsonify(out)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
