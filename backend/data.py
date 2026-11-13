import re, numpy as np
NAMES = ["Suspicious links", "Keyword count", "Sender reputation", "CAPS ratio", "Exclamations"]
KEYS = ["free","winner","win","prize","cash","urgent","offer","click","buy","cheap","credit","loan",
        "viagra","lottery","guarantee","bonus","claim","limited","act now","congratulations",
        "password","verify","suspended"]
BAD_TLD = (".xyz", ".ru", ".top", ".click", ".biz", ".win")
GOOD = ("gmail.com", "outlook.com", ".edu", ".edu.in", ".gov", ".ac.in", ".org")

def extract(text, sender=""):
    """Turn a raw email into the 5 numeric features (all scaled to 0..1)."""
    t = text.lower()
    links = len(re.findall(r"https?://|www\.|bit\.ly", t))
    kw = sum(t.count(k) for k in KEYS)
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    rep, s = 0.5, sender.lower().strip()
    if "@" in s:
        d = s.split("@")[-1]; rep = 0.55
        if d.endswith(GOOD): rep = 0.9
        if d.endswith(BAD_TLD): rep = 0.1
        if re.search(r"\d{3,}", s) or d.count("-") > 1: rep = min(rep, 0.2)
    return np.array([min(links/5, 1), min(kw/8, 1), rep, min(caps/0.6, 1), min(text.count("!")/6, 1)])

def make_dataset(n=2000, seed=42):
    """Synthetic emails with overlapping classes (realistic, not perfectly separable)."""
    r = np.random.default_rng(seed); h = n // 2
    def gen(spam, m):
        if spam: l, k, rep, c, e = r.poisson(2.2,m), r.poisson(4,m), r.beta(2,5,m), r.beta(3,6,m), r.poisson(2.5,m)
        else:    l, k, rep, c, e = r.poisson(.5,m), r.poisson(1,m), r.beta(5,2,m), r.beta(1.5,9,m), r.poisson(.6,m)
        return np.c_[np.minimum(l/5,1), np.minimum(k/8,1), rep, np.minimum(c/.6,1), np.minimum(e/6,1)]
    X = np.r_[gen(0, h), gen(1, h)]; y = np.r_[np.zeros(h), np.ones(h)].astype(int)
    p = r.permutation(n); X, y = X[p], y[p]; s = int(.8 * n)
    return X[:s], y[:s], X[s:], y[s:]

def xor_data(n=200, seed=1):
    r = np.random.default_rng(seed); c = r.integers(0, 2, (n, 2))
    return c + r.normal(0, .12, (n, 2)), (c[:, 0] ^ c[:, 1])
