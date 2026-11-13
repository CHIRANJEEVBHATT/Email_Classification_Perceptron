"""Train the single-layer perceptron on the official UCI Spambase dataset (57 features)."""
import os
import numpy as np
from perceptron import Perceptron, metrics, logreg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "spambase.data")

SPAMBASE_FEATURE_NAMES = [
    # 48 word frequency features
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference",
    # 6 character frequency features
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#",
    # 3 capital run length features
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"
]

def load_and_preprocess(filepath=DATA_PATH, test_size=0.2, seed=42):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Spambase data not found at {filepath}")
    
    print(f"Loading data from: {filepath}")
    raw_data = np.loadtxt(filepath, delimiter=",")
    X, y = raw_data[:, :-1], raw_data[:, -1].astype(int)
    print(f"Total samples: {len(X)} (Features: {X.shape[1]}, Spam: {np.sum(y == 1)}, Ham: {np.sum(y == 0)})")

    # Shuffle dataset to ensure representative class distributions in splits
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Standardize features (zero mean, unit variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    X_norm = (X - mean) / std

    # Train / Test split
    split_idx = int((1.0 - test_size) * len(X))
    X_train, y_train = X_norm[:split_idx], y[:split_idx]
    X_test, y_test = X_norm[split_idx:], y[split_idx:]
    
    return (X_train, y_train), (X_test, y_test), (mean, std)

def train_spambase(epochs=50, lr=0.01):
    (X_tr, y_tr), (X_te, y_te), (mean, std) = load_and_preprocess()

    print(f"\n--- Training Perceptron (epochs={epochs}, learning_rate={lr}) ---")
    model = Perceptron(n=X_tr.shape[1], lr=lr)
    history = model.fit(X_tr, y_tr, epochs=epochs, seed=42)

    train_m = metrics(y_tr, model.predict(X_tr))
    test_m = metrics(y_te, model.predict(X_te))

    # Baseline comparison: Logistic Regression
    lw, lb = logreg(X_tr, y_tr, it=800, lr=0.5)
    logreg_test_acc = float(((X_te @ lw + lb >= 0) == y_te).mean())

    print("\n=================== Evaluation Results ===================")
    print(f"Training Accuracy:   {train_m['accuracy'] * 100:.2f}%")
    print(f"Test Accuracy:       {test_m['accuracy'] * 100:.2f}% (LogReg Baseline: {logreg_test_acc * 100:.2f}%)")
    print(f"Test Precision:      {test_m['precision'] * 100:.2f}%")
    print(f"Test Recall:         {test_m['recall'] * 100:.2f}%")
    print(f"Test F1-Score:       {test_m['f1'] * 100:.2f}%")
    print("\nConfusion Matrix (Test Set):")
    print(f"  True Negative (Ham):  {test_m['tn']:4d} | False Positive: {test_m['fp']:4d}")
    print(f"  False Negative:       {test_m['fn']:4d} | True Positive (Spam):  {test_m['tp']:4d}")

    # Top features by absolute weight
    top_indices = np.argsort(np.abs(model.w))[::-1][:10]
    print("\nTop 10 Most Influential Features:")
    for rank, idx in enumerate(top_indices, 1):
        feat_name = SPAMBASE_FEATURE_NAMES[idx]
        direction = "SPAM (+)" if model.w[idx] > 0 else "HAM (-)"
        print(f"  {rank:2d}. {feat_name:<30} weight: {model.w[idx]:+.4f} [{direction}]")

    # Save model weights and standardization params
    save_path = os.path.join(HERE, "data", "spambase_perceptron.npz")
    np.savez(save_path, w=model.w, b=model.b, mean=mean, std=std, hist=history)
    print(f"\nTrained model weights saved to: {save_path}")
    return model, test_m

if __name__ == "__main__":
    train_spambase()

