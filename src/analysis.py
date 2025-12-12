"""
src/analysis.py


Usage:
    python -m src.analysis
or
    python src/analysis.py
(from repository root)

Requirements:
    - numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm
    - qiskit, qiskit-machine-learning, qiskit-aer (for noisy sims; optional but recommended)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils import resample

# Try to import Qiskit kernels and simulators
try:
    from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel, FidelityQuantumKernel
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Try to import user's experiment class if present
TRY_IMPORT_EXPERIMENT = True
experiment_class = None
if TRY_IMPORT_EXPERIMENT:
    try:
        from src.run_qsvm_experiment import QSVMExperiment
        experiment_class = QSVMExperiment
    except Exception:
        experiment_class = None

# Output folder
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Utility helpers -----------------------------------------------------------
def set_seeds(seed=42):
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)

def pca_scatter(X, y, title, fname):
    p = PCA(n_components=2)
    Z = p.fit_transform(X)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=y, palette="deep", s=60)
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def save_confusion(cm, labels, fname, title="Confusion Matrix"):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def kernel_heatmap(K, fname, title="Kernel Gram Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(K, cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def kernel_alignment_score(K, y):
    # y mapped to +/-1 for binary; for multiclass compute one-vs-rest alignment average
    y = np.asarray(y)
    if len(np.unique(y)) == 2:
        yv = (y*2 - 1).reshape(-1,1)
        yyT = yv @ yv.T
        return float(np.sum(K * yyT) / np.sqrt(np.sum(K**2) * np.sum(yyT**2)))
    else:
        # multiclass: average alignment across classes
        classes = np.unique(y)
        vals = []
        for c in classes:
            yv = (y == c).astype(int).reshape(-1,1)
            yyT = yv @ yv.T
            vals.append(float(np.sum(K * yyT) / np.sqrt(np.sum(K**2) * np.sum(yyT**2))))
        return float(np.mean(vals))

def kernel_similarity(K1, K2):
    # Pearson correlation between flattened upper triangles
    iu = np.triu_indices_from(K1, k=1)
    v1 = K1[iu]; v2 = K2[iu]
    if np.std(v1)==0 or np.std(v2)==0: return 0.0
    return float(np.corrcoef(v1, v2)[0,1])

def paired_bootstrap_test(accA, accB, n_boot=5000, seed=42):
    # accA, accB are arrays of per-sample correctness (0/1) for same test set
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(accA)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(np.mean(accA[idx] - accB[idx]))
    diffs = np.array(diffs)
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return p, np.percentile(diffs, [2.5, 97.5])

# Core experiment runners --------------------------------------------------
def load_and_prep(binary=True, pca_components=4, test_size=0.2, random_state=42, scale=False):
    data = load_wine()
    X = data.data; y = data.target
    if binary:
        mask = (y == 0) | (y == 1)
        X = X[mask]; y = y[mask]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if pca_components:
        pca = PCA(n_components=pca_components)
        Xs = pca.fit_transform(Xs)
    if scale:
        Xs = (Xs - Xs.min()) / (Xs.max() - Xs.min()) * np.pi
    return train_test_split(Xs, y, test_size=test_size, random_state=random_state, stratify=y)

def compute_statevector_kernel(feature_map, X_train, X_test):
    # Uses FidelityStatevectorKernel if available
    if QISKIT_AVAILABLE:
        kernel = FidelityStatevectorKernel(feature_map=feature_map)
        K_train = kernel.evaluate(X_train, X_train)
        K_test = kernel.evaluate(X_test, X_train)
        return K_train, K_test
    else:
        raise RuntimeError("Qiskit not available for statevector kernel. Install qiskit and qiskit-machine-learning.")

def compute_noisy_kernel_featuremap(feature_map, X_train, X_test, noise_p=0.01, shots=1024):
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit Aer required for noisy sims.")
    # Build noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_p, 1), ["rx","ry","rz"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_p, 2), ["cz","cx"])
    backend = AerSimulator(noise_model=noise_model, shots=shots)
    # Build Gram via U(x_i)^\dagger U(x_j) measurement of all-zero
    n_train = len(X_train); n_test = len(X_test)
    feature_dim = X_train.shape[1]
    K_train = np.zeros((n_train, n_train))
    K_test = np.zeros((n_test, n_train))
    from qiskit import transpile
    for i in range(n_train):
        for j in range(n_train):
            circ_i = feature_map.assign_parameters({p: float(X_train[i, k]) for k, p in enumerate(feature_map.parameters)})
            circ_j = feature_map.assign_parameters({p: float(X_train[j, k]) for k, p in enumerate(feature_map.parameters)})
            circ = circ_i.inverse().compose(circ_j).measure_all(inplace=False)
            tq = transpile(circ, backend)
            job = backend.run(tq)
            res = job.result()
            counts = res.get_counts()
            K_train[i,j] = counts.get('0'*feature_dim, 0)/shots
    for i in range(n_test):
        for j in range(n_train):
            circ_i = feature_map.assign_parameters({p: float(X_test[i, k]) for k, p in enumerate(feature_map.parameters)})
            circ_j = feature_map.assign_parameters({p: float(X_train[j, k]) for k, p in enumerate(feature_map.parameters)})
            circ = circ_i.inverse().compose(circ_j).measure_all(inplace=False)
            tq = transpile(circ, backend)
            job = backend.run(tq)
            res = job.result()
            counts = res.get_counts()
            K_test[i,j] = counts.get('0'*feature_dim, 0)/shots
    return K_train, K_test

def run_full_analysis(save_report=True):
    set_seeds(42)
    # load data (binary recommended for QSVM experiments)
    X_train, X_test, y_train, y_test = load_and_prep(binary=True, pca_components=4, test_size=0.2, random_state=42)
    labels = np.unique(y_train).tolist()

    summary = []
    kernel_store = {}

    # 1) Classical baseline
    t0 = time.time()
    clf_rbf = SVC(kernel='rbf', gamma='scale')
    clf_rbf.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    preds_rbf = clf_rbf.predict(X_test)
    t_pred = time.time() - t0
    acc_rbf = accuracy_score(y_test, preds_rbf)
    f1_rbf = f1_score(y_test, preds_rbf, average='weighted')
    cm_rbf = confusion_matrix(y_test, preds_rbf)
    summary.append({
        "model":"RBF-SVM",
        "accuracy":acc_rbf,
        "f1":f1_rbf,
        "kernel_time_s":0.0,
        "svm_train_s":t_train,
        "svm_predict_s":t_pred
    })
    save_confusion(cm_rbf, labels, os.path.join(RESULTS_DIR, "cm_rbf.png"), title="Classical RBF Confusion Matrix")
    pca_scatter(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]), "PCA Scatter (dataset)", os.path.join(RESULTS_DIR, "pca_scatter.png"))

    # 2) QSVM experiments - use ZZ and Pauli, plus custom if present
    feature_maps = []
    fm_names = []

    if QISKIT_AVAILABLE:
        try:
            feature_maps.append(ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)); fm_names.append("ZZFeatureMap")
            feature_maps.append(PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, paulis=['X','Y','Z'])); fm_names.append("PauliFeatureMap")
        except Exception as e:
            print("Warning building builtin maps:", e)

    # If custom map available in src.run_qsvm_experiment, attempt to use it
    if experiment_class is not None:
        try:
            # instantiate to reuse custom map if implemented
            inst = experiment_class()
            # attempt to build a custom map (function name may vary)
            try:
                custom_map = inst.__dict__.get('custom_map') or inst.__dict__.get('feature_map') or None
            except Exception:
                custom_map = None
            # fallback: try to import from src.qsvm_feature_maps
            if custom_map is None:
                from src.qsvm_feature_maps import custom_param_map
                custom_map = custom_param_map(X_train.shape[1], reps=2, alpha=0.5)
            feature_maps.append(custom_map); fm_names.append("CustomParamMap")
        except Exception:
            pass

    # If Qiskit not available, attempt to skip QSVM
    if not QISKIT_AVAILABLE:
        print("Qiskit not available: skipping QSVM statevector/noisy computations. Install qiskit & qiskit-machine-learning for full analysis.")
    else:
        # Loop feature maps
        for fmap, name in zip(feature_maps, fm_names):
            print("Running QSVM with", name)
            # compute kernel & measure time
            t0 = time.time()
            try:
                K_train, K_test = compute_statevector_kernel(fmap, X_train, X_test)
            except Exception as e:
                # fallback to FidelityQuantumKernel (if earlier API differs)
                try:
                    from qiskit_machine_learning.kernels import FidelityQuantumKernel
                    kernel_obj = FidelityQuantumKernel(feature_map=fmap)
                    K_train = kernel_obj.evaluate(X_train, X_train)
                    K_test = kernel_obj.evaluate(X_test, X_train)
                except Exception as e2:
                    raise e2
            kt = time.time() - t0

            # Train SVM on precomputed kernel
            t0 = time.time()
            svm = SVC(kernel='precomputed')
            svm.fit(K_train, y_train)
            t_train = time.time() - t0
            t0 = time.time()
            preds = svm.predict(K_test)
            t_pred = time.time() - t0

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            cm = confusion_matrix(y_test, preds)

            # kernel alignment
            align = kernel_alignment_score(K_train, y_train)

            summary.append({
                "model": f"QSVM-{name}",
                "accuracy": float(acc),
                "f1": float(f1),
                "kernel_time_s": float(kt),
                "svm_train_s": float(t_train),
                "svm_predict_s": float(t_pred),
                "alignment": float(align)
            })

            # save kernel and confusion
            kernel_store[name] = (K_train, K_test)
            kernel_heatmap(K_train, os.path.join(RESULTS_DIR, f"kernel_{name}.png"), title=f"Kernel {name}")
            save_confusion(cm, labels, os.path.join(RESULTS_DIR, f"cm_{name}.png"), title=f"Confusion {name}")

    # 3) Noisy simulation for best feature map (if Aer available)
    noise_results = []
    if QISKIT_AVAILABLE:
        try:
            best_fm_name = max([s for s in summary if s["model"].startswith("QSVM")], key=lambda x: x["accuracy"])["model"].split("-",1)[1]
            best_idx = fm_names.index(best_fm_name)
            best_map = feature_maps[best_idx]
        except Exception:
            best_map = feature_maps[0] if len(feature_maps)>0 else None

        if best_map is not None and 'AerSimulator' in globals():
            noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
            for p in noise_levels:
                print(f"Running noisy sim at p={p}")
                try:
                    t0 = time.time()
                    Kt_noisy, Kte_noisy = compute_noisy_kernel_featuremap(best_map, X_train, X_test, noise_p=p, shots=1024)
                except Exception as e:
                    print("Noisy sim failed:", e)
                    break
                t_noisy = time.time() - t0
                svm_n = SVC(kernel='precomputed')
                svm_n.fit(Kt_noisy, y_train)
                preds_n = svm_n.predict(Kte_noisy)
                acc_n = accuracy_score(y_test, preds_n)
                f1_n = f1_score(y_test, preds_n, average='weighted')
                noise_results.append({"noise_p":p, "accuracy":acc_n, "f1":f1_n, "kernel_time_s":t_noisy})
            # produce noise plot
            if len(noise_results)>0:
                df_noise = pd.DataFrame(noise_results)
                plt.figure(figsize=(6,4))
                plt.plot(df_noise['noise_p'], df_noise['accuracy'], marker='o')
                plt.title("Noisy Simulator: Accuracy vs Depolarizing p")
                plt.xlabel("Depolarizing p"); plt.ylabel("Accuracy")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, "noise_accuracy.png"), dpi=200)
                plt.close()
    else:
        print("Skipping noisy sims — qiskit-aer not available.")

    # 4) Kernel similarity matrix among computed kernels
    if len(kernel_store) >= 2:
        names = list(kernel_store.keys())
        sim_mat = np.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(len(names)):
                sim_mat[i,j] = kernel_similarity(kernel_store[names[i]][0], kernel_store[names[j]][0])
        plt.figure(figsize=(5,4))
        sns.heatmap(sim_mat, xticklabels=names, yticklabels=names, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Kernel Similarity (corr of upper triangle)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "kernel_similarity.png"), dpi=200)
        plt.close()

    # 5) Runtime & accuracy bar charts
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)

    # bar: accuracy
    plt.figure(figsize=(6,4))
    sns.barplot(x="model", y="accuracy", data=df_summary)
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=200)
    plt.close()

    # stacked runtime bars (kernel_time + train + predict)
    if "kernel_time_s" in df_summary.columns:
        df_runtime = df_summary.set_index("model")[["kernel_time_s","svm_train_s","svm_predict_s"]].fillna(0)
        df_runtime.plot(kind="bar", stacked=False, figsize=(8,4), logy=True)
        plt.title("Runtime comparison (log scale)")
        plt.ylabel("Seconds (log)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "runtime_comparison.png"), dpi=200)
        plt.close()

    # 6) Paired bootstrap test against RBF baseline (if per-sample predictions available)
    # compute per-sample correctness arrays for best QSVM (if available)
    paired_test_results = {}
    try:
        # find first QSVM record and its predictions by recomputing (cheap)
        qsvm_rows = [r for r in summary if r["model"].startswith("QSVM")]
        if len(qsvm_rows)>0:
            name = qsvm_rows[0]["model"].split("-",1)[1]
            Kt_b, Kte_b = kernel_store[name]
            clf_b = SVC(kernel='precomputed'); clf_b.fit(Kt_b, y_train); preds_b = clf_b.predict(Kte_b)
            acc_a = (preds_b == y_test).astype(int)
            acc_b = (preds_rbf == y_test).astype(int)
            pval, ci = paired_bootstrap_test(acc_a, acc_b, n_boot=2000)
            paired_test_results = {"p_value": float(pval), "delta_ci_2.5": float(ci[0]), "delta_ci_97.5": float(ci[1])}
    except Exception:
        paired_test_results = {}

    # 7) Save report (Markdown)
    report_lines = ["# QSVM Analysis Report", "", "## Summary table", "", df_summary.to_markdown(index=False), ""]
    if len(noise_results)>0:
        report_lines += ["## Noisy Simulator Results", "", pd.DataFrame(noise_results).to_markdown(index=False), ""]
    if paired_test_results:
        report_lines += ["## Paired bootstrap test (best QSVM vs RBF)", "", json.dumps(paired_test_results, indent=2), ""]
    report_lines += ["## Figures", "", "- accuracy_comparison.png", "- runtime_comparison.png", "- kernel_similarity.png", "- noise_accuracy.png", "- pca_scatter.png", ""]

    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write("\n".join(report_lines))

    print("All results saved to", RESULTS_DIR)
    return df_summary, kernel_store, noise_results

if __name__ == "__main__":
    df_summary, kernel_store, noise_results = run_full_analysis(save_report=True)