import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.utils import QuantumInstance

import matplotlib.pyplot as plt
import seaborn as sns
import os


class QSVMExperiment:
    def __init__(self):
        self.backend = BasicAer.get_backend("statevector_simulator")
        self.quantum_instance = QuantumInstance(self.backend)

    def load_and_prepare_data(self):
        data = load_wine()
        X, y = data.data, data.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Reduce to 4 features for QSVM
        X = X[:, :4]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def classical_svm(self, X_train, X_test, y_train, y_test):
        model = SVC(kernel="rbf")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        return accuracy_score(y_test, preds), confusion_matrix(y_test, preds)

    def qsvm(self, feature_map, X_train, X_test, y_train, y_test):
        fidelity = FidelityQuantumKernel(feature_map=feature_map)

        K_train = fidelity.evaluate(X_train)
        K_test = fidelity.evaluate(X_test, X_train)

        svm = SVC(kernel="precomputed")
        svm.fit(K_train, y_train)

        preds = svm.predict(K_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        return acc, cm, K_train

    def save_heatmap(self, matrix, filename):
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, cmap="viridis")
        plt.title("Kernel Matrix")
        plt.savefig(filename, dpi=300)
        plt.close()

    def run(self):
        print("🚀 Running QSVM vs Classical SVM Experiment...")

        X_train, X_test, y_train, y_test = self.load_and_prepare_data()

        # Classical SVM
        c_acc, c_cm = self.classical_svm(X_train, X_test, y_train, y_test)
        print(f"Classical SVM Accuracy: {c_acc:.4f}")

        # QSVM with ZZFeatureMap
        zz_map = ZZFeatureMap(feature_dimension=4, reps=2)
        q_acc_zz, q_cm_zz, K_zz = self.qsvm(
            zz_map, X_train, X_test, y_train, y_test
        )
        print(f"QSVM (ZZFeatureMap) Accuracy: {q_acc_zz:.4f}")

        # QSVM with PauliFeatureMap
        pauli_map = PauliFeatureMap(feature_dimension=4, reps=2, paulis=["X", "Y", "Z"])
        q_acc_pm, q_cm_pm, K_pm = self.qsvm(
            pauli_map, X_train, X_test, y_train, y_test
        )
        print(f"QSVM (PauliFeatureMap) Accuracy: {q_acc_pm:.4f}")

        # Save results
        os.makedirs("../results", exist_ok=True)

        self.save_heatmap(K_zz, "../results/kernel_matrix_zz.png")
        self.save_heatmap(K_pm, "../results/kernel_matrix_pauli.png")

        print(" Experiment completed.")
        print("Kernel matrices saved in /results/")


if __name__ == "__main__":
    experiment = QSVMExperiment()
    experiment.run()