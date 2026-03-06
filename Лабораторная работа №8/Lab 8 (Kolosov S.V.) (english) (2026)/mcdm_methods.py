# -*- coding: utf-8 -*-
"""
Module with multi-criteria decision-making methods for Laboratory Work #8
Variant 1: Selection of diagnostic method by the criterion of "degree of method integration"
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from typing import List, Dict, Tuple, Optional
import warnings

class MatrixProcessor:
    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = np.array([
                [1.0, 1.0, 3.0, 1.0],
                [1.0, 1.0, 5.0, 3.0],
                [1 / 3, 1 / 5, 1.0, 1 / 5],
                [1.0, 1 / 3, 5.0, 1.0]
            ])
        else:
            # Check matrix validity
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square.")
            if np.any(matrix <= 0):
                raise ValueError("All matrix elements must be positive.")
            self.matrix = matrix
        self.n = self.matrix.shape[0]  # Initialize self.n
        self.weights = None
        self.ranks = None
        self.consistency_ratio = None
        self.consistency_index = None
        self.random_index = None
        self.history = []
        self._calculate_initial_metrics()

    def _calculate_initial_metrics(self):
        """Internal method for initial metrics calculation at initialization."""
        self.consistency_ratio, self.consistency_index, self.random_index = self.calculate_consistency()

    def distr_method(self) -> np.ndarray:
        """Distributive method: row sum normalization"""
        row_sums = np.sum(self.matrix, axis=1)
        weights = row_sums / np.sum(row_sums)
        return weights

    def ideal_method(self) -> np.ndarray:
        """Ideal point method: distance to ideal point (minimization)"""
        ideal = np.max(self.matrix, axis=0)
        normalized = self.matrix / ideal
        row_sums = np.sum(normalized, axis=1)
        weights = row_sums / np.sum(row_sums)
        return weights

    def multiplicative_method(self) -> np.ndarray:
        """Multiplicative method: geometric mean of rows"""
        geom_means = np.exp(np.mean(np.log(self.matrix), axis=1))
        weights = geom_means / np.sum(geom_means)
        return weights

    def gubopa_method(self) -> np.ndarray:
        """GUBOPA method: product of row elements"""
        products = np.prod(self.matrix, axis=1)
        weights = products / np.sum(products)
        return weights

    def mai_method(self) -> np.ndarray:
        """MAI method: normalized column sums"""
        col_sums = np.sum(self.matrix, axis=0)
        weights = col_sums / np.sum(col_sums)
        return weights

    def get_weights_comparison(self) -> Tuple[List[str], np.ndarray]:
        """Return list of methods and weight matrix (5 × n)"""
        methods = ["Distributive", "Ideal", "Multiplicative", "GUBOPA", "MAI"]
        weights_list = [
            self.distr_method(),
            self.ideal_method(),
            self.multiplicative_method(),
            self.gubopa_method(),
            self.mai_method()
        ]
        weights = np.array(weights_list)  # shape: (5, n)
        return methods, weights

    def calculate_kendall_tau(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Calculate Kendall tau coefficients between method pairs"""
        methods, weights = self.get_weights_comparison()
        n_methods = len(methods)
        tau_matrix = np.zeros((n_methods, n_methods))
        p_matrix = np.zeros((n_methods, n_methods))
        tau_results = []

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                tau, p = kendalltau(weights[i], weights[j])
                tau_matrix[i, j] = tau_matrix[j, i] = tau
                p_matrix[i, j] = p_matrix[j, i] = p
                tau_results.append({
                    "Method 1": methods[i],
                    "Method 2": methods[j],
                    "Kendall Tau": tau,
                    "p-value": p
                })

        tau_df = pd.DataFrame(tau_results)
        ranks = np.argsort(-weights, axis=1) + 1  # rank 1 = best
        return tau_df, tau_matrix, p_matrix, methods, ranks

    def check_transitivity(self) -> List[Tuple[int, int, int]]:
        """Check transitivity: if a > b and b > c, but c > a — violation"""
        inconsistencies = []
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if i != j and j != k and i != k:
                        if (self.matrix[i, j] > 1 and self.matrix[j, k] > 1 and self.matrix[k, i] > 1):
                            inconsistencies.append((i + 1, j + 1, k + 1))
        return inconsistencies

    def compare_with_theoretical_expectations(self) -> Dict:
        """Compare with expected results (e.g., for the reference matrix from the manual)"""
        expected_ranks = [1, 2, 4, 3]  # for variant 1 from the manual
        _, weights = self.get_weights_comparison()
        ranks = np.argsort(-weights, axis=1) + 1
        match_count = sum(1 for r in ranks if np.array_equal(r, expected_ranks))
        return {
            "expected_ranks": expected_ranks,
            "match_count": match_count,
            "total_methods": len(ranks)
        }

    def calculate_consistency(self) -> Tuple[float, float, float]:
        """Calculate Saaty consistency: CI, RI, CR"""
        n = self.n
        eigenvalues = np.linalg.eigvals(self.matrix)
        lambda_max = np.real(np.max(eigenvalues))
        CI = (lambda_max - n) / (n - 1)
        RI_values = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        RI = RI_values.get(n, 0.90)
        CR = CI / RI if RI != 0 else 0.0
        return CR, CI, RI

    def analyze_rank_reversal(
        self,
        new_matrix: np.ndarray,
        orig_weights: Optional[np.ndarray] = None,
        orig_ranks: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Analyze rank reversal when adding a new alternative.
        :param new_matrix: new matrix (n+1)×(n+1)
        :param orig_weights: weights from the original matrix (5 × n)
        :param orig_ranks: ranks from the original matrix (5 × n)
        :return: list with results for each method
        """
        if orig_weights is None or orig_ranks is None:
            methods, orig_weights = self.get_weights_comparison()
            orig_ranks = np.argsort(-orig_weights, axis=1) + 1

        n_orig = orig_weights.shape[1]  # e.g., 4
        n_new = new_matrix.shape[0]  # e.g., 5

        new_processor = MatrixProcessor(new_matrix)
        _, new_weights = new_processor.get_weights_comparison()
        new_ranks = np.argsort(-new_weights, axis=1) + 1  # shape: (5, n_new)

        results = []
        for idx, method in enumerate(["Distributive", "Ideal", "Multiplicative", "GUBOPA", "MAI"]):
            old_rankings = orig_ranks[idx][:n_orig]  # first n_orig alternatives (old)
            new_rankings = new_ranks[idx][:n_orig]  # ranks of old alternatives in the new system

            reversal_detected = False
            reversal_pairs = []
            for i in range(n_orig):
                for j in range(i + 1, n_orig):
                    pos_i_old = int(old_rankings[i])
                    pos_j_old = int(old_rankings[j])
                    pos_i_new = int(new_rankings[i])
                    pos_j_new = int(new_rankings[j])

                    if (pos_i_old < pos_j_old) and (pos_i_new > pos_j_new):
                        reversal_detected = True
                        reversal_pairs.append((i + 1, j + 1, f"Old: A{i + 1} > A{j + 1}, New: A{i + 1} < A{j + 1}"))
                    elif (pos_j_old < pos_i_old) and (pos_j_new > pos_i_new):
                        reversal_detected = True
                        reversal_pairs.append((j + 1, i + 1, f"Old: A{j + 1} > A{i + 1}, New: A{j + 1} < A{i + 1}"))

            results.append({
                "method": method,
                "original_ranks": old_rankings.tolist(),
                "new_ranks": new_rankings.tolist(),
                "reversal_detected": reversal_detected,
                "reversal_pairs": reversal_pairs,
                "critical_value": None  # will be filled in find_critical_value
            })

        return results

    def find_critical_value(
        self,
        new_matrix: np.ndarray,
        target_alternative: int = 1,
        parameter_range: Tuple[float, float] = (0.1, 10.0),
        steps: int = 100
    ) -> Dict:
        """
        Find the critical value of the parameter that causes rank reversal of the target alternative.
        Improved version: analyzes all elements of the target_alternative row.
        """
        base_matrix = new_matrix.copy()
        n = base_matrix.shape[0]
        results = []

        param_vals = np.linspace(parameter_range[0], parameter_range[1], steps)
        orig_processor = MatrixProcessor(self.matrix)
        _, orig_weights = orig_processor.get_weights_comparison()
        orig_ranks = np.argsort(-orig_weights, axis=1) + 1

        for val in param_vals:
            test_mat = base_matrix.copy()
            # Change all elements of the target_alternative row (except diagonal)
            for j in range(n):
                if j != target_alternative - 1:
                    test_mat[target_alternative - 1, j] = val
                    test_mat[j, target_alternative - 1] = 1.0 / val
            test_proc = MatrixProcessor(test_mat)

            _, new_weights = test_proc.get_weights_comparison()
            new_ranks = np.argsort(-new_weights, axis=1) + 1

            old_pos = np.where(orig_ranks[0] == target_alternative)[0][0]
            new_pos = np.where(new_ranks[0] == target_alternative)[0][0]

            if old_pos != new_pos:
                results.append({"param": val, "rank_change": old_pos - new_pos})
                break

        if results:
            crit_val = results[0]["param"]
            return {
                "critical_value": crit_val,
                "method": "Distributive",
                "target_alt": target_alternative
            }
        else:
            return {
                "critical_value": None,
                "reason": "No reversal found in the specified range"
            }

    def add_alternative_and_analyze(self, new_row: List[float]) -> Dict:
        """
        Add a new alternative (row) and analyze rank reversal.
        :param new_row: list of n+1 elements (last is 1.0)
        :return: dict with results
        """
        n = self.matrix.shape[0]

        if len(new_row) != n + 1:
            raise ValueError(f"new_row must contain {n + 1} elements.")
        new_matrix = np.eye(n + 1)
        new_matrix[:n, :n] = self.matrix
        new_matrix[n, :n] = np.array(new_row[:n])
        new_matrix[:n, n] = 1.0 / np.array(new_row[:n])

        methods, orig_weights = self.get_weights_comparison()
        orig_ranks = np.argsort(-orig_weights, axis=1) + 1

        reversal_results = self.analyze_rank_reversal(new_matrix, orig_weights, orig_ranks)

        new_processor = MatrixProcessor(new_matrix)
        new_methods, new_weights = new_processor.get_weights_comparison()
        new_ranks = np.argsort(-new_weights, axis=1) + 1

        return {
            "new_matrix": new_matrix,
            "new_weights": dict(zip(new_methods, new_weights)),
            "new_ranks": new_ranks,
            "reversal_results": reversal_results,
            "orig_weights": dict(zip(methods, orig_weights)),
            "orig_ranks": orig_ranks
        }

    def find_inconsistent_pairs(self) -> List[Dict]:
        """Find the most inconsistent pairs of elements."""
        eigenvalues, eigenvecs = np.linalg.eig(self.matrix.T)
        principal_eigenvector = np.real(eigenvecs[:, np.argmax(eigenvalues)])
        principal_eigenvector = np.abs(principal_eigenvector)
        w = principal_eigenvector / np.sum(principal_eigenvector)

        inconsistencies = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    predicted_val = w[i] / w[j]
                    actual_val = self.matrix[i, j]
                    diff = abs(predicted_val - actual_val)
                    inconsistencies.append({
                        "pair": f"A{i + 1} - A{j + 1}",
                        "original_value": actual_val,
                        "predicted_value": predicted_val,
                        "difference": diff
                    })

        inconsistencies.sort(key=lambda x: x["difference"], reverse=True)
        return inconsistencies[:5]

    def generate_report_data(self) -> Dict:
        """Generate all data needed for reporting."""
        methods, weights = self.get_weights_comparison()
        ranks = np.argsort(-weights, axis=1) + 1

        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Alternative {i + 1}" for i in range(self.n)]
        )

        df_ranks = pd.DataFrame(
            ranks,
            index=methods,
            columns=[f"Alternative {i + 1}" for i in range(self.n)]
        )

        tau_df, tau_matrix, p_matrix, _, _ = self.calculate_kendall_tau()

        consistency_data = {
            "ratio": self.consistency_ratio,
            "index": self.consistency_index,
            "random_index": self.random_index,
            "is_consistent": self.consistency_ratio < 0.1
        }

        best_alternatives = {}
        for i, method in enumerate(methods):
            best_idx = np.argmax(weights[i])
            best_alternatives[method] = f"Alternative {best_idx + 1}"

        top_2_counts = {}
        for i, method in enumerate(methods):
            ranks_for_method = ranks[i]
            top_2_alt_indices = np.where(ranks_for_method <= 2)[0]
            count = len(top_2_alt_indices)
            top_2_counts[method] = count

        final_recommendation_method = max(top_2_counts, key=top_2_counts.get)
        final_recommendation_alt = best_alternatives[final_recommendation_method]

        recommendations = [
            f"Matrix is {'consistent' if consistency_data['is_consistent'] else 'inconsistent'}.",
            f"Most stable method (by number in top-2): {final_recommendation_method}.",
            f"Recommended alternative by {final_recommendation_method}: {final_recommendation_alt}."
        ]

        return {
            "input_matrix": pd.DataFrame(
                self.matrix,
                columns=[f"A{i+1}" for i in range(self.n)],
                index=[f"A{i+1}" for i in range(self.n)]
            ),
            "weights_comparison": df_weights,
            "ranks": df_ranks,
            "tau_results": tau_df,
            "consistency": consistency_data,
            "best_alternatives": best_alternatives,
            "final_recommendation": f"{final_recommendation_alt} ({final_recommendation_method})",
            "recommendations": recommendations
        }

    def run_diagnostic(self) -> Dict:
        """Check the correctness of all methods."""
        diagnostics = {}

        try:
            self.distr_method()
            diagnostics['distr_method'] = True
        except Exception as e:
            diagnostics['distr_method'] = False
            print(f"Error in distributive method: {e}")

        try:
            self.ideal_method()
            diagnostics['ideal_method'] = True
        except Exception as e:
            diagnostics['ideal_method'] = False
            print(f"Error in ideal method: {e}")

        try:
            self.multiplicative_method()
            diagnostics['multiplicative_method'] = True
        except Exception as e:
            diagnostics['multiplicative_method'] = False
            print(f"Error in multiplicative method: {e}")

        try:
            self.gubopa_method()
            diagnostics['gubopa_method'] = True
        except Exception as e:
            diagnostics['gubopa_method'] = False
            print(f"Error in GUBOPA method: {e}")

        try:
            self.mai_method()
            diagnostics['mai_method'] = True
        except Exception as e:
            diagnostics['mai_method'] = False
            print(f"Error in MAI: {e}")

        try:
            self.calculate_consistency()
            diagnostics['calculate_consistency'] = True
        except Exception as e:
            diagnostics['calculate_consistency'] = False
            print(f"Error in consistency calculation: {e}")

        try:
            self.calculate_kendall_tau()
            diagnostics['calculate_kendall_tau'] = True
        except Exception as e:
            diagnostics['calculate_kendall_tau'] = False
            print(f"Error in Kendall calculation: {e}")

        return diagnostics

    def generate_test_data_from_manual(self) -> Dict:
        """Generate test data based on examples from methodical guidelines."""
        return {
            "variant_1": {
                "description": "Selection of diagnostic method by the criterion of degree of method integration.",
                "matrix": self.matrix.copy(),
                "expected_results": self.compare_with_theoretical_expectations()
            }
        }

    def analyze_sensitivity(self, parameter_range: Tuple[float, float] = (0.5, 2.0), steps: int = 20) -> Dict:
        """
        Analyze sensitivity of weights to changes in matrix elements.
        :param parameter_range: parameter change range.
        :param steps: number of steps.
        :return: dictionary with analysis results.
        """
        results = {}
        base_matrix = self.matrix.copy()
        param_vals = np.linspace(parameter_range[0], parameter_range[1], steps)

        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    sensitivity_data = []
                    for val in param_vals:
                        test_mat = base_matrix.copy()
                        test_mat[i, j] = val
                        test_mat[j, i] = 1.0 / val
                        test_proc = MatrixProcessor(test_mat)
                        _, weights = test_proc.get_weights_comparison()
                        sensitivity_data.append(weights[0])  # Use the first method

                    results[f"A{i + 1} vs A{j + 1}"] = {
                        "parameter_values": param_vals.tolist(),
                        "weights": sensitivity_data
                    }

        return results

    def analyze_stability(self, noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """
        Analyze stability of ranks when noise is added to the matrix.
        :param noise_levels: noise levels.
        :return: dictionary with analysis results.
        """
        results = {}
        base_matrix = self.matrix.copy()

        for noise_level in noise_levels:
            rank_changes = []
            for _ in range(10):  # 10 repetitions for each noise level
                noise = np.random.uniform(-noise_level, noise_level, size=base_matrix.shape)
                perturbed_matrix = base_matrix + noise
                perturbed_matrix = np.abs(perturbed_matrix)  # Ensure positivity
                perturbed_matrix = (perturbed_matrix + perturbed_matrix.T) / 2  # Symmetrize

                test_proc = MatrixProcessor(perturbed_matrix)
                _, weights = test_proc.get_weights_comparison()
                ranks = np.argsort(-weights, axis=1) + 1

                rank_changes.append(ranks[0])  # Use the first method

            results[f"noise_{noise_level}"] = {
                "rank_changes": rank_changes,
                "avg_rank_change": np.mean(np.abs(np.diff(rank_changes, axis=0)))
            }

        return results
