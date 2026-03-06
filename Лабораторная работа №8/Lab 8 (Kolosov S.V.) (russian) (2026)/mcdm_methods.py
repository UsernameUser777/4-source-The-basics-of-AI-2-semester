# -*- coding: utf-8 -*-
"""
Модуль с методами многокритериального анализа для лабораторной работы №8
Вариант 1: Выбор метода диагностирования по критерию "степень интегрированности метода"
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
            # Проверка корректности матрицы
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Матрица должна быть квадратной.")
            if np.any(matrix <= 0):
                raise ValueError("Все элементы матрицы должны быть положительными.")
            self.matrix = matrix

        self.n = self.matrix.shape[0]
        self.weights = None
        self.ranks = None
        self.consistency_ratio = None
        self.consistency_index = None
        self.random_index = None
        self.history = []
        self._calculate_initial_metrics()

    def _calculate_initial_metrics(self):
        """Внутренний метод для первоначального расчета метрик при инициализации."""
        self.consistency_ratio, self.consistency_index, self.random_index = self.calculate_consistency()

    def distr_method(self) -> np.ndarray:
        """Дистрибутивный метод: нормализация сумм строк"""
        row_sums = np.sum(self.matrix, axis=1)
        weights = row_sums / np.sum(row_sums)
        return weights

    def ideal_method(self) -> np.ndarray:
        """Метод идеальной точки: расстояние до идеальной точки (минимизация)"""
        ideal = np.max(self.matrix, axis=0)
        normalized = self.matrix / ideal
        row_sums = np.sum(normalized, axis=1)
        weights = row_sums / np.sum(row_sums)
        return weights

    def multiplicative_method(self) -> np.ndarray:
        """Мультипликативный метод: геометрическое среднее строк"""
        geom_means = np.exp(np.mean(np.log(self.matrix), axis=1))
        weights = geom_means / np.sum(geom_means)
        return weights

    def gubopa_method(self) -> np.ndarray:
        """Метод ГУБОПА: произведение элементов строк"""
        products = np.prod(self.matrix, axis=1)
        weights = products / np.sum(products)
        return weights

    def mai_method(self) -> np.ndarray:
        """Метод МАИ: сумма элементов столбцов (нормализованная)"""
        col_sums = np.sum(self.matrix, axis=0)
        weights = col_sums / np.sum(col_sums)
        return weights

    def get_weights_comparison(self) -> Tuple[List[str], np.ndarray]:
        """Возвращает список методов и матрицу весов (5 × n)"""
        methods = ["Дистрибутивный", "Идеальный", "Мультипликативный", "ГУБОПА", "МАИ"]
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
        """Рассчитывает коэффициенты Кендалла между парами методов"""
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
                    "Метод 1": methods[i],
                    "Метод 2": methods[j],
                    "Коэффициент Кендалла": tau,
                    "p-value": p
                })

        tau_df = pd.DataFrame(tau_results)
        ranks = np.argsort(-weights, axis=1) + 1  # rank 1 = best
        return tau_df, tau_matrix, p_matrix, methods, ranks

    def check_transitivity(self) -> List[Tuple[int, int, int]]:
        """Проверка транзитивности: если a > b и b > c, но c > a — нарушение"""
        inconsistencies = []
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if i != j and j != k and i != k:
                        if (self.matrix[i, j] > 1 and self.matrix[j, k] > 1 and self.matrix[k, i] > 1):
                            inconsistencies.append((i + 1, j + 1, k + 1))
        return inconsistencies

    def compare_with_theoretical_expectations(self) -> Dict:
        """Сравнение с ожидаемыми результатами (например, для эталонной матрицы по методичке)"""
        expected_ranks = [1, 2, 4, 3]  # для варианта 1 по методичке
        _, weights = self.get_weights_comparison()
        ranks = np.argsort(-weights, axis=1) + 1
        match_count = sum(1 for r in ranks if np.array_equal(r, expected_ranks))
        return {
            "expected_ranks": expected_ranks,
            "match_count": match_count,
            "total_methods": len(ranks)
        }

    def calculate_consistency(self) -> Tuple[float, float, float]:
        """Расчёт согласованности по Саати: CI, RI, CR"""
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
        Анализ реверса рангов при добавлении новой альтернативы.
        :param new_matrix: новая матрица (n+1)×(n+1)
        :param orig_weights: веса по исходной матрице (5 × n)
        :param orig_ranks: ранги по исходной матрице (5 × n)
        :return: список словарей с результатами для каждого метода
        """
        if orig_weights is None or orig_ranks is None:
            methods, orig_weights = self.get_weights_comparison()
            orig_ranks = np.argsort(-orig_weights, axis=1) + 1

        n_orig = orig_weights.shape[1]  # напр. 4
        n_new = new_matrix.shape[0]  # напр. 5

        new_processor = MatrixProcessor(new_matrix)
        _, new_weights = new_processor.get_weights_comparison()
        new_ranks = np.argsort(-new_weights, axis=1) + 1  # shape: (5, n_new)

        results = []
        for idx, method in enumerate(["Дистрибутивный", "Идеальный", "Мультипликативный", "ГУБОПА", "МАИ"]):
            old_rankings = orig_ranks[idx][:n_orig]  # первые n_orig альтернатив (старые)
            new_rankings = new_ranks[idx][:n_orig]  # ранги старых альтернатив в новой системе

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
                        reversal_pairs.append((i + 1, j + 1, f"Старый: A{i + 1} > A{j + 1}, Новый: A{i + 1} < A{j + 1}"))

                    elif (pos_j_old < pos_i_old) and (pos_j_new > pos_i_new):
                        reversal_detected = True
                        reversal_pairs.append((j + 1, i + 1, f"Старый: A{j + 1} > A{i + 1}, Новый: A{j + 1} < A{i + 1}"))

            results.append({
                "method": method,
                "original_ranks": old_rankings.tolist(),
                "new_ranks": new_rankings.tolist(),
                "reversal_detected": reversal_detected,
                "reversal_pairs": reversal_pairs,
                "critical_value": None  # будет заполнено в find_critical_value
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
        Поиск критического значения параметра, вызывающего реверс ранга целевой альтернативы.
        Улучшенная версия: анализирует все элементы строки target_alternative.
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
            # Изменяем все элементы строки target_alternative (кроме диагонали)
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
                "method": "Дистрибутивный",
                "target_alt": target_alternative
            }
        else:
            return {
                "critical_value": None,
                "reason": "Реверс не найден в заданном диапазоне"
            }

    def add_alternative_and_analyze(self, new_row: List[float]) -> Dict:
        """
        Добавляет новую альтернативу (строку) и анализирует реверс рангов.
        :param new_row: список из n+1 элементов (последний — 1.0)
        :return: dict с результатами
        """
        n = self.matrix.shape[0]

        if len(new_row) != n + 1:
            raise ValueError(f"new_row должен содержать {n + 1} элементов.")

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
        """Находит наиболее несогласованные пары элементов."""
        eigenvals, eigenvecs = np.linalg.eig(self.matrix.T)
        principal_eigenvector = np.real(eigenvecs[:, np.argmax(eigenvals)])
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
        """Генерирует все данные, необходимые для отчета."""
        methods, weights = self.get_weights_comparison()
        ranks = np.argsort(-weights, axis=1) + 1

        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Альтернатива {i + 1}" for i in range(self.n)]
        )

        df_ranks = pd.DataFrame(
            ranks,
            index=methods,
            columns=[f"Альтернатива {i + 1}" for i in range(self.n)]
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
            best_alternatives[method] = f"Альтернатива {best_idx + 1}"

        top_2_counts = {}
        for i, method in enumerate(methods):
            ranks_for_method = ranks[i]
            top_2_alt_indices = np.where(ranks_for_method <= 2)[0]
            count = len(top_2_alt_indices)
            top_2_counts[method] = count

        final_recommendation_method = max(top_2_counts, key=top_2_counts.get)
        final_recommendation_alt = best_alternatives[final_recommendation_method]

        recommendations = [
            f"Матрица {'согласована' if consistency_data['is_consistent'] else 'несогласована'}.",
            f"Наиболее стабильный метод (по кол-ву в топ-2): {final_recommendation_method}.",
            f"Рекомендуемая альтернатива по {final_recommendation_method}: {final_recommendation_alt}."
        ]

        return {
            "input_matrix": pd.DataFrame(
                self.matrix,
                columns=[f"A{i + 1}" for i in range(self.n)],
                index=[f"A{i + 1}" for i in range(self.n)]
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
        """Проверяет корректность работы всех методов."""
        diagnostics = {}

        try:
            self.distr_method()
            diagnostics['distr_method'] = True
        except Exception as e:
            diagnostics['distr_method'] = False
            print(f"Ошибка в дистрибутивном методе: {e}")

        try:
            self.ideal_method()
            diagnostics['ideal_method'] = True
        except Exception as e:
            diagnostics['ideal_method'] = False
            print(f"Ошибка в идеальном методе: {e}")

        try:
            self.multiplicative_method()
            diagnostics['multiplicative_method'] = True
        except Exception as e:
            diagnostics['multiplicative_method'] = False
            print(f"Ошибка в мультипликативном методе: {e}")

        try:
            self.gubopa_method()
            diagnostics['gubopa_method'] = True
        except Exception as e:
            diagnostics['gubopa_method'] = False
            print(f"Ошибка в методе ГУБОПА: {e}")

        try:
            self.mai_method()
            diagnostics['mai_method'] = True
        except Exception as e:
            diagnostics['mai_method'] = False
            print(f"Ошибка в МАИ: {e}")

        try:
            self.calculate_consistency()
            diagnostics['calculate_consistency'] = True
        except Exception as e:
            diagnostics['calculate_consistency'] = False
            print(f"Ошибка в расчете согласованности: {e}")

        try:
            self.calculate_kendall_tau()
            diagnostics['calculate_kendall_tau'] = True
        except Exception as e:
            diagnostics['calculate_kendall_tau'] = False
            print(f"Ошибка в расчете Кендалла: {e}")

        return diagnostics

    def generate_test_data_from_manual(self) -> Dict:
        """Генерирует тестовые данные на основе примеров из методических указаний."""
        return {
            "variant_1": {
                "description": "Выбор метода диагностирования по критерию степени интегрированности метода.",
                "matrix": self.matrix.copy(),
                "expected_results": self.compare_with_theoretical_expectations()
            }
        }

    # Новые функции для улучшения анализа чувствительности и стабильности

    def analyze_sensitivity(self, parameter_range: Tuple[float, float] = (0.5, 2.0), steps: int = 20) -> Dict:
        """
        Анализ чувствительности весов к изменению элементов матрицы.
        :param parameter_range: диапазон изменения параметра.
        :param steps: количество шагов.
        :return: словарь с результатами анализа.
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
                        sensitivity_data.append(weights[0])  # Используем первый метод

                    results[f"A{i + 1}_vs_A{j + 1}"] = {
                        "parameter_values": param_vals.tolist(),
                        "weights": sensitivity_data
                    }

        return results

    def analyze_stability(self, noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """
        Анализ стабильности рангов при добавлении шума к матрице.
        :param noise_levels: уровни шума.
        :return: словарь с результатами анализа.
        """
        results = {}
        base_matrix = self.matrix.copy()

        for noise_level in noise_levels:
            rank_changes = []
            for _ in range(10):  # 10 повторений для каждого уровня шума
                noise = np.random.uniform(-noise_level, noise_level, size=base_matrix.shape)
                perturbed_matrix = base_matrix + noise
                perturbed_matrix = np.abs(perturbed_matrix)  # Обеспечиваем положительность
                perturbed_matrix = (perturbed_matrix + perturbed_matrix.T) / 2  # Симметризация

                test_proc = MatrixProcessor(perturbed_matrix)
                _, weights = test_proc.get_weights_comparison()
                ranks = np.argsort(-weights, axis=1) + 1

                rank_changes.append(ranks[0])  # Используем первый метод

            results[f"noise_{noise_level}"] = {
                "rank_changes": rank_changes,
                "avg_rank_change": np.mean(np.abs(np.diff(rank_changes, axis=0)))
            }

        return results
