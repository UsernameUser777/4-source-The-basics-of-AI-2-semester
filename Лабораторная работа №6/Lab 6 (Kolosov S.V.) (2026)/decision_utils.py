# decision_utils.py
# Автор: Колосов С.В., ИВТ-3, 4 курс
# Лабораторная работа №6, вариант №1, 2026 г.
# Утилиты для принятия решений: матрица рисков, профиль, Венна, экспорт, рекомендации

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from fpdf import FPDF
import os

# Импортируем логгер
try:
    from utils.logger import logger
except ImportError:
    logger = logging.getLogger("DecisionUtils")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def calculate_risk_matrix(
        payoffs: List[List[float]],  # [альтернативы][состояния]
        probabilities: List[float]
) -> Tuple[List[List[float]], float]:
    """
    Вычисляет матрицу рисков (сожалений) и ожидаемый риск для каждой альтернативы.
    """
    try:
        payoffs_np = np.array(payoffs, dtype=np.float64)

        if payoffs_np.ndim != 2:
            raise ValueError("payoffs должна быть двумерной матрицей")

        n_alternatives, n_states = payoffs_np.shape

        if len(probabilities) != n_states:
            logger.error(f"Длина вероятностей {len(probabilities)} не соответствует кол-ву состояний {n_states}")
            return [], 0.0

        # Максимальный выигрыш по каждому состоянию
        max_payoffs_per_state = np.max(payoffs_np, axis=0)

        # Матрица рисков (сожалений)
        risk_matrix = max_payoffs_per_state - payoffs_np
        risk_matrix = np.nan_to_num(risk_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        # Ожидаемый риск для каждой альтернативы
        expected_risks = np.sum(risk_matrix * np.array(probabilities), axis=1)

        # Минимальный ожидаемый риск
        min_expected_risk = float(np.min(expected_risks)) if expected_risks.size > 0 else 0.0

        return risk_matrix.tolist(), min_expected_risk
    except Exception as e:
        logger.error(f"Ошибка в calculate_risk_matrix: {e}")
        return [], 0.0


def plot_profile_risk_return(
        alternatives: List[Dict[str, float]]
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Строит профиль риск-доходность для альтернатив.
    """
    try:
        if not alternatives:
            logger.error("plot_profile_risk_return: список альтернатив пуст")
            return None, None

        names = []
        returns = []
        risks = []

        for alt in alternatives:
            names.append(alt.get("name", "N/A"))
            returns.append(alt.get("return", 0.0))
            risks.append(alt.get("risk", 0.0))

        if not returns or not risks:
            logger.error("plot_profile_risk_return: пустые данные по доходности или риску")
            return None, None

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(risks, returns, c=range(len(names)), cmap='viridis', s=100, alpha=0.7)

        ax.set_xlabel("Риск (Std Dev)")
        ax.set_ylabel("Ожидаемая доходность")
        ax.set_title("Профиль риск-доходность")
        ax.grid(True, linestyle='--', alpha=0.6)

        # Подписи точек
        for i, txt in enumerate(names):
            ax.annotate(txt, (risks[i], returns[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.colorbar(scatter, ax=ax)
        plt.tight_layout()

        return fig, ax
    except Exception as e:
        logger.error(f"Ошибка в plot_profile_risk_return: {e}")
        return None, None


def plot_venn_diagram(
        sets: List[Set[Any]],
        labels: Optional[List[str]] = None,
        title: str = "Диаграмма Венна"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Строит диаграмму Венна для 2 или 3 множеств.
    """
    try:
        try:
            from matplotlib_venn import venn2, venn3
            has_venn = True
        except ImportError:
            has_venn = False
            logger.warning("matplotlib_venn не установлен. Используется упрощённая диаграмма.")

        if len(sets) == 2 and has_venn:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                venn2(
                    subsets=(len(sets[0] - sets[1]), len(sets[1] - sets[0]), len(sets[0] & sets[1])),
                    set_labels=labels or ("Множество А", "Множество В"),
                    ax=ax
                )
                ax.set_title(title)
                plt.tight_layout()
                return fig, ax
            except Exception:
                has_venn = False

        elif len(sets) == 3 and has_venn:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                venn3(
                    subsets=(
                        len(sets[0] - sets[1] - sets[2]),
                        len(sets[1] - sets[0] - sets[2]),
                        len(sets[0] & sets[1] - sets[2]),
                        len(sets[2] - sets[0] - sets[1]),
                        len(sets[0] & sets[2] - sets[1]),
                        len(sets[1] & sets[2] - sets[0]),
                        len(sets[0] & sets[1] & sets[2])
                    ),
                    set_labels=labels or ("A", "B", "C"),
                    ax=ax
                )
                ax.set_title(title)
                plt.tight_layout()
                return fig, ax
            except Exception:
                has_venn = False

        # Резервная реализация: 2 круга вручную
        if len(sets) >= 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            circle1 = plt.Circle((0.3, 0.5), 0.3, color='skyblue', alpha=0.5,
                                 label=labels[0] if labels and len(labels) > 0 else 'A')
            circle2 = plt.Circle((0.7, 0.5), 0.3, color='lightcoral', alpha=0.5,
                                 label=labels[1] if labels and len(labels) > 1 else 'B')
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(loc='upper right')
            ax.set_title(title)
            plt.tight_layout()
            return fig, ax
        else:
            logger.error(f"plot_venn_diagram: поддерживает 2 или 3 множества, получено {len(sets)}")
            return None, None
    except Exception as e:
        logger.error(f"Ошибка в plot_venn_diagram: {e}")
        return None, None


def export_results_to_excel(data: Dict[str, Any], filepath: str) -> bool:
    """
    Экспортирует результаты анализа в Excel.
    """
    try:
        # Импортируем функции из risk_analysis для расчётов
        try:
            from risk_analysis import calculate_expected_value, calculate_variance_and_std, calculate_utility
        except ImportError:
            logger.warning("Не удалось импортировать функции из risk_analysis")

            def calculate_expected_value(o, p):
                return 0.0

            def calculate_variance_and_std(o, p, e):
                return (0.0, 0.0)

            def calculate_utility(o, p):
                return 0.0

        df_data = {}

        if 'options' in data:
            rows = []
            for opt in data['options']:
                row = {
                    'Название': opt.get('name', 'N/A'),
                    'Ожидаемое значение': calculate_expected_value(
                        opt.get('outcomes', []),
                        opt.get('probabilities', [])
                    ),
                    'Стандартное отклонение': calculate_variance_and_std(
                        opt.get('outcomes', []),
                        opt.get('probabilities', []),
                        0
                    )[1],
                    'Полезность': calculate_utility(
                        opt.get('outcomes', []),
                        opt.get('probabilities', [])
                    )
                }
                rows.append(row)
            df_data['Анализ опций'] = pd.DataFrame(rows)

        if 'monte_carlo' in data:
            mc_data = data['monte_carlo']
            df_data['Монте-Карло'] = pd.DataFrame({
                'Метрика': ['Среднее', 'Стд. откл.', 'Мин', 'Макс', 'CI_Lower', 'CI_Upper'],
                'Значение': [
                    mc_data.get('mean', 0.0),
                    mc_data.get('std_dev', 0.0),
                    mc_data.get('min', 0.0),
                    mc_data.get('max', 0.0),
                    mc_data.get('ci_lower', 0.0),
                    mc_data.get('ci_upper', 0.0)
                ]
            })

        if not df_data:
            logger.warning("export_results_to_excel: нет данных для экспорта")
            return False

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in df_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Результаты экспортированы в Excel: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта в Excel {filepath}: {e}")
        return False


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Лабораторная работа №6. Анализ рисков', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')


def export_results_to_pdf(data: Dict[str, Any], filepath: str) -> bool:
    """
    Экспортирует текстовые результаты в PDF.
    """
    try:
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        if 'summary' in data:
            summary_str = data['summary']
            lines = summary_str.split('\n')
            for line in lines:
                try:
                    pdf.cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                except:
                    safe_line = line.encode('utf-8', 'ignore').decode('utf-8')
                    pdf.cell(0, 10, safe_line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        else:
            pdf.cell(0, 10, "Нет текстового отчета для экспорта.", ln=True)

        pdf.output(filepath)
        logger.info(f"Результаты экспортированы в PDF: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта в PDF {filepath}: {e}")
        return False


def fuzzy_set_membership(x: float, params: Dict[str, float]) -> float:
    """
    Вычисляет степень принадлежности к нечеткому множеству (треугольное).
    """
    try:
        a = params.get("a", 0.0)
        b = params.get("b", 1.0)
        c = params.get("c", 2.0)

        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Ошибка в fuzzy_set_membership: {e}")
        return 0.0


def pairwise_comparison_matrix(
        criteria_names: List[str],
        comparisons: List[Tuple[int, int, float]]
) -> np.ndarray:
    """
    Создаёт матрицу парных сравнений.
    """
    try:
        n = len(criteria_names)
        matrix = np.ones((n, n))

        for i, j, val in comparisons:
            if 0 <= i < n and 0 <= j < n:
                matrix[i][j] = val
                matrix[j][i] = 1.0 / val if val != 0 else 0.0

        return matrix
    except Exception as e:
        logger.error(f"Ошибка в pairwise_comparison_matrix: {e}")
        return np.ones((len(criteria_names), len(criteria_names)))


def generate_recommendations(data: Dict[str, Any]) -> str:
    """
    Генерирует рекомендации на основе анализа.
    """
    try:
        recommendations = "===== РЕКОМЕНДАЦИИ =====\n\n"

        if 'monte_carlo' in data:
            mc = data['monte_carlo']
            std_dev = mc.get("std_dev", 0.0)
            ci_lower = mc.get("ci_lower", 0.0)
            ci_upper = mc.get("ci_upper", 0.0)

            if std_dev > 20:
                recommendations += "- Рекомендуется диверсификация или хеджирование из-за высокого риска (Std.Dev > 20).\n"
            else:
                recommendations += "- Уровень риска (Std.Dev) находится в приемлемом диапазоне.\n"

            if ci_lower < 0:
                recommendations += "- Возможен убыток в худшем сценарии (нижняя граница доверительного интервала < 0).\n"

            if abs(ci_upper - ci_lower) > 100:
                recommendations += "- Высокая неопределенность прогноза (широкий доверительный интервал).\n"

        if 'var' in data:
            var_val = data['var'].get("value", 0.0)
            if var_val < -50:
                recommendations += "- VaR указывает на потенциальные значительные потери (>50).\n"

        if 'risk_matrix' in data:
            _, min_exp_risk = data['risk_matrix']
            if min_exp_risk > 10:
                recommendations += "- Минимальный ожидаемый риск выше порогового значения.\n"

        recommendations += "\n===== КОНЕЦ РЕКОМЕНДАЦИЙ ====="
        return recommendations
    except Exception as e:
        logger.error(f"Ошибка в generate_recommendations: {e}")
        return "===== РЕКОМЕНДАЦИИ =====\nНе удалось сгенерировать рекомендации.\n===== КОНЕЦ РЕКОМЕНДАЦИЙ ====="


if __name__ == "__main__":
    print("--- Тест decision_utils.py ---")

    payoffs = [[100, 50, -20], [80, 60, 10], [120, 40, -30]]
    probs = [0.3, 0.5, 0.2]

    risk_mat, min_risk = calculate_risk_matrix(payoffs, probs)
    print(f"Матрица рисков:\n{risk_mat}")
    print(f"Мин. ожидаемый риск: {min_risk}")

    alts = [{"name": "A1", "return": 10, "risk": 5}, {"name": "A2", "return": 15, "risk": 8}]
    fig, ax = plot_profile_risk_return(alts)
    if fig:
        fig.savefig("profile_test.png")
        print("Профиль риск-доходность сохранён")

    s1 = {1, 2, 3}
    s2 = {2, 3, 4}
    fig_v, ax_v = plot_venn_diagram([s1, s2], ["S1", "S2"])
    if fig_v:
        fig_v.savefig("venn_test.png")
        print("Диаграмма Венна сохранена в venn_test.png")

    data_for_export = {
        "options": [
            {"name": "Опция 1", "outcomes": [100, -50], "probabilities": [0.6, 0.4]},
            {"name": "Опция 2", "outcomes": [80, 20], "probabilities": [0.5, 0.5]}
        ],
        "monte_carlo": {
            "mean": 100.5, "std_dev": 15.2, "min": 70.0, "max": 130.0,
            "ci_lower": 85.0, "ci_upper": 115.0
        },
        "summary": "Это пример текстового отчета."
    }

    export_results_to_excel(data_for_export, "report.xlsx")
    export_results_to_pdf(data_for_export, "report.pdf")

    recs = generate_recommendations(data_for_export)
    print(recs)

    print("Файл decision_utils.py работает корректно.")
