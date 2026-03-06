# main.py
# Автор: Колосов С.В., ИВТ-3, 4 курс
# Лабораторная работа №6, вариант №1, 2026 г.
# Главный модуль с графическим интерфейсом

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты из наших модулей
try:
    from risk_analysis import (
        calculate_expected_value, calculate_variance_and_std, calculate_utility,
        value_at_risk, expected_shortfall, monte_carlo_simulation, scenario_analysis,
        ahp_analyze, topsis_analyze, pairwise_comparison_matrix, fuzzy_set_membership,
        generate_recommendations, sensitivity_analysis
    )
    from decision_utils import (
        calculate_risk_matrix, plot_profile_risk_return, plot_venn_diagram,
        export_results_to_excel, export_results_to_pdf, generate_recommendations as du_generate_rec
    )
    from visualization import (
        plot_decision_tree, plot_probability_distribution, plot_weibull_analysis,
        plot_expected_shortfall as viz_plot_es, plot_risk_attitude, safe_save_fig
    )
    from utils.logger import logger
    from utils.file_io import load_project, save_project
except ImportError as e:
    logger = logging.getLogger("MainApp")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.error(f"Ошибка импорта модулей: {e}")
    messagebox.showerror("Ошибка", f"Ошибка импорта модулей: {e}")


class RiskAnalysisApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Лабораторная работа №6 - Анализ рисков (Колосов С.В.)")
        self.root.geometry("1200x800")

        self.options: List[Dict[str, Any]] = []
        self.criteria: List[str] = []
        self.pairwise_comparisons: List[Tuple[int, int, float]] = []
        self.history: List[str] = []

        self.setup_style()
        self.create_widgets()
        self.apply_dark_mode()
        self.load_variant_1_data()

    def setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#2E2E2E", foreground="white", fieldbackground="#2E2E2E")
        self.style.configure("TNotebook.Tab", background="#4A4A4A", foreground="white")
        self.style.map("Treeview", background=[('selected', '#4A4A4A')])
        self.style.configure("TFrame", background="#2B2B2B")
        self.style.configure("TLabel", background="#2B2B2B", foreground="white")
        self.style.configure("TButton", background="#4A4A4A", foreground="white")
        self.style.configure("Header.TLabel", background="#2B2B2B", foreground="#FFD700", font=('Arial', 10, 'bold'))

    def apply_dark_mode(self):
        self.root.configure(bg="#2B2B2B")

    def load_variant_1_data(self):
        """Загружает данные для Варианта №1 по умолчанию"""
        # Вариант 1: Строительство завода
        # n = 1 (номер варианта)
        # P(благоприятный) = 0.5 - 0.01*1 = 0.49
        # P(неблагоприятный) = 0.5 + 0.01*1 = 0.51
        # Крупный завод: 200+8*1 = 208 млн (благоприятный), -180-5*1 = -185 млн (неблагоприятный)
        # Малый завод: 100+5*1 = 105 млн (благоприятный), -20-8*1 = -28 млн (неблагоприятный)

        self.option_tree.delete(*self.option_tree.get_children())
        self.option_tree.insert("", "end", values=("Крупный завод", "208, -185", "0.49, 0.51"))
        self.option_tree.insert("", "end", values=("Малый завод", "105, -28", "0.49, 0.51"))
        self.add_to_history("Загружены данные для Варианта №1")

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка "Ввод данных"
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Ввод данных")

        ttk.Label(self.input_frame, text="Опции и исходы:", style='Header.TLabel').pack(anchor=tk.W, padx=10, pady=5)

        self.option_tree = ttk.Treeview(self.input_frame, columns=("name", "outcomes", "probs"), show="headings",
                                        height=6)
        self.option_tree.heading("name", text="Название")
        self.option_tree.heading("outcomes", text="Исходы (через запятую)")
        self.option_tree.heading("probs", text="Вероятности (через запятую)")
        self.option_tree.column("name", width=150)
        self.option_tree.column("outcomes", width=200)
        self.option_tree.column("probs", width=200)
        self.option_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_frame_input = ttk.Frame(self.input_frame)
        btn_frame_input.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame_input, text="Добавить опцию", command=self.add_option_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_input, text="Удалить опцию", command=self.delete_selected_option).pack(side=tk.LEFT,
                                                                                                    padx=5)
        ttk.Button(btn_frame_input, text="Загрузить Вариант 1", command=self.load_variant_1_data).pack(side=tk.LEFT,
                                                                                                       padx=5)

        # Вкладка "Аналитика"
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text="Аналитика")

        analytics_inner_frame = ttk.Frame(self.analytics_frame)
        analytics_inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame_an = ttk.Frame(analytics_inner_frame)
        btn_frame_an.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame_an, text="Дерево решений", command=self.create_decision_tree).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an, text="Распределение вероятностей", command=self.create_probability_distribution).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an, text="Value at Risk (VaR)", command=self.perform_var_analysis).pack(side=tk.LEFT,
                                                                                                     padx=5)
        ttk.Button(btn_frame_an, text="Expected Shortfall", command=self.perform_expected_shortfall_analysis).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an, text="Монте-Карло", command=self.perform_monte_carlo_analysis).pack(side=tk.LEFT,
                                                                                                     padx=5)
        ttk.Button(btn_frame_an, text="Анализ сценариев", command=self.perform_scenario_analysis).pack(side=tk.LEFT,
                                                                                                       padx=5)
        ttk.Button(btn_frame_an, text="Анализ Вейбулла", command=self.perform_weibull_analysis).pack(side=tk.LEFT,
                                                                                                     padx=5)
        ttk.Button(btn_frame_an, text="Диаграмма Венна", command=self.create_venn_diagram).pack(side=tk.LEFT, padx=5)

        btn_frame_an2 = ttk.Frame(analytics_inner_frame)
        btn_frame_an2.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame_an2, text="Метод анализа иерархий (МАИ)", command=self.perform_ahp_analysis).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an2, text="Метод TOPSIS", command=self.perform_topsis_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an2, text="Профиль риск-доходность", command=self.create_profile_risk_return).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an2, text="Анализ чувствительности", command=self.perform_sensitivity_analysis).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_an2, text="Функция полезности", command=self.create_utility_analysis).pack(side=tk.LEFT,
                                                                                                        padx=5)

        # Вкладка "Отчёты"
        self.reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reports_frame, text="Отчёты")

        reports_inner_frame = ttk.Frame(self.reports_frame)
        reports_inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(reports_inner_frame, text="Результаты анализа:", style='Header.TLabel').pack(anchor=tk.W, padx=5,
                                                                                               pady=5)
        self.report_text = scrolledtext.ScrolledText(reports_inner_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E",
                                                     fg="white")
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame_rep = ttk.Frame(reports_inner_frame)
        btn_frame_rep.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame_rep, text="Экспорт в Excel", command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_rep, text="Экспорт в PDF", command=self.export_to_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame_rep, text="Очистить отчёт", command=self.clear_report).pack(side=tk.LEFT, padx=5)

        # Вкладка "История"
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="История")

        history_inner_frame = ttk.Frame(self.history_frame)
        history_inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(history_inner_frame, text="История действий:", style='Header.TLabel').pack(anchor=tk.W, padx=5,
                                                                                             pady=5)
        self.history_text = scrolledtext.ScrolledText(history_inner_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E",
                                                      fg="white")
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame_hist = ttk.Frame(history_inner_frame)
        btn_frame_hist.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame_hist, text="Очистить историю", command=self.clear_history).pack(side=tk.LEFT, padx=5)

        # Вкладка "Настройки"
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Настройки")

        settings_inner_frame = ttk.Frame(self.settings_frame)
        settings_inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(settings_inner_frame, text="Настройки приложения:", style='Header.TLabel').pack(anchor=tk.W, padx=5,
                                                                                                  pady=5)
        ttk.Label(settings_inner_frame, text="Сохранять историю:", style='Header.TLabel').pack(anchor=tk.W, padx=5,
                                                                                               pady=5)
        self.save_history_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_inner_frame, variable=self.save_history_var).pack(anchor=tk.W, padx=5, pady=5)
        ttk.Button(settings_inner_frame, text="Сбросить настройки", command=self.reset_settings).pack(pady=10)

        # Меню
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить проект", command=self.save_project)
        file_menu.add_command(label="Загрузить проект", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)

        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Вид", menu=view_menu)
        view_menu.add_command(label="Темный режим", command=self.apply_dark_mode)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Методические указания", command=self.show_help)

        # Окна анализа
        self.var_window = None
        self.es_window = None
        self.mc_window = None
        self.scenario_window = None
        self.weibull_window = None
        self.venn_window = None
        self.tree_window = None
        self.prob_dist_window = None
        self.profile_rr_window = None
        self.ahp_window = None
        self.topsis_window = None
        self.sensitivity_window = None
        self.utility_window = None

    def add_option_row(self):
        self.option_tree.insert("", "end", values=("Опция X", "100, 200, -50", "0.3, 0.5, 0.2"))

    def delete_selected_option(self):
        selected = self.option_tree.selection()
        if selected:
            self.option_tree.delete(selected)

    def get_options_from_table(self) -> List[Dict[str, Any]]:
        options = []
        for child in self.option_tree.get_children():
            values = self.option_tree.item(child)["values"]
            if len(values) >= 3:
                name = values[0]
                try:
                    outcomes = [float(x.strip()) for x in values[1].split(",")]
                    probs = [float(x.strip()) for x in values[2].split(",")]

                    if len(outcomes) != len(probs):
                        messagebox.showwarning("Предупреждение",
                                               f"Несоответствие количества исходов и вероятностей для опции: {name}")
                        continue

                    if not np.isclose(sum(probs), 1.0, atol=0.01):
                        messagebox.showwarning("Предупреждение", f"Сумма вероятностей для опции {name} не равна 1.")
                        continue

                except ValueError:
                    logger.warning(f"Неверный формат данных для опции '{name}'. Пропущено.")
                    continue

                options.append({
                    "name": name,
                    "outcomes": outcomes,
                    "probabilities": probs
                })
        return options

    def add_to_history(self, action: str):
        self.history.append(action)
        if self.save_history_var.get():
            self.history_text.config(state='normal')
            self.history_text.insert(tk.END, f"{action}\n")
            self.history_text.config(state='disabled')
            self.history_text.yview(tk.END)

    def clear_history(self):
        self.history = []
        self.history_text.config(state='normal')
        self.history_text.delete(1.0, tk.END)
        self.history_text.config(state='disabled')

    def clear_report(self):
        self.report_text.config(state='normal')
        self.report_text.delete(1.0, tk.END)
        self.report_text.config(state='disabled')

    def reset_settings(self):
        self.save_history_var.set(True)
        messagebox.showinfo("Настройки", "Настройки сброшены.")

    def show_about(self):
        messagebox.showinfo("О программе",
                            "Лабораторная работа №6 по Основам ИИ\n"
                            "Тема: Принятие решений в условиях риска\n"
                            "Автор: Колосов С.В., ИВТ-3, 4 курс\n"
                            "Вариант: №1\n"
                            "2026 г.")

    def show_help(self):
        help_text = """
        Лабораторная работа №6. Вариант №1

        Задача: Выбор между строительством крупного или малого завода.

        Данные:
        - Крупный завод: 208 млн (благоприятный), -185 млн (неблагоприятный)
        - Малый завод: 105 млн (благоприятный), -28 млн (неблагоприятный)
        - Вероятности: 0.49 (благоприятный), 0.51 (неблагоприятный)

        Доступные функции:
        1. Дерево решений - визуализация вариантов
        2. VaR/ES - оценка рисков
        3. Монте-Карло - симуляция сценариев
        4. Анализ чувствительности - устойчивость решения
        5. Функция полезности - отношение к риску
        """
        messagebox.showinfo("Справка", help_text)

    def create_decision_tree(self):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для построения дерева.")
                return

            fig, ax = plot_decision_tree(self.options)
            if fig:
                self.open_figure_window(fig, "Дерево решений")
                self.add_to_history("Создано дерево решений")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить дерево решений.")
        except Exception as e:
            logger.error(f"Ошибка в create_decision_tree: {e}")
            messagebox.showerror("Ошибка", f"Ошибка построения дерева: {e}")

    def create_probability_distribution(self):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для построения распределения.")
                return

            fig, ax = plot_probability_distribution(self.options)
            if fig:
                self.open_figure_window(fig, "Распределение вероятностей")
                self.add_to_history("Создано распределение вероятностей")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить распределение.")
        except Exception as e:
            logger.error(f"Ошибка в create_probability_distribution: {e}")
            messagebox.showerror("Ошибка", f"Ошибка построения распределения: {e}")

    def perform_var_analysis(self):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для VaR.")
                return

            all_returns = []
            for opt in self.options:
                all_returns.extend(opt.get("outcomes", []))

            if not all_returns:
                messagebox.showwarning("Предупреждение", "Нет данных возвратов для VaR.")
                return

            var_95 = value_at_risk(all_returns, 0.95, "historical")
            var_99 = value_at_risk(all_returns, 0.99, "historical")

            result_text = f"Value at Risk (VaR):\n"
            result_text += f"95%: {var_95:.2f}\n"
            result_text += f"99%: {var_99:.2f}\n\n"

            self.report_text.config(state='normal')
            self.report_text.insert(tk.END, result_text)
            self.report_text.config(state='disabled')
            self.report_text.yview(tk.END)

            fig, ax = plt.subplots()
            sns.histplot(all_returns, kde=True, ax=ax)
            ax.axvline(var_95, color='r', linestyle='--', label=f'VaR 95%: {var_95:.2f}')
            ax.axvline(var_99, color='g', linestyle='--', label=f'VaR 99%: {var_99:.2f}')
            ax.set_title("Value at Risk (VaR)")
            ax.legend()
            plt.tight_layout()

            self.open_figure_window(fig, "Value at Risk (VaR)")
            self.add_to_history(f"Выполнен анализ VaR: 95%: {var_95:.2f}, 99%: {var_99:.2f}")
        except Exception as e:
            logger.error(f"Ошибка в perform_var_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа VaR: {e}")

    def perform_expected_shortfall_analysis(self):
        try:
            if self.es_window and self.es_window.winfo_exists():
                self.es_window.lift()
                return

            self.es_window = tk.Toplevel(self.root)
            self.es_window.title("Expected Shortfall")
            self.es_window.geometry("400x200")

            frame = ttk.Frame(self.es_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Уровень доверия (0.95 или 0.99):").pack(anchor=tk.W)
            confidence_var = tk.DoubleVar(value=0.95)
            ttk.Entry(frame, textvariable=confidence_var).pack(fill=tk.X, pady=5)
            ttk.Button(frame, text="Запустить", command=lambda: self.run_es_analysis(confidence_var.get())).pack(
                pady=10)

            results_frame = ttk.Frame(self.es_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.es_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.es_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_expected_shortfall_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска ES: {e}")

    def run_es_analysis(self, confidence_level: float):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для ES.")
                return

            all_returns = []
            for opt in self.options:
                all_returns.extend(opt.get("outcomes", []))

            if not all_returns:
                messagebox.showwarning("Предупреждение", "Нет данных возвратов для ES.")
                return

            es_val = expected_shortfall(all_returns, confidence_level, "historical")

            result_text = f"Expected Shortfall ({confidence_level * 100:.0f}%):\n"
            result_text += f"Значение: {es_val:.2f}\n\n"

            self.es_text.config(state='normal')
            self.es_text.delete(1.0, tk.END)
            self.es_text.insert(tk.END, result_text)
            self.es_text.config(state='disabled')

            fig, ax = viz_plot_es(
                mean=np.mean(all_returns),
                std=np.std(all_returns),
                confidence_level=confidence_level
            )
            if fig:
                self.open_figure_window(fig, f"Expected Shortfall {confidence_level * 100:.0f}%")

            self.add_to_history(f"Выполнен анализ ES {confidence_level * 100:.0f}%: {es_val:.2f}")
        except Exception as e:
            logger.error(f"Ошибка в run_es_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа ES: {e}")

    def perform_monte_carlo_analysis(self):
        try:
            if self.mc_window and self.mc_window.winfo_exists():
                self.mc_window.lift()
                return

            self.mc_window = tk.Toplevel(self.root)
            self.mc_window.title("Монте-Карло Симуляция")
            self.mc_window.geometry("500x300")

            frame = ttk.Frame(self.mc_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Начальное значение:").pack(anchor=tk.W)
            init_val_var = tk.DoubleVar(value=100.0)
            ttk.Entry(frame, textvariable=init_val_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Средняя доходность (mu):").pack(anchor=tk.W)
            mu_var = tk.DoubleVar(value=0.05)
            ttk.Entry(frame, textvariable=mu_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Волатильность (sigma):").pack(anchor=tk.W)
            sigma_var = tk.DoubleVar(value=0.2)
            ttk.Entry(frame, textvariable=sigma_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Горизонт (дней):").pack(anchor=tk.W)
            horizon_var = tk.IntVar(value=252)
            ttk.Entry(frame, textvariable=horizon_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Количество симуляций:").pack(anchor=tk.W)
            n_sim_var = tk.IntVar(value=1000)
            ttk.Entry(frame, textvariable=n_sim_var).pack(fill=tk.X, pady=2)

            ttk.Button(frame, text="Запустить", command=lambda: self.run_mc_analysis(
                init_val_var.get(), mu_var.get(), sigma_var.get(), horizon_var.get(), n_sim_var.get()
            )).pack(pady=10)

            results_frame = ttk.Frame(self.mc_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.mc_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.mc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_monte_carlo_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска МС: {e}")

    def run_mc_analysis(self, init_val: float, mu: float, sigma: float, horizon: int, n_sim: int):
        try:
            results = monte_carlo_simulation(init_val, mu, sigma, horizon, n_sim)

            result_text = f"Монте-Карло Симуляция:\n"
            result_text += f"Среднее: {results['mean']:.2f}\n"
            result_text += f"Стд. откл.: {results['std_dev']:.2f}\n"
            result_text += f"Мин: {results['min']:.2f}\n"
            result_text += f"Макс: {results['max']:.2f}\n"
            result_text += f"95% ДИ: [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]\n\n"

            self.mc_text.config(state='normal')
            self.mc_text.delete(1.0, tk.END)
            self.mc_text.insert(tk.END, result_text)

            recs = generate_recommendations(results)
            self.mc_text.insert(tk.END, recs + "\n\n")
            self.mc_text.config(state='disabled')

            self.add_to_history(f"Выполнен анализ Монте-Карло (n={n_sim})")
        except Exception as e:
            logger.error(f"Ошибка в run_mc_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка симуляции МС: {e}")

    def perform_scenario_analysis(self):
        try:
            if self.scenario_window and self.scenario_window.winfo_exists():
                self.scenario_window.lift()
                return

            self.scenario_window = tk.Toplevel(self.root)
            self.scenario_window.title("Анализ сценариев")
            self.scenario_window.geometry("400x200")

            frame = ttk.Frame(self.scenario_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Базовое значение:").pack(anchor=tk.W)
            base_var = tk.DoubleVar(value=100.0)
            ttk.Entry(frame, textvariable=base_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Фактор оптимистичный (0.1=+10%):").pack(anchor=tk.W)
            opt_var = tk.DoubleVar(value=0.1)
            ttk.Entry(frame, textvariable=opt_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Фактор пессимистичный (0.1=-10%):").pack(anchor=tk.W)
            pess_var = tk.DoubleVar(value=0.1)
            ttk.Entry(frame, textvariable=pess_var).pack(fill=tk.X, pady=2)

            ttk.Button(frame, text="Запустить", command=lambda: self.run_scenario_analysis(
                base_var.get(), opt_var.get(), pess_var.get()
            )).pack(pady=10)

            results_frame = ttk.Frame(self.scenario_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.scenario_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.scenario_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_scenario_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска сценариев: {e}")

    def run_scenario_analysis(self, base: float, opt_f: float, pess_f: float):
        try:
            results = scenario_analysis(base, opt_f, pess_f)

            result_text = f"Анализ сценариев:\n"
            result_text += f"Оптимистичный: {results['optimistic']:.2f}\n"
            result_text += f"Базовый: {results['base']:.2f}\n"
            result_text += f"Пессимистичный: {results['pessimistic']:.2f}\n\n"

            self.scenario_text.config(state='normal')
            self.scenario_text.delete(1.0, tk.END)
            self.scenario_text.insert(tk.END, result_text)
            self.scenario_text.config(state='disabled')

            self.add_to_history(f"Выполнен анализ сценариев (база: {base})")
        except Exception as e:
            logger.error(f"Ошибка в run_scenario_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа сценариев: {e}")

    def perform_weibull_analysis(self):
        try:
            if self.weibull_window and self.weibull_window.winfo_exists():
                self.weibull_window.lift()
                return

            self.weibull_window = tk.Toplevel(self.root)
            self.weibull_window.title("Анализ Вейбулла")
            self.weibull_window.geometry("400x150")

            frame = ttk.Frame(self.weibull_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Параметр формы (shape):").pack(anchor=tk.W)
            shape_var = tk.DoubleVar(value=1.5)
            ttk.Entry(frame, textvariable=shape_var).pack(fill=tk.X, pady=2)

            ttk.Label(frame, text="Параметр масштаба (scale):").pack(anchor=tk.W)
            scale_var = tk.DoubleVar(value=2.0)
            ttk.Entry(frame, textvariable=scale_var).pack(fill=tk.X, pady=2)

            ttk.Button(frame, text="Запустить", command=lambda: self.run_weibull_analysis(
                shape_var.get(), scale_var.get()
            )).pack(pady=10)
        except Exception as e:
            logger.error(f"Ошибка в perform_weibull_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска Вейбулла: {e}")

    def run_weibull_analysis(self, shape: float, scale: float):
        try:
            fig, ax = plot_weibull_analysis(shape=shape, scale=scale)
            if fig:
                self.open_figure_window(fig, f"Анализ Вейбулла (k={shape}, λ={scale})")
                self.add_to_history(f"Выполнен анализ Вейбулла (k={shape}, λ={scale})")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить график Вейбулла.")
        except Exception as e:
            logger.error(f"Ошибка в run_weibull_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа Вейбулла: {e}")

    def create_venn_diagram(self):
        try:
            s1 = {1, 2, 3, 4}
            s2 = {3, 4, 5, 6}
            s3 = {4, 5, 6, 7}

            sets = [s1, s2, s3]
            labels = ["Множество A", "Множество B", "Множество C"]

            fig, ax = plot_venn_diagram(sets, labels)
            if fig:
                self.open_figure_window(fig, "Диаграмма Венна")
                self.add_to_history("Создана диаграмма Венна")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить диаграмму Венна.")
        except Exception as e:
            logger.error(f"Ошибка в create_venn_diagram: {e}")
            messagebox.showerror("Ошибка", f"Ошибка построения диаграммы Венна: {e}")

    def create_profile_risk_return(self):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для профиля.")
                return

            alternatives = []
            for opt in self.options:
                outcomes = opt.get("outcomes", [])
                probs = opt.get("probabilities", [])

                if len(outcomes) != len(probs):
                    continue

                ev = calculate_expected_value(outcomes, probs)
                _, std = calculate_variance_and_std(outcomes, probs, ev)

                alternatives.append({
                    "name": opt["name"],
                    "return": ev,
                    "risk": std
                })

            if not alternatives:
                messagebox.showwarning("Предупреждение", "Нет корректных данных для профиля.")
                return

            fig, ax = plot_profile_risk_return(alternatives)
            if fig:
                self.open_figure_window(fig, "Профиль риск-доходность")
                self.add_to_history("Создан профиль риск-доходность")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить профиль.")
        except Exception as e:
            logger.error(f"Ошибка в create_profile_risk_return: {e}")
            messagebox.showerror("Ошибка", f"Ошибка построения профиля: {e}")

    def perform_sensitivity_analysis(self):
        try:
            if self.sensitivity_window and self.sensitivity_window.winfo_exists():
                self.sensitivity_window.lift()
                return

            self.sensitivity_window = tk.Toplevel(self.root)
            self.sensitivity_window.title("Анализ чувствительности")
            self.sensitivity_window.geometry("500x300")

            frame = ttk.Frame(self.sensitivity_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Изменение вероятности (например, 0.05):").pack(anchor=tk.W)
            prob_change_var = tk.DoubleVar(value=0.05)
            ttk.Entry(frame, textvariable=prob_change_var).pack(fill=tk.X, pady=5)

            ttk.Button(frame, text="Запустить", command=lambda: self.run_sensitivity_analysis(
                prob_change_var.get()
            )).pack(pady=10)

            results_frame = ttk.Frame(self.sensitivity_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.sensitivity_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.sensitivity_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_sensitivity_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска анализа чувствительности: {e}")

    def run_sensitivity_analysis(self, prob_change: float):
        try:
            self.options = self.get_options_from_table()
            if not self.options:
                messagebox.showwarning("Предупреждение", "Нет данных для анализа.")
                return

            results_text = "===== АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ =====\n\n"

            for opt in self.options:
                outcomes = opt.get("outcomes", [])
                probs = opt.get("probabilities", [])

                if len(outcomes) != len(probs):
                    continue

                sens = sensitivity_analysis(outcomes, probs, prob_change)

                results_text += f"Опция: {opt['name']}\n"
                results_text += f"  Базовое EV: {sens['base_ev']:.2f}\n"
                results_text += f"  EV (+{prob_change:.2f}): {sens['ev_up']:.2f}\n"
                results_text += f"  EV (-{prob_change:.2f}): {sens['ev_down']:.2f}\n"
                results_text += f"  Чувствительность: {sens['sensitivity']:.2f}\n\n"

            self.sensitivity_text.config(state='normal')
            self.sensitivity_text.delete(1.0, tk.END)
            self.sensitivity_text.insert(tk.END, results_text)
            self.sensitivity_text.config(state='disabled')

            self.add_to_history(f"Выполнен анализ чувствительности (Δp={prob_change})")
        except Exception as e:
            logger.error(f"Ошибка в run_sensitivity_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа чувствительности: {e}")

    def create_utility_analysis(self):
        try:
            if self.utility_window and self.utility_window.winfo_exists():
                self.utility_window.lift()
                return

            self.utility_window = tk.Toplevel(self.root)
            self.utility_window.title("Функция полезности")
            self.utility_window.geometry("500x300")

            frame = ttk.Frame(self.utility_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Тип функции полезности:").pack(anchor=tk.W)
            utility_type_var = tk.StringVar(value="exponential")
            ttk.Combobox(frame, textvariable=utility_type_var,
                         values=["exponential", "logarithmic", "power", "linear"]).pack(fill=tk.X, pady=5)

            ttk.Label(frame, text="Параметр a:").pack(anchor=tk.W)
            param_a_var = tk.DoubleVar(value=0.05)
            ttk.Entry(frame, textvariable=param_a_var).pack(fill=tk.X, pady=2)

            ttk.Button(frame, text="Построить график", command=lambda: self.run_utility_analysis(
                utility_type_var.get(), param_a_var.get()
            )).pack(pady=10)
        except Exception as e:
            logger.error(f"Ошибка в create_utility_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска анализа полезности: {e}")

    def run_utility_analysis(self, utility_type: str, param_a: float):
        try:
            fig, ax = plot_risk_attitude(utility_type=utility_type, risk_params={"a": param_a})
            if fig:
                self.open_figure_window(fig, f"Функция полезности ({utility_type})")
                self.add_to_history(f"Построена функция полезности ({utility_type})")
            else:
                messagebox.showerror("Ошибка", "Не удалось построить график.")
        except Exception as e:
            logger.error(f"Ошибка в run_utility_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка построения функции полезности: {e}")

    def perform_ahp_analysis(self):
        try:
            if self.ahp_window and self.ahp_window.winfo_exists():
                self.ahp_window.lift()
                return

            self.ahp_window = tk.Toplevel(self.root)
            self.ahp_window.title("Метод анализа иерархий (МАИ)")
            self.ahp_window.geometry("600x400")

            frame = ttk.Frame(self.ahp_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Количество критериев:").pack(anchor=tk.W)
            num_crit_var = tk.IntVar(value=3)
            ttk.Entry(frame, textvariable=num_crit_var, width=5).pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="Запустить", command=lambda: self.run_ahp_setup(num_crit_var.get())).pack(
                side=tk.LEFT, padx=5)

            results_frame = ttk.Frame(self.ahp_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.ahp_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.ahp_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_ahp_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска МАИ: {e}")

    def run_ahp_setup(self, n: int):
        try:
            if hasattr(self, 'ahp_compare_window') and self.ahp_compare_window.winfo_exists():
                self.ahp_compare_window.destroy()

            self.ahp_n = n
            self.ahp_compare_window = tk.Toplevel(self.ahp_window)
            self.ahp_compare_window.title("Парные сравнения МАИ")
            self.ahp_compare_window.geometry("600x500")

            entries = {}
            for i in range(n):
                for j in range(i + 1, n):
                    ttk.Label(self.ahp_compare_window, text=f"C{i + 1} vs C{j + 1}").grid(row=i, column=j, padx=5,
                                                                                          pady=5, sticky=tk.W)
                    entry = tk.Entry(self.ahp_compare_window, width=8)
                    entry.grid(row=i, column=j, padx=5, pady=25)
                    entries[(i, j)] = entry

            def submit_comparisons():
                comparisons = []
                for (i, j), entry in entries.items():
                    try:
                        val = float(entry.get())
                        if val <= 0:
                            messagebox.showwarning("Предупреждение",
                                                   f"Значение для C{i + 1} vs C{j + 1} должно быть > 0.")
                            return
                        comparisons.append((i, j, val))
                    except ValueError:
                        messagebox.showwarning("Предупреждение", f"Неверное значение для C{i + 1} vs C{j + 1}.")
                        return

                self.pairwise_comparisons = comparisons
                self.run_ahp_analysis_final()
                self.ahp_compare_window.destroy()

            ttk.Button(self.ahp_compare_window, text="Завершить", command=submit_comparisons).grid(row=n, column=0,
                                                                                                   columnspan=n,
                                                                                                   pady=20)
        except Exception as e:
            logger.error(f"Ошибка в run_ahp_setup: {e}")
            messagebox.showerror("Ошибка", f"Ошибка настройки МАИ: {e}")

    def run_ahp_analysis_final(self):
        try:
            if not self.pairwise_comparisons:
                messagebox.showwarning("Предупреждение", "Нет данных для МАИ.")
                return

            matrix = pairwise_comparison_matrix([f"C{i + 1}" for i in range(self.ahp_n)], self.pairwise_comparisons)
            weights, ci, cr = ahp_analyze(matrix)

            result_text = f"МАИ Анализ:\n"
            result_text += f"Веса критериев:\n"
            for i, w in enumerate(weights):
                result_text += f"  C{i + 1}: {w:.3f}\n"
            result_text += f"Индекс согласованности (CI): {ci:.3f}\n"
            result_text += f"Отношение согласованности (CR): {cr:.3f}\n"
            result_text += f"{'-' * 20}\n"

            self.ahp_text.config(state='normal')
            self.ahp_text.delete(1.0, tk.END)
            self.ahp_text.insert(tk.END, result_text)
            self.ahp_text.config(state='disabled')

            self.add_to_history(f"Выполнен анализ МАИ (n={self.ahp_n}, CR={cr:.3f})")
        except Exception as e:
            logger.error(f"Ошибка в run_ahp_analysis_final: {e}")
            messagebox.showerror("Ошибка", f"Ошибка анализа МАИ: {e}")

    def perform_topsis_analysis(self):
        try:
            if self.topsis_window and self.topsis_window.winfo_exists():
                self.topsis_window.lift()
                return

            self.topsis_window = tk.Toplevel(self.root)
            self.topsis_window.title("Метод TOPSIS")
            self.topsis_window.geometry("600x400")

            frame = ttk.Frame(self.topsis_window)
            frame.pack(padx=10, pady=10)

            ttk.Label(frame, text="Количество альтернатив:").pack(anchor=tk.W)
            n_alt_var = tk.IntVar(value=3)
            ttk.Entry(frame, textvariable=n_alt_var, width=5).pack(side=tk.LEFT, padx=5)

            ttk.Label(frame, text="Количество критериев:").pack(anchor=tk.W)
            n_crit_var = tk.IntVar(value=3)
            ttk.Entry(frame, textvariable=n_crit_var, width=5).pack(side=tk.LEFT, padx=5)

            ttk.Button(frame, text="Запустить", command=lambda: self.run_topsis_setup(
                n_alt_var.get(), n_crit_var.get()
            )).pack(side=tk.LEFT, padx=5)

            results_frame = ttk.Frame(self.topsis_window)
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.topsis_text = tk.Text(results_frame, wrap=tk.WORD, state='disabled', bg="#1E1E1E", fg="white")
            self.topsis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        except Exception as e:
            logger.error(f"Ошибка в perform_topsis_analysis: {e}")
            messagebox.showerror("Ошибка", f"Ошибка запуска TOPSIS: {e}")

    def run_topsis_setup(self, n_alt: int, n_crit: int):
        try:
            if hasattr(self, 'topsis_input_window') and self.topsis_input_window.winfo_exists():
                self.topsis_input_window.destroy()

            self.topsis_n_alt = n_alt
            self.topsis_n_crit = n_crit
            self.topsis_input_window = tk.Toplevel(self.topsis_window)
            self.topsis_input_window.title("Ввод данных TOPSIS")
            self.topsis_input_window.geometry("800x600")

            ttk.Label(self.topsis_input_window, text="Матрица решений (строки=альтернативы, столбцы=критерии):").pack(
                anchor=tk.W, padx=5, pady=5)

            matrix_frame = ttk.Frame(self.topsis_input_window)
            matrix_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            self.topsis_entries = []
            for i in range(n_alt):
                row = []
                for j in range(n_crit):
                    entry = tk.Entry(matrix_frame, width=8)
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    row.append(entry)
                self.topsis_entries.append(row)

            ttk.Label(self.topsis_input_window, text="Веса критериев (через запятую):").pack(anchor=tk.W, padx=5,
                                                                                             pady=5)
            self.topsis_weights_entry = tk.Entry(self.topsis_input_window)
            self.topsis_weights_entry.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(self.topsis_input_window, text="Типы критериев (max=1, min=0, через запятую):").pack(anchor=tk.W,
                                                                                                           padx=5,
                                                                                                           pady=5)
            self.topsis_types_entry = tk.Entry(self.topsis_input_window)
            self.topsis_types_entry.pack(fill=tk.X, padx=5, pady=5)

            def submit_topsis_data():
                try:
                    decision_matrix = []
                    for i in range(n_alt):
                        row = []
                        for j in range(n_crit):
                            val = float(self.topsis_entries[i][j].get())
                            row.append(val)
                        decision_matrix.append(row)

                    weights_str = self.topsis_weights_entry.get()
                    weights = [float(x.strip()) for x in weights_str.split(",")]

                    types_str = self.topsis_types_entry.get()
                    types = [x.strip() == "1" for x in types_str.split(",")]

                    if len(weights) != n_crit or len(types) != n_crit:
                        messagebox.showwarning("Предупреждение",
                                               "Количество весов/типов не совпадает с количеством критериев.")
                        return

                    scores = topsis_analyze(np.array(decision_matrix), np.array(weights), np.array(types))

                    result_text = f"TOPSIS Анализ:\n"
                    for i, score in enumerate(scores):
                        result_text += f"Альтернатива {i + 1}: {score:.3f}\n"
                    result_text += f"Лучшая альтернатива: #{np.argmax(scores) + 1}\n"
                    result_text += f"{'-' * 20}\n"

                    self.topsis_text.config(state='normal')
                    self.topsis_text.delete(1.0, tk.END)
                    self.topsis_text.insert(tk.END, result_text)
                    self.topsis_text.config(state='disabled')

                    self.add_to_history(f"Выполнен анализ TOPSIS (альт={n_alt}, крит={n_crit})")
                    self.topsis_input_window.destroy()
                except ValueError:
                    messagebox.showwarning("Предупреждение", "Неверный формат чисел.")
                except Exception as ex:
                    logger.error(f"Ошибка в submit_topsis_data: {ex}")
                    messagebox.showerror("Ошибка", f"Ошибка обработки данных TOPSIS: {ex}")

            ttk.Button(self.topsis_input_window, text="Запустить TOPSIS", command=submit_topsis_data).pack(pady=10)
        except Exception as e:
            logger.error(f"Ошибка в run_topsis_setup: {e}")
            messagebox.showerror("Ошибка", f"Ошибка настройки TOPSIS: {e}")

    def open_figure_window(self, fig, title: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("800x600")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Сохранить как PNG", command=lambda: safe_save_fig(fig, "figure.png")).pack(
            side=tk.LEFT, padx=5)

    def export_to_excel(self):
        try:
            data_to_export = {
                "options": self.get_options_from_table(),
                "summary": self.report_text.get(1.0, tk.END).strip()
            }

            filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if filename:
                success = export_results_to_excel(data_to_export, filename)
                if success:
                    messagebox.showinfo("Экспорт", f"Данные экспортированы в {filename}")
                    self.add_to_history(f"Экспортировано в Excel: {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Ошибка", "Не удалось экспортировать в Excel.")
        except Exception as e:
            logger.error(f"Ошибка в export_to_excel: {e}")
            messagebox.showerror("Ошибка", f"Ошибка экспорта в Excel: {e}")

    def export_to_pdf(self):
        try:
            data_to_export = {
                "summary": self.report_text.get(1.0, tk.END).strip()
            }

            filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if filename:
                success = export_results_to_pdf(data_to_export, filename)
                if success:
                    messagebox.showinfo("Экспорт", f"Отчёт экспортирован в {filename}")
                    self.add_to_history(f"Экспортировано в PDF: {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Ошибка", "Не удалось экспортировать в PDF.")
        except Exception as e:
            logger.error(f"Ошибка в export_to_pdf: {e}")
            messagebox.showerror("Ошибка", f"Ошибка экспорта в PDF: {e}")

    def save_project(self):
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if filename:
                data = {
                    "options": self.get_options_from_table(),
                    "history": self.history,
                    "settings": {
                        "save_history": self.save_history_var.get()
                    }
                }
                success = save_project(data, filename)
                if success:
                    messagebox.showinfo("Проект", f"Проект сохранён в {filename}")
                    self.add_to_history(f"Проект сохранён: {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Ошибка", "Не удалось сохранить проект.")
        except Exception as e:
            logger.error(f"Ошибка в save_project: {e}")
            messagebox.showerror("Ошибка", f"Ошибка сохранения проекта: {e}")

    def load_project(self):
        try:
            filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if filename:
                data = load_project(filename)
                if data:
                    self.option_tree.delete(*self.option_tree.get_children())
                    for opt in data.get("options", []):
                        outcomes_str = ",".join(map(str, opt.get("outcomes", [])))
                        probs_str = ",".join(map(str, opt.get("probabilities", [])))
                        self.option_tree.insert("", "end", values=(opt.get("name", ""), outcomes_str, probs_str))

                    self.history = data.get("history", [])
                    self.history_text.config(state='normal')
                    self.history_text.delete(1.0, tk.END)
                    for h in self.history:
                        self.history_text.insert(tk.END, f"{h}\n")
                    self.history_text.config(state='disabled')

                    settings = data.get("settings", {})
                    self.save_history_var.set(settings.get("save_history", True))

                    messagebox.showinfo("Проект", f"Проект загружен из {filename}")
                    self.add_to_history(f"Проект загружен: {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Ошибка", "Не удалось загрузить проект.")
        except Exception as e:
            logger.error(f"Ошибка в load_project: {e}")
            messagebox.showerror("Ошибка", f"Ошибка загрузки проекта: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RiskAnalysisApp(root)
    root.mainloop()
