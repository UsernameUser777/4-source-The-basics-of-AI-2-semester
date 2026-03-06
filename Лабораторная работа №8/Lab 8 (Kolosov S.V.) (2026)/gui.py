# -*- coding: utf-8 -*-
"""Модуль графического интерфейса для приложения анализа иерархий."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, TA_CENTER
from reportlab.lib.units import inch
import json
import csv
import webbrowser

from mcdm_methods import MatrixProcessor  # Используем обновлённый модуль

class MCDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторная работа №8 - Анализ иерархий")
        self.root.geometry("1200x800")
        self.style = ttk.Style()
        self.matrix_processor = MatrixProcessor()
        self.current_matrix = self.matrix_processor.matrix.copy()
        self.matrix_entries = [[None for _ in range(4)] for _ in range(4)]
        self.history = []
        self.history_position = -1
        self.visualizations = {}
        self.reversal_results = {}
        self.sensitivity_results = {}
        self.stability_results = {}

        self.setup_ui()
        self.update_matrix_entries()
        self.update_consistency_info()
        self.add_to_history("Инициализация")

    def setup_ui(self):
        # Меню
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Открыть", command=self.load_matrix)
        file_menu.add_command(label="Сохранить", command=self.save_matrix)
        file_menu.add_separator()
        file_menu.add_command(label="Экспорт в CSV", command=self.export_to_csv)
        file_menu.add_command(label="Экспорт в JSON", command=self.export_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Темная тема", command=lambda: self.toggle_theme(dark=True))
        view_menu.add_command(label="Светлая тема", command=lambda: self.toggle_theme(dark=False))
        view_menu.add_command(label="Режим презентации", command=self.start_presentation_mode)
        menubar.add_cascade(label="Вид", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self.show_about)
        help_menu.add_command(label="Руководство пользователя", command=self.show_help)
        help_menu.add_command(label="Примеры из методички", command=self.show_methodical_examples)
        menubar.add_cascade(label="Справка", menu=help_menu)

        # Notebook (вкладки)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Вкладка "Ввод данных" ---
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Ввод данных")
        self.create_input_tab()

        # --- Вкладка "Результаты" ---
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Результаты")
        self.create_results_tab()

        # --- Вкладка "Реверс рангов" ---
        self.reversal_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reversal_frame, text="Реверс рангов")
        self.create_reversal_tab()

        # --- Вкладка "Чувствительность" ---
        self.sensitivity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sensitivity_frame, text="Чувствительность")
        self.create_sensitivity_tab()

        # --- Вкладка "Стабильность" ---
        self.stability_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stability_frame, text="Стабильность")
        self.create_stability_tab()

        # --- Вкладка "Согласованность" ---
        self.consistency_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.consistency_frame, text="Согласованность")
        self.create_consistency_tab()

        # --- Вкладка "Визуализация" ---
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Визуализация")
        self.create_visualization_tab()

        # --- Вкладка "Отчет" ---
        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="Отчет")
        self.create_report_tab()

        # --- Вкладка "Помощь" ---
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="Помощь")
        self.create_help_tab()

    def create_input_tab(self):
        # История
        history_frame = ttk.LabelFrame(self.input_frame, text="История изменений")
        history_frame.pack(fill=tk.X, padx=10, pady=5)
        self.history_text = tk.Text(history_frame, height=4, state='disabled')
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        hist_btn_frame = ttk.Frame(history_frame)
        hist_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(hist_btn_frame, text="<< Назад", command=self.history_back).pack(side=tk.LEFT)
        ttk.Button(hist_btn_frame, text="Вперед >>", command=self.history_forward).pack(side=tk.LEFT, padx=(5, 0))

        # Матрица
        matrix_frame = ttk.LabelFrame(self.input_frame, text="Матрица парных сравнений (4x4)")
        matrix_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Заголовки столбцов
        for j in range(4):
            ttk.Label(matrix_frame, text=f"A{j+1}").grid(row=0, column=j + 1, padx=5, pady=5)

        # Заголовки строк и ячейки
        for i in range(4):
            ttk.Label(matrix_frame, text=f"A{i+1}").grid(row=i + 1, column=0, padx=5, pady=5)
            for j in range(4):
                var = tk.StringVar(value=str(self.current_matrix[i, j]))
                entry = ttk.Entry(matrix_frame, textvariable=var, width=8)
                entry.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                self.matrix_entries[i][j] = var
                var.trace_add("write", lambda *args, i=i, j=j: self.on_matrix_entry_change(i, j))

        # Кнопки управления
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Обновить", command=self.update_from_entries).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сбросить", command=self.reset_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Проверить симметрию", command=self.check_symmetry).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сравнить с идеальной", command=self.compare_with_ideal).pack(side=tk.LEFT, padx=5)

    def create_results_tab(self):
        # Текстовое поле для результатов
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Кнопки для расчётов
        btn_frame = ttk.Frame(self.results_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Рассчитать веса", command=self.calculate_and_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сравнить методы", command=self.compare_methods).pack(side=tk.LEFT, padx=5)

    def create_reversal_tab(self):
        # Настройка для добавления новой альтернативы
        setup_frame = ttk.LabelFrame(self.reversal_frame, text="Добавление новой альтернативы")
        setup_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(setup_frame, text="Введите веса сравнения новой альтернативы с существующими (A1-A4):").grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.new_alt_entries = []
        for i in range(4):
            ttk.Label(setup_frame, text=f"Сравнение с A{i+1}:").grid(row=i+1, column=0, padx=5, pady=3, sticky=tk.W)
            var = tk.StringVar(value="1.0")
            entry = ttk.Entry(setup_frame, textvariable=var, width=8)
            entry.grid(row=i+1, column=1, padx=5, pady=3)
            self.new_alt_entries.append(var)

        ttk.Button(setup_frame, text="Добавить альтернативу и исследовать реверс",
                   command=self.add_alternative_and_analyze).grid(row=5, column=0, columnspan=2, pady=10)

        # Поле для результатов
        results_frame = ttk.LabelFrame(self.reversal_frame, text="Результаты исследования реверса рангов")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.reversal_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.reversal_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_sensitivity_tab(self):
        # Описание анализа чувствительности
        desc_frame = ttk.LabelFrame(self.sensitivity_frame, text="Описание")
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        desc_text = ("Анализ чувствительности показывает, как изменяются веса альтернатив "
                     "при небольших изменениях в матрице парных сравнений.")
        ttk.Label(desc_frame, text=desc_text, wraplength=1100, justify=tk.LEFT).pack(padx=10, pady=5, anchor=tk.W)

        # Кнопка для запуска анализа
        ttk.Button(self.sensitivity_frame, text="Выполнить анализ чувствительности",
                   command=self.analyze_sensitivity).pack(pady=10)

        # Поле для результатов
        results_frame = ttk.LabelFrame(self.sensitivity_frame, text="Результаты анализа чувствительности")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.sensitivity_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.sensitivity_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_stability_tab(self):
        # Описание анализа стабильности
        desc_frame = ttk.LabelFrame(self.stability_frame, text="Описание")
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        desc_text = ("Анализ стабильности оценивает устойчивость ранжирования альтернатив "
                     "к небольшим изменениям в исходных данных.\n"
                     "Более широкие интервалы указывают на более устойчивое решение.")
        ttk.Label(desc_frame, text=desc_text, wraplength=1100, justify=tk.LEFT).pack(padx=10, pady=5, anchor=tk.W)

        # Кнопка для запуска анализа
        ttk.Button(self.stability_frame, text="Выполнить анализ стабильности",
                   command=self.analyze_stability).pack(pady=10)

        # Поле для результатов
        results_frame = ttk.LabelFrame(self.stability_frame, text="Интервалы устойчивости")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stability_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.stability_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_consistency_tab(self):
        # Метрики согласованности
        info_frame = ttk.LabelFrame(self.consistency_frame, text="Метрики согласованности")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        self.cr_label = ttk.Label(info_frame, text="", font=("Arial", 10, "bold"))
        self.cr_label.pack(pady=5)
        self.ci_label = ttk.Label(info_frame, text="")
        self.ci_label.pack(pady=5)
        self.ri_label = ttk.Label(info_frame, text="")
        self.ri_label.pack(pady=5)
        self.rec_label = ttk.Label(info_frame, text="", wraplength=1100, justify=tk.LEFT)
        self.rec_label.pack(pady=10)

        # Действия по улучшению согласованности
        action_frame = ttk.LabelFrame(self.consistency_frame, text="Действия по улучшению согласованности")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(action_frame, text="Проверить транзитивность", command=self.check_transitivity).pack(pady=5)
        ttk.Button(action_frame, text="Показать несогласованные пары", command=self.show_inconsistent_pairs).pack(pady=5)

    def create_visualization_tab(self):
        # Управление визуализациями
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(control_frame, text="Обновить все графики", command=self.update_all_visualizations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить график как SVG", command=self.save_current_plot_svg).pack(side=tk.LEFT, padx=5)

        # Графики
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_report_tab(self):
        # Управление отчётами
        controls_frame = ttk.Frame(self.report_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(controls_frame, text="Сгенерировать отчет (PDF)", command=self.generate_pdf_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Сгенерировать отчет (LaTeX)", command=self.generate_latex_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Сгенерировать отчет (HTML)", command=self.generate_html_report).pack(side=tk.LEFT, padx=5)

        # Поле для отображения отчёта
        self.report_text = scrolledtext.ScrolledText(self.report_frame, wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_help_tab(self):
        # Руководство пользователя
        text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.show_step_by_step_guide(text_widget=text)

    # --- Методы обработки событий ---
    def on_matrix_entry_change(self, i, j):
        """Обновление матрицы при изменении ячейки."""
        try:
            val = float(self.matrix_entries[i][j].get())
            self.current_matrix[i, j] = val
            if i != j:  # Обновляем симметричный элемент
                self.matrix_entries[j][i].set(str(1.0 / val))
                self.current_matrix[j, i] = 1.0 / val
        except ValueError:
            pass  # Игнорировать некорректный ввод

    def update_from_entries(self):
        """Обновление матрицы из полей ввода."""
        try:
            for i in range(4):
                for j in range(4):
                    self.current_matrix[i, j] = float(self.matrix_entries[i][j].get())
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_consistency_info()
            self.add_to_history("Обновление матрицы")
            messagebox.showinfo("Успех", "Матрица обновлена.")
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные данные в ячейках.")

    def reset_matrix(self):
        """Сброс матрицы к исходному состоянию."""
        self.current_matrix = np.array([
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 5.0, 3.0],
            [1 / 3, 1 / 5, 1.0, 1 / 5],
            [1.0, 1 / 3, 5.0, 1.0]
        ])
        self.matrix_processor = MatrixProcessor(self.current_matrix)
        self.update_matrix_entries()
        self.update_consistency_info()
        self.add_to_history("Сброс матрицы")

    def update_matrix_entries(self):
        """Обновление полей ввода из текущей матрицы."""
        for i in range(4):
            for j in range(4):
                self.matrix_entries[i][j].set(f"{self.current_matrix[i, j]:.4f}")

    def check_symmetry(self):
        """Проверка обратной симметрии матрицы."""
        is_symmetric = True
        violations = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    if abs(self.current_matrix[i, j] * self.current_matrix[j, i] - 1.0) > 1e-9:
                        is_symmetric = False
                        violations.append((i + 1, j + 1))

        if is_symmetric:
            messagebox.showinfo("Симметрия", "Матрица обладает обратной симметрией!")
        else:
            messagebox.showwarning("Предупреждение", f"Обнаружены нарушения обратной симметрии: {violations}")

    def compare_with_ideal(self):
        """Сравнение текущей матрицы с идеальной."""
        ideal_matrix = np.array([
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 5.0, 3.0],
            [1 / 3, 1 / 5, 1.0, 1 / 5],
            [1.0, 1 / 3, 5.0, 1.0]
        ])
        current_matrix = self.current_matrix

        compare_window = tk.Toplevel(self.root)
        compare_window.title("Сравнение с идеальной матрицей")
        compare_window.geometry("800x600")

        frame_current = ttk.LabelFrame(compare_window, text="Текущая матрица")
        frame_current.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_ideal = ttk.LabelFrame(compare_window, text="Идеальная матрица")
        frame_ideal.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        for i in range(4):
            for j in range(4):
                val = current_matrix[i, j]
                bg_color = "white"
                if i != j:
                    ideal_val = ideal_matrix[i, j]
                    if abs(val - ideal_val) > 0.5:
                        bg_color = "yellow"
                label = ttk.Label(frame_current, text=f"{val:.2f}", background=bg_color, width=10, anchor="center")
                label.grid(row=i, column=j, padx=2, pady=2)

        for i in range(4):
            for j in range(4):
                label = ttk.Label(frame_ideal, text=f"{ideal_matrix[i, j]:.2f}", width=10, anchor="center")
                label.grid(row=i, column=j, padx=2, pady=2)

    # --- Методы истории ---
    def add_to_history(self, action):
        """Добавление действия в историю."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history = self.history[:self.history_position + 1]
        self.history.append({
            'matrix': self.current_matrix.copy(),
            'action': action,
            'timestamp': timestamp
        })
        self.history_position = len(self.history) - 1
        self.update_history_display()

    def update_history_display(self):
        """Обновление отображения истории."""
        self.history_text.config(state='normal')
        self.history_text.delete(1.0, tk.END)
        for idx, entry in enumerate(self.history):
            marker = ">>>" if idx == self.history_position else " "
            self.history_text.insert(tk.END, f"{marker}[{entry['timestamp']}] {entry['action']}\n")
        self.history_text.config(state='disabled')

    def history_back(self):
        """Переход назад по истории."""
        if self.history_position > 0:
            self.history_position -= 1
            self.current_matrix = self.history[self.history_position]['matrix'].copy()
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_matrix_entries()
            self.update_consistency_info()
            self.calculate_and_display()
            self.update_all_visualizations()
            self.update_history_display()

    def history_forward(self):
        """Переход вперед по истории."""
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self.current_matrix = self.history[self.history_position]['matrix'].copy()
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_matrix_entries()
            self.update_consistency_info()
            self.calculate_and_display()
            self.update_all_visualizations()
            self.update_history_display()

    # --- Методы обновления информации ---
    def update_consistency_info(self):
        """Обновление информации о согласованности."""
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        self.cr_label.config(text=f"Отношение согласованности (OC): {cr:.4f}")
        self.ci_label.config(text=f"Индекс согласованности (ИС): {ci:.4f}")
        self.ri_label.config(text=f"Случайный индекс (RI): {ri:.4f}")

        if cr < 0.1:
            rec_text = "Матрица согласована. Результаты можно считать надежными."
            self.rec_label.config(text=rec_text, foreground="green")
        else:
            rec_text = ("Матрица несогласована (OC > 0.1). Рекомендуется пересмотреть "
                        "парные сравнения для повышения согласованности.")
            self.rec_label.config(text=rec_text, foreground="red")

    def calculate_and_display(self):
        """Расчет и отображение весов и рангов."""
        methods, weights = self.matrix_processor.get_weights_comparison()
        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Альтернатива {i + 1}" for i in range(4)]
        )
        ranks = np.argsort(-weights, axis=1) + 1
        df_ranks = pd.DataFrame(
            ranks,
            index=methods,
            columns=[f"Альтернатива {i + 1}" for i in range(4)]
        )

        result_str = "=== Результаты расчета весов ===\n\n"
        result_str += str(df_weights.round(4))
        result_str += "\n\n=== Ранги альтернатив ===\n\n"
        result_str += str(df_ranks)

        # Коэффициенты Кендалла
        tau_df, _, _, _, _ = self.matrix_processor.calculate_kendall_tau()
        result_str += "\n\n=== Коэффициенты Кендалла между методами ===\n\n"
        result_str += str(tau_df.round(4))

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state='disabled')

    def compare_methods(self):
        """Сравнение методов расчета весов."""
        methods, weights = self.matrix_processor.get_weights_comparison()
        tau_df, tau_matrix, _, _, _ = self.matrix_processor.calculate_kendall_tau()

        result_str = "=== Сравнение методов ===\n\n"
        for i, method in enumerate(methods):
            result_str += f"{method}: {weights[i].round(4)}\n"
        result_str += "\n=== Коэффициенты Кендалла ===\n\n"
        result_str += str(tau_df.round(4))

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state='disabled')

    # --- Методы анализа ---
    def add_alternative_and_analyze(self):
        """Добавление новой альтернативы и анализ реверса рангов."""
        try:
            new_row = [float(v.get()) for v in self.new_alt_entries]
            if len(new_row) != 4:
                raise ValueError("Необходимо ввести 4 значения.")
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные данные для новой альтернативы: {e}")
            return

        result = self.matrix_processor.add_alternative_and_analyze(new_row)
        self.reversal_results = result

        report_str = "=== Результаты добавления новой альтернативы ===\n\n"
        report_str += "Новая матрица (5x5):\n"
        report_str += str(result["new_matrix"].round(4)) + "\n\n"
        report_str += "Веса по методам (новые):\n"
        for method, w in result["new_weights"].items():
            report_str += f"{method}: {np.array(w).round(4)}\n"
        report_str += "\nРанги по методам (новые):\n"
        for i, method in enumerate(["Дистрибутивный", "Идеальный", "Мультипликативный", "ГУБОПА", "МАИ"]):
            report_str += f"{method}: {result['new_ranks'][i]}\n"

        report_str += "\n=== Анализ реверса рангов ===\n"
        for res in result["reversal_results"]:
            report_str += f"\nМетод: {res['method']}\n"
            report_str += f"Оригинальные ранги (старые): {res['original_ranks']}\n"
            report_str += f"Новые ранги (старых альтернатив): {res['new_ranks']}\n"
            if res["reversal_detected"]:
                report_str += f"Реверс обнаружен! Пары: {res['reversal_pairs']}\n"
            else:
                report_str += "Реверс НЕ обнаружен.\n"

        self.reversal_results_text.config(state='normal')
        self.reversal_results_text.delete(1.0, tk.END)
        self.reversal_results_text.insert(tk.END, report_str)
        self.reversal_results_text.config(state='disabled')

    def analyze_sensitivity(self):
        """Анализ чувствительности весов к изменению элементов матрицы."""
        baseline_weights = self.matrix_processor.get_weights_comparison()[1]
        changes = np.linspace(-0.2, 0.2, 5)
        element_to_vary = (0, 1)  # a12
        results = []
        original_val = self.current_matrix[element_to_vary]

        for delta in changes:
            temp_matrix = self.current_matrix.copy()
            new_val = max(0.1, original_val + delta)  # Ограничиваем диапазон
            temp_matrix[element_to_vary] = new_val
            temp_matrix[element_to_vary[1], element_to_vary[0]] = 1.0 / new_val

            temp_processor = MatrixProcessor(temp_matrix)
            _, new_weights = temp_processor.get_weights_comparison()
            weight_diffs = np.abs(new_weights - baseline_weights)
            avg_change = np.mean(weight_diffs)
            results.append({
                "element_value": new_val,
                "avg_weight_change": avg_change
            })

        report_str = "=== Результаты анализа чувствительности ===\n\n"
        for r in results:
            report_str += (f"Значение a{element_to_vary[0] + 1},{element_to_vary[1] + 1}: {r['element_value']:.2f}, "
                           f"Среднее изменение весов: {r['avg_weight_change']:.4f}\n")

        self.sensitivity_results_text.config(state='normal')
        self.sensitivity_results_text.delete(1.0, tk.END)
        self.sensitivity_results_text.insert(tk.END, report_str)
        self.sensitivity_results_text.config(state='disabled')

    def analyze_stability(self):
        """Анализ стабильности рангов при добавлении шума к матрице."""
        baseline_weights = self.matrix_processor.get_weights_comparison()[1]
        noise_levels = np.linspace(0.0, 0.1, 6)
        results = []

        for level in noise_levels:
            rank_changes = []
            for _ in range(10):  # 10 повторений для каждого уровня шума
                noise = np.random.uniform(-level, level, size=self.current_matrix.shape)
                perturbed_matrix = self.current_matrix + noise
                perturbed_matrix = np.abs(perturbed_matrix)  # Обеспечиваем положительность
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            perturbed_matrix[i, j] = max(0.1, perturbed_matrix[i, j])
                            perturbed_matrix[j, i] = 1.0 / perturbed_matrix[i, j]

                test_proc = MatrixProcessor(perturbed_matrix)
                _, weights = test_proc.get_weights_comparison()
                ranks = np.argsort(-weights, axis=1) + 1
                rank_changes.append(ranks[0])  # Используем первый метод

            results.append({
                "noise_level": level,
                "avg_std": np.mean(np.std(rank_changes, axis=0))
            })

        report_str = "=== Результаты анализа стабильности ===\n\n"
        for r in results:
            report_str += (f"Уровень шума: {r['noise_level']:.2f}, "
                           f"Среднее стандартное отклонение рангов: {r['avg_std']:.4f}\n")

        self.stability_results_text.config(state='normal')
        self.stability_results_text.delete(1.0, tk.END)
        self.stability_results_text.insert(tk.END, report_str)
        self.stability_results_text.config(state='disabled')

    def check_transitivity(self):
        """Проверка транзитивности матрицы."""
        inconsistencies = self.matrix_processor.check_transitivity()
        if inconsistencies:
            msg = f"Найдены нарушения транзитивности: {inconsistencies}"
        else:
            msg = "Нарушения транзитивности не обнаружены."
        messagebox.showinfo("Проверка транзитивности", msg)

    def show_inconsistent_pairs(self):
        """Отображение несогласованных пар."""
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        if cr >= 0.1:
            inconsistent_pairs = self.matrix_processor.find_inconsistent_pairs()
            msg = (f"Матрица несогласована (OC={cr:.4f}).\n"
                   f"Наиболее несогласованные пары:\n{inconsistent_pairs}")
        else:
            msg = "Матрица согласована. Несогласованные пары отсутствуют."
        messagebox.showinfo("Несогласованные пары", msg)

    # --- Методы визуализации ---
    def update_all_visualizations(self):
        """Обновление всех визуализаций."""
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()
        self.axs[1, 0].clear()
        self.axs[1, 1].clear()

        # 1. График весов
        methods, weights = self.matrix_processor.get_weights_comparison()
        x = np.arange(len(methods))
        width = 0.15
        for i in range(4):
            self.axs[0, 0].bar(x + i * width, weights[:, i], width, label=f'A{i + 1}')
        self.axs[0, 0].set_xlabel('Метод')
        self.axs[0, 0].set_ylabel('Вес')
        self.axs[0, 0].set_title('Сравнение весов по методам')
        self.axs[0, 0].set_xticks(x + width * 1.5)
        self.axs[0, 0].set_xticklabels(methods, rotation=45)
        self.axs[0, 0].legend()

        # 2. Тепловая карта матрицы
        im = self.axs[0, 1].imshow(self.current_matrix, cmap='viridis', aspect='auto')
        self.axs[0, 1].set_title('Тепловая карта матрицы')
        self.axs[0, 1].set_xticks(range(4))
        self.axs[0, 1].set_yticks(range(4))
        self.axs[0, 1].set_xticklabels([f'A{i + 1}' for i in range(4)])
        self.axs[0, 1].set_yticklabels([f'A{i + 1}' for i in range(4)])
        self.fig.colorbar(im, ax=self.axs[0, 1])

        # 3. Ранги
        ranks = np.argsort(-weights, axis=1) + 1
        for i in range(4):
            self.axs[1, 0].plot(methods, ranks[:, i], marker='o', label=f'A{i + 1}')
        self.axs[1, 0].set_xlabel('Метод')
        self.axs[1, 0].set_ylabel('Ранг')
        self.axs[1, 0].set_title('Ранги альтернатив по методам')
        self.axs[1, 0].legend()
        self.axs[1, 0].tick_params(axis='x', rotation=45)

        # 4. Гистограмма ОС
        cr, _, _ = self.matrix_processor.calculate_consistency()
        self.axs[1, 1].bar(['OC'], [cr], color=['green' if cr < 0.1 else 'red'])
        self.axs[1, 1].axhline(y=0.1, color='r', linestyle='--', label='Порог (0.1)')
        self.axs[1, 1].set_ylabel('Значение')
        self.axs[1, 1].set_title('Отношение согласованности (ОС)')
        self.axs[1, 1].legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def save_current_plot_svg(self):
        """Сохранение текущего графика в SVG."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            self.fig.savefig(filename, format='svg')

    # --- Методы отчётов ---
    def generate_pdf_report(self):
        """Генерация отчёта в формате PDF."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not filename:
            return

        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = styles['Title']
        title_style.alignment = TA_CENTER
        story.append(Paragraph("Отчет по анализу иерархий", title_style))
        story.append(Spacer(1, 12))

        # Исходная матрица
        story.append(Paragraph("1. Исходная матрица", styles['Heading2']))
        data = [[""] + [f"A{i+1}" for i in range(4)]]
        for i, row in enumerate(self.current_matrix):
            data.append([f"A{i+1}"] + [f"{val:.4f}" for val in row])
        t = Table(data)
        t.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Результаты расчета весов
        story.append(Paragraph("2. Результаты расчета весов", styles['Heading2']))
        methods, weights = self.matrix_processor.get_weights_comparison()
        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Альтернатива {i+1}" for i in range(4)]
        )
        data_w = [df_weights.columns.tolist()]
        data_w += df_weights.round(4).values.tolist()
        data_w[0] = ["Метод"] + data_w[0]
        for i, row in enumerate(data_w[1:], 1):
            data_w[i] = [methods[i - 1]] + row[1:]
        t_w = Table(data_w)
        t_w.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, '#000000')
        ]))
        story.append(t_w)
        story.append(Spacer(1, 12))

        # Согласованность
        story.append(Paragraph("3. Проверка согласованности", styles['Heading2']))
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        cons_text = (f"Отношение согласованности (ОС): {cr:.4f}<br/>"
                     f"Индекс согласованности (ИС): {ci:.4f}<br/>"
                     f"Случайный индекс (RI): {ri:.4f}")
        story.append(Paragraph(cons_text, styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        messagebox.showinfo("Отчет", f"PDF отчет успешно сохранен в {filename}")

    def generate_latex_report(self):
        """Генерация отчёта в формате LaTeX."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")]
        )
        if not filename:
            return

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\\documentclass[12pt]{article}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage[russian]{babel}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\usepackage{longtable}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\geometry{margin=1in}\n")
            f.write("\\title{Отчет по анализу иерархий}\n")
            f.write("\\author{Колосов С.В.}\n")
            f.write("\\date{\\today}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\\n")
            f.write("\\section{Исходная матрица}\n")
            f.write("\\begin{longtable}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("& Альтернатива 1 & Альтернатива 2 & Альтернатива 3 & Альтернатива 4 \\\\\n")
            f.write("\\hline\n")
            for i, row in enumerate(self.current_matrix):
                f.write(f"Альтернатива {i + 1} & ")
                f.write(" & ".join([f"{val:.4f}" for val in row]))
                f.write("\\\\\n")
            f.write("\\hline\n")
            f.write("\\end{longtable}\n")
            f.write("\\n")
            f.write("\\section{Результаты расчета весов}\n")
            methods, weights = self.matrix_processor.get_weights_comparison()
            f.write("\\begin{longtable}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Метод & Альтернатива 1 & Альтернатива 2 & Альтернатива 3 & Альтернатива 4 \\\\\n")
            f.write("\\hline\n")
            for i, method in enumerate(methods):
                f.write(f"{method} & ")
                f.write(" & ".join([f"{w:.4f}" for w in weights[i]]))
                f.write("\\\\\n")
            f.write("\\hline\n")
            f.write("\\end{longtable}\n")
            f.write("\\n")
            f.write("\\section{Проверка согласованности}\n")
            cr, ci, ri = self.matrix_processor.calculate_consistency()
            f.write(f"Отношение согласованности (ОС): {cr:.4f} \\\\\n")
            f.write(f"Индекс согласованности (ИС): {ci:.4f} \\\\\n")
            f.write(f"Случайный индекс (RI): {ri:.4f} \\\\\n")
            f.write("\\end{document}\n")

        messagebox.showinfo("Отчет", f"LaTeX отчет успешно сохранен в {filename}")

    def generate_html_report(self):
        """Генерация отчёта в формате HTML."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        if not filename:
            return

        cr, ci, ri = self.matrix_processor.calculate_consistency()
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8"/>
    <title>Отчет по анализу иерархий</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Отчет по анализу иерархий</h1>
    <h2>Исходная матрица</h2>
    <table>
        <tr><th></th><th>Альтернатива 1</th><th>Альтернатива 2</th><th>Альтернатива 3</th><th>Альтернатива 4</th></tr>
"""
        for i, row in enumerate(self.current_matrix):
            html_content += f"        <tr><td>Альтернатива {i + 1}</td>"
            for val in row:
                html_content += f"<td>{val:.4f}</td>"
            html_content += "</tr>\n"
        html_content += """    </table>
    <h2>Результаты расчета весов</h2>
    <table>
        <tr><th>Метод</th><th>Альтернатива 1</th><th>Альтернатива 2</th><th>Альтернатива 3</th><th>Альтернатива 4</th></tr>
"""
        methods, weights = self.matrix_processor.get_weights_comparison()
        for i, method in enumerate(methods):
            html_content += f"        <tr><td>{method}</td>"
            for w in weights[i]:
                html_content += f"<td>{w:.4f}</td>"
            html_content += "</tr>\n"
        html_content += """    </table>
    <h2>Проверка согласованности</h2>
    <p>Отношение согласованности (ОС): {cr:.4f}</p>
    <p>Индекс согласованности (ИС): {ci:.4f}</p>
    <p>Случайный индекс (RI): {ri:.4f}</p>
</body>
</html>
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        messagebox.showinfo("Отчет", f"HTML отчет успешно сохранен в {filename}")

    # --- Методы справки ---
    def show_step_by_step_guide(self, text_widget=None):
        """Отображение пошагового руководства."""
        guide_content = """ЛАБОРАТОРНАЯ РАБОТА №8. МНОГОКРИТЕРИАЛЬНЫЙ АНАЛИЗ ИЕРАРХИЙ

Шаг 1: Ввод данных
- Перейдите на вкладку "Ввод данных".
- Введите значения в матрицу парных сравнений (4x4).
- Нажмите "Обновить".

Шаг 2: Проверка согласованности
- Перейдите на вкладку "Согласованность".
- Проверьте значение ОС (должно быть < 0.1).
- При необходимости скорректируйте матрицу.

Шаг 3: Расчет весов
- Перейдите на вкладку "Результаты".
- Нажмите "Рассчитать веса".
- Проанализируйте полученные веса и ранги по методам.

Шаг 4: Анализ реверса рангов
- Перейдите на вкладку "Реверс рангов".
- Введите веса для новой альтернативы.
- Нажмите "Добавить альтернативу и исследовать реверс".
- Проанализируйте, изменился ли ранг первой альтернативы.

Шаг 5: Генерация отчета
- Перейдите на вкладку "Отчет".
- Нажмите "Сгенерировать отчет".
- Сохраните результаты.
"""
        if text_widget:
            text_widget.config(state='normal')
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, guide_content)
            text_widget.config(state='disabled')
        else:
            guide_window = tk.Toplevel(self.root)
            guide_window.title("Пошаговое руководство")
            guide_window.geometry("700x500")
            text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, guide_content)
            text.config(state='disabled')

    def show_methodical_examples(self):
        """Отображение примеров из методички."""
        examples_window = tk.Toplevel(self.root)
        examples_window.title("Примеры из методички")
        examples_window.geometry("800x600")
        text = scrolledtext.ScrolledText(examples_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        examples_content = """ПРИМЕРЫ ИЗ МЕТОДИЧЕСКИХ УКАЗАНИЙ
МАИ (Метод анализа иерархий) предполагает построение иерархии целей, критериев и альтернатив.
Для парных сравнений используется шкала Саати (1, 3, 5, 7, 9).
Матрица должна быть согласована (OC < 0.1).
Веса вычисляются, например, как нормализованные суммы строк (дистрибутивный метод) или через собственный вектор (МАИ).

Пример заполнения матрицы:
Если альтернатива A1 "предпочтительнее" A2, то a12 = 3, a21 = 1/3.
"""
        text.insert(tk.END, examples_content)
        text.config(state='disabled')

    def show_help(self):
        """Отображение справки."""
        self.show_step_by_step_guide(text_widget=None)

    def show_about(self):
        """Отображение информации о программе."""
        about_text = """Лабораторная работа №8 по Основам ИИ
Вариант 1

Автор: Колосов С.В.
Группа: IVT-3
Курс: 4

(c) 2026"""
        messagebox.showinfo("О программе", about_text)

    # --- Методы темы ---
    def toggle_theme(self, dark=True):
        """Переключение темной/светлой темы."""
        if dark:
            self.style.theme_use('clam')
            self.root.configure(bg='#2b2b2b')
            self.style.configure('TFrame', background='#3c3f41')
            self.style.configure('TLabel', background='#3c3f41', foreground='white')
            self.style.configure('TButton', background='#4a4d52', foreground='white')
            self.style.map('TButton', background=[('active', '#5a5d62')])
            messagebox.showinfo("Тема", "Активирована тёмная тема")
        else:
            self.style.theme_use('default')
            self.root.configure(bg='SystemButtonFace')
            self.style.configure('TFrame', background='')
            self.style.configure('TLabel', background='', foreground='black')
            self.style.configure('TButton', background='', foreground='black')
            messagebox.showinfo("Тема", "Активирована светлая тема")

    def start_presentation_mode(self):
        """Запуск режима презентации."""
        pres_window = tk.Toplevel(self.root)
        pres_window.attributes('-fullscreen', True)
        pres_window.configure(bg='black')
        label = tk.Label(pres_window, text="РЕЖИМ ПРЕЗЕНТАЦИИ\nНажмите ESC для выхода",
                         fg='white', bg='black', font=("Arial", 24))
        label.pack(expand=True)
        pres_window.bind('<Escape>', lambda e: pres_window.destroy())

    # --- Методы сохранения/загрузки ---
    def load_matrix(self):
        """Загрузка матрицы из файла."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    matrix = np.array(data['matrix'])
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename, header=None)
                matrix = df.values
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат файла.")
                return

            if matrix.shape != (4, 4):
                raise ValueError("Матрица должна быть размером 4x4.")

            self.current_matrix = matrix
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_matrix_entries()
            self.update_consistency_info()
            self.add_to_history("Загрузка матрицы")
            messagebox.showinfo("Успех", f"Матрица загружена из {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def save_matrix(self):
        """Сохранение матрицы в файл."""
        filename = filedialog.asksaveasfilename(
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({"matrix": self.current_matrix.tolist()}, f, indent=4, ensure_ascii=False)
            elif filename.endswith('.csv'):
                df = pd.DataFrame(self.current_matrix)
                df.to_csv(filename, index=False, header=False)
            else:
                messagebox.showerror("Ошибка", "Неподдерживаемый формат файла.")
                return

            messagebox.showinfo("Успех", f"Матрица сохранена в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def export_to_csv(self):
        """Экспорт данных в CSV."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            df = pd.DataFrame(self.current_matrix)
            df.to_csv(filename, index=False)
            messagebox.showinfo("Экспорт", f"Данные экспортированы в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать: {e}")

    def export_to_json(self):
        """Экспорт данных в JSON."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            data = {
                "input_matrix": self.current_matrix.tolist(),
                "calculated_weights": self.matrix_processor.get_weights_comparison()[1].tolist(),
                "consistency": self.matrix_processor.calculate_consistency()
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            messagebox.showinfo("Экспорт", f"Данные экспортированы в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать: {e}")

def main():
    root = tk.Tk()
    app = MCDAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
