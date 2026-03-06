# main.py (рабочая версия, проверенная, с улучшениями и подробными комментариями)
"""
Главный модуль лабораторной работы №5 по курсу "Основы искусственного интеллекта".
Решение задачи линейного программирования симплекс-методом с графическим интерфейсом.
Вариант №1: Оптимизация загрузки грузопассажирского судна (отсеки 1,2,3; грузы 1-5).
Режим ручного ввода: пользовательские данные.
Автор: Колосов Станислав
Дата: 2026
"""

import sys
import os
import csv
import json
import logging # Добавляем logging
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Импорты модулей с обработкой ошибок
try:
    from simplex_solver import SimplexSolver
    from sensitivity_analysis import SensitivityAnalyzer
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    messagebox.showerror("Ошибка", f"Не удалось импортировать необходимые модули: {e}\nПроверьте, находятся ли файлы simplex_solver.py и sensitivity_analysis.py в той же папке.")
    sys.exit(1)

# --- УСТАНОВКА КОДИРОВКИ ---
# Python 3.13 по умолчанию использует UTF-8 для строк.
# При работе с файлами всегда указываем encoding='utf-8' для корректной обработки кириллицы.
# Это важно для экспорта, импорта и отображения данных в интерфейсе.

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
# Используем logging для отладки и протоколирования действий пользователя и ошибок
logging.basicConfig(
    level=logging.INFO,  # Уровень логирования (INFO, DEBUG, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.log", encoding='utf-8'), # Лог в файл
        logging.StreamHandler(sys.stdout)  # Лог в консоль
    ]
)
logger = logging.getLogger(__name__) # Создаём логгер для этого модуля


class SimplexApp:
    """
    Главный класс приложения, управляющий GUI и логикой взаимодействия с солвером.
    """
    def __init__(self, root: Tk):
        """
        Инициализирует главное окно и все его компоненты.
        """
        self.root = root
        self.root.title("Лабораторная работа №5: Симплекс-метод (Вариант №1)")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)

        # --- ДАННЫЕ ПО УМОЛЧАНИЮ (Вариант №1) ---
        # Эти данные используются для встроенного расчёта
        self.default_compartments = [
            {'id': 1, 'volume': 500, 'max_weight': 700},
            {'id': 2, 'volume': 1000, 'max_weight': 800},
            {'id': 3, 'volume': 1500, 'max_weight': 1300}
        ]
        # Обновленные данные для грузов (в соответствии с 9 фаза.pdf)
        self.default_cargos = [
            {'index': 1, 'name': 'Мини-тракторы', 'weight': 0.35, 'volume': 3.0, 'price': 8.0, 'availability': 100},
            {'index': 2, 'name': 'Бумага', 'weight': 1.6, 'volume': 1.0, 'price': 21.5, 'availability': 1000},
            {'index': 3, 'name': 'Контейнеры', 'weight': 5.0, 'volume': 6.5, 'price': 51.0, 'availability': 200},
            {'index': 4, 'name': 'Металлопрокат', 'weight': 35.0, 'volume': 6.0, 'price': 275.0, 'availability': 200},
            {'index': 5, 'name': 'Пиломатериалы', 'weight': 4.0, 'volume': 6.0, 'price': 110.0, 'availability': 350}
        ]

        # --- ПЕРЕМЕННЫЕ ДЛЯ ВСТРОЕННОГО ВАРИАНТА ---
        self.compartments = self.default_compartments
        self.cargos = self.default_cargos
        # Переменные для хранения экземпляров солвера и анализатора
        self.solver: SimplexSolver = None
        self.analyzer: SensitivityAnalyzer = None
        # Переменная для хранения исходного вектора b (для сценарного анализа)
        self.original_b = None

        # --- ПЕРЕМЕННЫЕ ДЛЯ РУЧНОГО ВВОДА ---
        # Списки для хранения данных, введённых пользователем
        self.manual_compartments = []
        self.manual_cargos = []
        # Переменные для хранения экземпляров солвера и анализатора для ручного ввода
        self.manual_solver: SimplexSolver = None
        self.manual_analyzer: SensitivityAnalyzer = None
        # Переменная для хранения исходного вектора b для ручного ввода
        self.manual_original_b = None

        # --- ПЕРЕМЕННЫЕ ДЛЯ ИНТЕРФЕЙСА ---
        # Переменные для хранения ссылок на виджеты, таблицы, графики
        self.figures = {}
        self.canvas_widgets = {}
        self.progress_var = DoubleVar() # Переменная для прогресс-бара

        # Переменные для хранения ссылок на StringVar для полей ввода в ручном режиме
        self.manual_comp_vars = [] # [{'id': StringVar, 'volume': StringVar, 'max_weight': StringVar, 'frame': Widget}, ...]
        self.manual_cargo_vars = [] # [{'name': StringVar, 'weight': StringVar, ... , 'frame': Widget}, ...]

        # --- СОЗДАНИЕ ИНТЕРФЕЙСА ---
        # Вызов вспомогательных методов для построения меню, вкладок, виджетов
        self._create_menu()
        self._create_notebook()
        self._create_input_tab()
        self._create_solution_tab()
        self._create_sensitivity_tab()
        self._create_scenarios_tab()
        self._create_results_tab()
        self._create_plots_tab()
        self._create_manual_input_tab()

        # Применение стилей к виджетам
        self._style_widgets()
        # Отображение данных варианта на вкладке "Исходные данные"
        self._display_variant_data()

        # Логируем успешную инициализацию
        logger.info("Приложение SimplexApp инициализировано.")


    def _create_menu(self):
        """
        Создаёт верхнее меню приложения (Файл, Действия, Справка).
        """
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Подменю "Файл"
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Экспортировать результаты в CSV", command=self._export_to_csv)
        file_menu.add_command(label="Экспортировать результаты в JSON", command=self._export_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        # Подменю "Действия"
        action_menu = Menu(menubar, tearoff=0)
        action_menu.add_command(label="Решить задачу (встроенный вариант)", command=self._solve_problem)
        action_menu.add_command(label="Решить задачу (ручной ввод)", command=self._solve_manual_problem)
        action_menu.add_command(label="Очистить результаты", command=self._clear_results)
        menubar.add_cascade(label="Действия", menu=action_menu)

        # Подменю "Справка"
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self._show_about)
        menubar.add_cascade(label="Справка", menu=help_menu)


    def _create_notebook(self):
        """
        Создаёт вкладки (Notebook) для разделения функционала приложения.
        """
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Создаём фреймы для каждой вкладки
        self.tab_input = ttk.Frame(self.notebook)
        self.tab_solution = ttk.Frame(self.notebook)
        self.tab_sensitivity = ttk.Frame(self.notebook)
        self.tab_scenarios = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_plots = ttk.Frame(self.notebook)
        self.tab_manual_input = ttk.Frame(self.notebook)

        # Добавляем вкладки с названиями
        self.notebook.add(self.tab_input, text="Исходные данные")
        self.notebook.add(self.tab_solution, text="Решение (встроенный)")
        self.notebook.add(self.tab_sensitivity, text="Анализ устойчивости (встроенный)")
        self.notebook.add(self.tab_scenarios, text="Сценарии (встроенный)")
        self.notebook.add(self.tab_results, text="Результаты (встроенный)")
        self.notebook.add(self.tab_plots, text="Графики (встроенный)")
        self.notebook.add(self.tab_manual_input, text="Ручной ввод")

        # Создаём прогресс-бар внизу главного окна
        self.progress_frame = Frame(self.root)
        self.progress_frame.pack(fill=X, padx=10, pady=5)
        Label(self.progress_frame, text="Прогресс:").pack(side=LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=LEFT, fill=X, expand=True, padx=5)


    def _create_input_tab(self):
        """
        Создаёт вкладку "Исходные данные" с отображением параметров судна и грузов.
        """
        header = Label(self.tab_input, text="Вариант №1: Оптимизация загрузки судна", font=("Arial", 14, "bold"), fg="#2c3e50")
        header.pack(pady=10)

        # --- Фрейм для отсеков ---
        frame_compartments = LabelFrame(self.tab_input, text="Отсеки судна", font=("Arial", 10, "bold"))
        frame_compartments.pack(fill=X, padx=15, pady=5)
        tree_comp = ttk.Treeview(frame_compartments, columns=("id", "volume", "weight"), show="headings", height=3)
        tree_comp.heading("id", text="№ отсека")
        tree_comp.heading("volume", text="Объём (м³)")
        tree_comp.heading("weight", text="Макс. вес (т)")
        tree_comp.column("id", width=100, anchor=CENTER)
        tree_comp.column("volume", width=150, anchor=CENTER)
        tree_comp.column("weight", width=150, anchor=CENTER)
        for comp in self.compartments:
            tree_comp.insert("", "end", values=(comp['id'], comp['volume'], comp['max_weight']))
        tree_comp.pack(fill=X, padx=5, pady=5)

        # --- Фрейм для грузов ---
        frame_cargos = LabelFrame(self.tab_input, text="Типы грузов", font=("Arial", 10, "bold"))
        frame_cargos.pack(fill=X, padx=15, pady=15)
        tree_cargo = ttk.Treeview(frame_cargos, columns=("idx", "name", "weight", "volume", "price", "avail"),
                                 show="headings", height=5)
        tree_cargo.heading("idx", text="№")
        tree_cargo.heading("name", text="Тип груза")
        tree_cargo.heading("weight", text="Вес (т)")
        tree_cargo.heading("volume", text="Объём (м³)")
        tree_cargo.heading("price", text="Цена (ден.ед.)")
        tree_cargo.heading("avail", text="Наличие (ед.)")
        for col, width, align in [("idx", 120, CENTER), ("name", 120, W), ("weight", 120, CENTER),
                                  ("volume", 120, CENTER), ("price", 120, CENTER), ("avail", 120, CENTER)]:
            tree_cargo.column(col, width=width, anchor=align)
        for cargo in self.cargos:
            tree_cargo.insert("", "end", values=(
                cargo['index'],
                cargo['name'],
                f"{cargo['weight']:.2f}",
                f"{cargo['volume']:.1f}",
                f"{cargo['price']:.1f}",
                cargo['availability']
            ))
        tree_cargo.pack(fill=X, padx=5, pady=5)

        # --- Фрейм для математической модели ---
        frame_model = LabelFrame(self.tab_input, text="Математическая модель", font=("Arial", 10, "bold"))
        frame_model.pack(fill=BOTH, expand=True, padx=15, pady=5)
        model_text = scrolledtext.ScrolledText(frame_model, wrap=WORD, height=12, font=("Courier New", 10))
        model_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        model_content = """Целевая функция: max F = 8·(x₁₁+x₁₂+x₁₃) + 21.5·(x₂₁+x₂₂+x₂₃) + 51·(x₃₁+x₃₂+x₃₃) +
    275·(x₄₁+x₄₂+x₄₃) + 110·(x₅₁+x₅₂+x₅₃) → max

Ограничения:

1. По весу отсеков:
   0.35x₁₁ + 1.6x₂₁ + 5x₃₁ + 35x₄₁ + 4x₅₁ ≤ 700   (отсек 1)
   0.35x₁₂ + 1.6x₂₂ + 5x₃₂ + 35x₄₂ + 4x₅₂ ≤ 800   (отсек 2)
   0.35x₁₃ + 1.6x₂₃ + 5x₃₃ + 35x₄₃ + 4x₅₃ ≤ 1300  (отсек 3)

2. По объёму отсеков:
   3x₁₁ + 1x₂₁ + 6.5x₃₁ + 6x₄₁ + 6x₅₁ ≤ 500       (отсек 1)
   3x₁₂ + 1x₂₂ + 6.5x₃₂ + 6x₄₂ + 6x₅₂ ≤ 1000      (отсек 2)
   3x₁₃ + 1x₂₃ + 6.5x₃₃ + 6x₄₃ + 6x₅₃ ≤ 1500      (отсек 3)

3. По наличию грузов:
   x₁₁ + x₁₂ + x₁₃ ≤ 100    (мини-тракторы)
   x₂₁ + x₂₂ + x₂₃ ≤ 1000   (бумага)
   x₃₁ + x₃₂ + x₃₃ ≤ 200    (контейнеры)
   x₄₁ + x₄₂ + x₄₃ ≤ 200    (металлопрокат)
   x₅₁ + x₅₂ + x₅₃ ≤ 350    (пиломатериалы)

4. Неотрицательность:
   xᵢⱼ ≥ 0 для всех i=1..5, j=1..3

Переменные решения: xᵢⱼ — количество единиц груза типа i в отсеке j (всего 15 переменных)
"""
        model_text.insert(END, model_content)
        model_text.config(state=DISABLED) # Делаем текст только для чтения


    def _create_solution_tab(self):
        """
        Создаёт вкладку "Решение (встроенный)" с кнопками, таблицей и статусом.
        """
        control_frame = Frame(self.tab_solution)
        control_frame.pack(fill=X, padx=10, pady=5)
        Label(control_frame, text="Выберите итерацию (встроенный):", font=("Arial", 10)).pack(side=LEFT, padx=5)
        self.iteration_var = StringVar(value="0")
        self.iteration_combo = ttk.Combobox(control_frame, textvariable=self.iteration_var, width=10, state="readonly")
        self.iteration_combo.pack(side=LEFT, padx=5)
        # Привязываем событие выбора к методу отображения таблицы
        self.iteration_combo.bind("<<ComboboxSelected>>", self._show_selected_iteration)
        Button(control_frame, text="Решить задачу (встроенный)", command=self._solve_problem,
               bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=RIGHT, padx=5)

        table_frame = LabelFrame(self.tab_solution, text="Симплекс-таблица (встроенный)", font=("Arial", 10, "bold"))
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.table_text = scrolledtext.ScrolledText(table_frame, wrap=NONE, font=("Courier New", 9))
        self.table_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.table_text.config(state=DISABLED)

        self.status_label = Label(self.tab_solution, text="Статус (встроенный): Ожидание решения...", font=("Arial", 10, "italic"), fg="#7f8c8d")
        self.status_label.pack(pady=5)


    def _create_sensitivity_tab(self):
        """
        Создаёт вкладку "Анализ устойчивости (встроенный)".
        """
        btn_frame = Frame(self.tab_sensitivity)
        btn_frame.pack(fill=X, padx=10, pady=10)
        Button(btn_frame, text="Сгенерировать отчёт об устойчивости (встроенный)",
               command=self._generate_sensitivity_report,
               bg="#27ae60", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT)
        report_frame = LabelFrame(self.tab_sensitivity, text="Отчёт об устойчивости (встроенный)", font=("Arial", 10, "bold"))
        report_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=WORD, font=("Courier New", 9))
        self.report_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.report_text.config(state=DISABLED)


    def _create_scenarios_tab(self):
        """
        Создаёт вкладку "Сценарии (встроенный)".
        """
        desc_frame = LabelFrame(self.tab_scenarios, text="Описание сценариев (встроенный)", font=("Arial", 10, "bold"))
        desc_frame.pack(fill=X, padx=10, pady=5)
        desc_text = """Сценарии:
1. Пиломатериалы: 400 ед. (вместо 350)
2. Бумага: 900 ед. (вместо 1000)
3. Контейнеры: 100 ед. (вместо 200)"""
        Label(desc_frame, text=desc_text, font=("Arial", 10), justify=LEFT, fg="#2c3e50").pack(padx=10, pady=10)
        btn_frame = Frame(self.tab_scenarios)
        btn_frame.pack(fill=X, padx=10, pady=10)
        Button(btn_frame, text="Рассчитать сценарии (встроенный)", command=self._calculate_scenarios,
               bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=15).pack()
        result_frame = LabelFrame(self.tab_scenarios, text="Результаты (встроенный)", font=("Arial", 10, "bold"))
        result_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.scenario_text = scrolledtext.ScrolledText(result_frame, wrap=WORD, font=("Arial", 10))
        self.scenario_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.scenario_text.config(state=DISABLED)


    def _create_results_tab(self):
        """
        Создаёт вкладку "Результаты (встроенный)" с таблицей оптимального плана.
        """
        header = Label(self.tab_results, text="Оптимальный план (встроенный)", font=("Arial", 14, "bold"), fg="#2c3e50")
        header.pack(pady=10)
        plan_frame = LabelFrame(self.tab_results, text="Распределение грузов (встроенный)", font=("Arial", 10, "bold"))
        plan_frame.pack(fill=BOTH, expand=True, padx=15, pady=5)
        columns = ["Груз", "Отсек 1", "Отсек 2", "Отсек 3", "Итого"]
        self.plan_tree = ttk.Treeview(plan_frame, columns=columns, show="headings", height=6)
        for col, width in zip(columns, [150, 120, 120, 120, 120]):
            self.plan_tree.heading(col, text=col)
            self.plan_tree.column(col, width=width, anchor=CENTER if col != "Груз" else W)
        self.plan_tree.pack(fill=BOTH, expand=True, padx=5, pady=5)
        profit_frame = Frame(self.tab_results)
        profit_frame.pack(fill=X, padx=15, pady=15)
        self.profit_label = Label(profit_frame, text="Максимальная прибыль: —", font=("Arial", 12, "bold"), fg="#e74c3c")
        self.profit_label.pack(side=LEFT)


    def _create_plots_tab(self):
        """
        Создаёт вкладку "Графики (встроенный)" с полем для отрисовки.
        """
        plot_control_frame = Frame(self.tab_plots)
        plot_control_frame.pack(fill=X, padx=10, pady=5)
        Button(plot_control_frame, text="Построить графики (встроенный)", command=self._plot_results,
               bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT, padx=5)
        Button(plot_control_frame, text="Сохранить график (встроенный)", command=self._save_plot,
               bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT, padx=5)
        self.plot_canvas_frame = LabelFrame(self.tab_plots, text="Графики (встроенный)", font=("Arial", 10, "bold"))
        self.plot_canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        # Инициализируем фигуру и оси matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # Создаём Canvas для встраивания matplotlib в tkinter
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas_agg.get_tk_widget().pack(fill=BOTH, expand=True)
        # Рисуем начальный график (пустой)
        self.canvas_agg.draw()


    def _create_manual_input_tab(self):
        """
        Создаёт вкладку "Ручной ввод" с возможностью динамического добавления отсеков и грузов.
        """
        manual_frame = Frame(self.tab_manual_input)
        manual_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # --- Фрейм для отсеков ---
        comp_frame = LabelFrame(manual_frame, text="Отсеки", font=("Arial", 10, "bold"))
        comp_frame.pack(fill=X, padx=5, pady=5)
        self.comp_entries_frame = Frame(comp_frame)
        self.comp_entries_frame.pack(fill=X, padx=5, pady=5)
        self.add_comp_btn = Button(comp_frame, text="Добавить отсек", command=self._add_comp_entry)
        self.add_comp_btn.pack(side=TOP, padx=5, pady=5)

        # --- Фрейм для грузов ---
        cargo_frame = LabelFrame(manual_frame, text="Типы грузов", font=("Arial", 10, "bold"))
        cargo_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.cargo_entries_frame = Frame(cargo_frame)
        self.cargo_entries_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.add_cargo_btn = Button(cargo_frame, text="Добавить груз", command=self._add_cargo_entry)
        self.add_cargo_btn.pack(side=TOP, padx=5, pady=5)

        # --- Кнопка запуска расчёта ---
        solve_manual_btn = Button(manual_frame, text="Рассчитать (ручной ввод)", command=self._solve_manual_problem,
                                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15)
        solve_manual_btn.pack(pady=10)


    def _add_comp_entry(self):
        """
        Добавляет строку полей ввода для нового отсека в фрейм self.comp_entries_frame.
        """
        frame = Frame(self.comp_entries_frame)
        frame.pack(fill=X, padx=2, pady=2)

        Label(frame, text="ID:").grid(row=0, column=0, sticky=W, padx=2)
        # ID автоматически генерируется, поле только для чтения
        id_var = StringVar(value=str(len(self.manual_comp_vars) + 1))
        Entry(frame, textvariable=id_var, width=5, state="disabled").grid(row=0, column=1, padx=2)

        Label(frame, text="Объём:").grid(row=0, column=2, sticky=W, padx=2)
        vol_var = StringVar()
        Entry(frame, textvariable=vol_var, width=10).grid(row=0, column=3, padx=2)

        Label(frame, text="Вес:").grid(row=0, column=4, sticky=W, padx=2)
        weight_var = StringVar()
        Entry(frame, textvariable=weight_var, width=10).grid(row=0, column=5, padx=2)

        remove_btn = Button(frame, text="Удалить", command=lambda f=frame: self._remove_comp_entry(f))
        remove_btn.grid(row=0, column=6, padx=5)

        # Сохраняем ссылки на StringVar и фрейм
        self.manual_comp_vars.append({'id': id_var, 'volume': vol_var, 'max_weight': weight_var, 'frame': frame})


    def _remove_comp_entry(self, frame):
        """
        Удаляет строку ввода отсека и очищает соответствующую запись из списка.
        """
        frame.destroy()
        # Обновляем список, удаляя запись, соответствующую удалённому фрейму
        self.manual_comp_vars = [v for v in self.manual_comp_vars if v['frame'] != frame]


    def _add_cargo_entry(self):
        """
        Добавляет строку полей ввода для нового груза в фрейм self.cargo_entries_frame.
        """
        frame = Frame(self.cargo_entries_frame)
        frame.pack(fill=X, padx=2, pady=2)

        Label(frame, text="Название:").grid(row=0, column=0, sticky=W, padx=2)
        name_var = StringVar()
        Entry(frame, textvariable=name_var, width=15).grid(row=0, column=1, padx=2)

        Label(frame, text="Вес:").grid(row=0, column=2, sticky=W, padx=2)
        weight_var = StringVar()
        Entry(frame, textvariable=weight_var, width=10).grid(row=0, column=3, padx=2)

        Label(frame, text="Объём:").grid(row=0, column=4, sticky=W, padx=2)
        volume_var = StringVar()
        Entry(frame, textvariable=volume_var, width=10).grid(row=0, column=5, padx=2)

        Label(frame, text="Цена:").grid(row=0, column=6, sticky=W, padx=2)
        price_var = StringVar()
        Entry(frame, textvariable=price_var, width=10).grid(row=0, column=7, padx=2)

        Label(frame, text="Наличие:").grid(row=0, column=8, sticky=W, padx=2)
        avail_var = StringVar()
        Entry(frame, textvariable=avail_var, width=10).grid(row=0, column=9, padx=2)

        remove_btn = Button(frame, text="Удалить", command=lambda f=frame: self._remove_cargo_entry(f))
        remove_btn.grid(row=0, column=10, padx=5)

        # Сохраняем ссылки на StringVar и фрейм
        self.manual_cargo_vars.append({
            'name': name_var, 'weight': weight_var, 'volume': volume_var,
            'price': price_var, 'availability': avail_var, 'frame': frame
        })


    def _remove_cargo_entry(self, frame):
        """
        Удаляет строку ввода груза и очищает соответствующую запись из списка.
        """
        frame.destroy()
        # Обновляем список, удаляя запись, соответствующую удалённому фрейму
        self.manual_cargo_vars = [v for v in self.manual_cargo_vars if v['frame'] != frame]


    def _solve_manual_problem(self):
        """
        Собирает данные из полей ввода на вкладке "Ручной ввод",
        строит задачу ЛП, решает её и отображает результаты.
        """
        logger.info("Запуск решения задачи ручного ввода.")
        try:
            # --- Сбор данных об отсеках ---
            compartments_data = []
            for var_dict in self.manual_comp_vars:
                # Валидация ввода: пробуем преобразовать в float
                vol_str = var_dict['volume'].get().strip()
                weight_str = var_dict['max_weight'].get().strip()
                if not vol_str or not weight_str:
                     messagebox.showerror("Ошибка ввода", f"Поле 'Объём' или 'Вес' для отсека {var_dict['id'].get()} пустое.")
                     return
                try:
                    vol = float(vol_str)
                    weight = float(weight_str)
                except ValueError:
                    messagebox.showerror("Ошибка ввода", f"Некорректное число в 'Объём' или 'Вес' для отсека {var_dict['id'].get()}.")
                    return

                if vol <= 0 or weight <= 0:
                    raise ValueError("Объём и вес должны быть положительными.")
                compartments_data.append({'id': int(var_dict['id'].get()), 'volume': vol, 'max_weight': weight})

            # --- Сбор данных о грузах ---
            cargos_data = []
            for i, var_dict in enumerate(self.manual_cargo_vars):
                name = var_dict['name'].get().strip()
                if not name:
                    raise ValueError("Название не может быть пустым.")

                # Валидация ввода: пробуем преобразовать в числа
                weight_str = var_dict['weight'].get().strip()
                volume_str = var_dict['volume'].get().strip()
                price_str = var_dict['price'].get().strip()
                avail_str = var_dict['availability'].get().strip()
                if not all([weight_str, volume_str, price_str, avail_str]):
                     messagebox.showerror("Ошибка ввода", f"Одно из полей для груза '{name}' пустое.")
                     return
                try:
                    weight = float(weight_str)
                    volume = float(volume_str)
                    price = float(price_str)
                    availability = int(avail_str) # Наличие обычно целое
                except ValueError:
                    messagebox.showerror("Ошибка ввода", f"Некорректное число в одном из полей для груза '{name}'.")
                    return

                if weight <= 0 or volume <= 0 or price <= 0 or availability < 0:
                    raise ValueError("Вес, объём, цена должны быть положительными. Наличие >= 0.")
                cargos_data.append({
                    'index': i+1, 'name': name, 'weight': weight, 'volume': volume,
                    'price': price, 'availability': availability
                })

            # Проверка, что введены хотя бы один отсек и один груз
            if not compartments_data or not cargos_data: # ИСПРАВЛЕНО: было cargo_
                messagebox.showwarning("Предупреждение", "Введите хотя бы один отсек и один груз.")
                return

            # --- Построение задачи и решение ---
            c, A, b, var_names, constraint_names = self._build_manual_problem_matrices(compartments_data, cargos_data) # ИСПРАВЛЕНО: было cargo_
            self.manual_original_b = b.copy() # Сохраняем исходный вектор b
            self.manual_solver = SimplexSolver(c, A, b, var_names, constraint_names)
            success, solution, optimal_value = self.manual_solver.solve(max_iterations=100)

            if not success:
                messagebox.showerror("Ошибка", "Не удалось найти оптимальное решение задачи ручного ввода!")
                logger.error("Решение задачи ручного ввода не найдено.")
                return

            # --- Отображение результата ---
            result_text = f"Задача ручного ввода решена!\nМаксимальная прибыль: {optimal_value:.2f} ден.ед.\n\nОптимальный план:\n"
            for i, cargo in enumerate(cargos_data): # ИСПРАВЛЕНО: было cargo_
                total_amount = 0.0
                cargo_result_line = f"{cargo['name']}: "
                for j, comp in enumerate(compartments_data): # ИСПРАВЛЕНО: было compartments_
                    var_idx = i * len(compartments_data) + j
                    amount = solution[var_idx]
                    if amount > 1e-5: # Показываем только ненулевые значения
                        cargo_result_line += f"{comp['id']}({amount:.2f}), "
                        total_amount += amount
                if total_amount > 1e-5:
                    result_text += f"{cargo_result_line[:-2]} (Итого: {total_amount:.2f})\n"
                else:
                    result_text += f"{cargo['name']}: 0 (не используется)\n"
            messagebox.showinfo("Результат (Ручной ввод)", result_text)
            logger.info(f"Задача ручного ввода решена. Прибыль: {optimal_value:.2f}")

            # --- Создание анализатора для ручного решения ---
            self.manual_analyzer = SensitivityAnalyzer(self.manual_solver)

        except ValueError as ve:
            # Ловим ошибки, связанные с валидацией данных (не числа, отрицательные и т.д.)
            messagebox.showerror("Ошибка ввода", f"Ошибка в данных: {str(ve)}")
            logger.error(f"Ошибка ввода в _solve_manual_problem: {ve}")
        except Exception as e:
            # Ловим любые другие ошибки
            messagebox.showerror("Ошибка", f"Ошибка при решении задачи ручного ввода:\n{str(e)}")
            logger.error(f"Ошибка в _solve_manual_problem: {e}")


    def _build_manual_problem_matrices(self, compartments_data: List[Dict], cargos_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Формирует матрицы c, A, b и списки имён для задачи ЛП на основе ручных данных.

        Аргументы:
            compartments_data (List[Dict]): Данные об отсеках.
            cargos_data (List[Dict]): Данные о грузах.

        Возвращает:
            Tuple[c, A, b, var_names, constraint_names]: Матрицы и списки для SimplexSolver.
        """
        num_compartments = len(compartments_data)
        num_cargos = len(cargos_data)
        n = num_cargos * num_compartments # Общее число переменных решения x_ij
        # m = num_compartments (вес) + num_compartments (объём) + num_cargos (наличие)
        m = num_compartments * 2 + num_cargos

        # --- Имена переменных ---
        # x_11, x_12, ..., x_1C, x_21, ..., x_R1, ..., x_RC
        var_names = [f"x{i+1}{j+1}" for i in range(num_cargos) for j in range(num_compartments)]

        # --- Вектор c (коэффициенты целевой функции) ---
        # c = [price_1, price_1, ..., price_1 (C раз), price_2, ..., price_R (C раз)]
        c = np.array([cargo['price'] for cargo in cargos_data for _ in range(num_compartments)])

        # --- Матрица A и вектор b (ограничения) ---
        A = np.zeros((m, n))
        b = np.zeros(m)
        constraint_names = []

        # 1. Ограничения по весу отсеков
        for j, comp in enumerate(compartments_data):
            for i, cargo in enumerate(cargos_data):
                # Индекс переменной x_ij в векторе x: i * num_compartments + j
                var_idx = i * num_compartments + j
                # Коэффициент при x_ij в ограничении по весу отсека j: weight_cargo_i
                A[j, var_idx] = cargo['weight']
            # Правая часть: max_weight отсека j
            b[j] = comp['max_weight']
            constraint_names.append(f"Вес_отсек{comp['id']}")

        # 2. Ограничения по объёму отсеков
        offset_vol = num_compartments # Смещение для строк ограничений по объёму
        for j, comp in enumerate(compartments_data):
            for i, cargo in enumerate(cargos_data):
                var_idx = i * num_compartments + j
                # Коэффициент при x_ij в ограничении по объёму отсека j: volume_cargo_i
                A[offset_vol + j, var_idx] = cargo['volume']
            b[offset_vol + j] = comp['volume']
            constraint_names.append(f"Объем_отсек{comp['id']}")

        # 3. Ограничения по наличию грузов
        offset_avail = 2 * num_compartments # Смещение для строк ограничений по наличию
        for i, cargo in enumerate(cargos_data):
            # Для груза i суммируем x_i1 + x_i2 + ... + x_iC
            for j in range(num_compartments):
                var_idx = i * num_compartments + j
                # Коэффициент при x_ij в ограничении по наличию груза i: 1
                A[offset_avail + i, var_idx] = 1.0
            # Правая часть: availability груза i
            b[offset_avail + i] = cargo['availability']
            constraint_names.append(f"Наличие_груз{cargo['index']}_{cargo['name']}")

        return c, A, b, var_names, constraint_names


    def _style_widgets(self):
        """
        Настраивает внешний вид виджетов с помощью ttk.Style.
        """
        style = ttk.Style()
        style.theme_use('clam') # Выбираем тему
        style.configure('TNotebook.Tab', padding=[12, 8], font=("Arial", 10))
        style.configure('TFrame', background="#f5f5f5")
        style.configure('TLabelframe', background="#f5f5f5", font=("Arial", 10, "bold"))


    def _display_variant_data(self):
        """
        Отображает данные текущего варианта (по умолчанию).
        Может быть расширено для динамического изменения данных.
        """
        # В текущей реализации данные отображаются в _create_input_tab
        pass


    def _solve_problem(self):
        """
        Решает встроенную задачу ЛП для варианта №1.
        """
        logger.info("Запуск решения встроенной задачи.")
        try:
            # --- Построение задачи ---
            c, A, b, var_names, constraint_names = self._build_problem_matrices()
            self.original_b = b.copy() # Сохраняем исходный вектор b
            self.solver = SimplexSolver(c, A, b, var_names, constraint_names)

            # --- Решение ---
            # Добавим отладочный вывод для проверки корректности задачи
            print("\n=== ПРОВЕРКА ПОСТРОЕНИЯ ЗАДАЧИ (Встроенный) ===")
            print(f"Вектор c (целевая функция, первые 10): {c[:10]}...")
            print(f"Вектор b (ограничения): {b}")
            print(f"Матрица A (первые 3 строки, первые 10 столбцов):\n{A[:3, :10]}")
            print(f"Имена переменных (первые 10): {var_names[:10]}")
            print(f"Имена ограничений: {constraint_names}")
            print("==========================================\n")

            success, solution, optimal_value = self.solver.solve(max_iterations=100)

            if not success:
                error_msg = "Не удалось найти оптимальное решение встроенного варианта."
                messagebox.showerror("Ошибка", error_msg)
                self.status_label.config(text="Статус (встроенный): Ошибка решения", fg="#e74c3c")
                logger.error(error_msg)
                return

            print("\n=== РЕЗУЛЬТАТЫ РЕШЕНИЯ (Встроенный) ===")
            print(f"Решение (первые 10): {solution[:10]}...")
            print(f"Прибыль (F): {optimal_value}")
            print("=====================================\n")

            # --- Обновление интерфейса ---
            self._update_solution_display()
            self._update_results_display(solution, optimal_value)
            # Создаём анализатор для найденного решения
            self.analyzer = SensitivityAnalyzer(self.solver)
            status_text = f"Статус (встроенный): Решение найдено! Прибыль = {optimal_value:.2f} ден.ед."
            self.status_label.config(text=status_text, fg="#27ae60")
            messagebox.showinfo("Успех", f"Встроенный вариант решён! Прибыль: {optimal_value:.2f} ден.ед.")
            logger.info(f"Встроенная задача решена. Прибыль: {optimal_value:.2f}")

        except Exception as e:
            error_msg = f"Ошибка (встроенный): {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            self.status_label.config(text="Статус (встроенный): Ошибка", fg="#e74c3c")
            logger.error(error_msg)


    def _build_problem_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Формирует матрицы c, A, b и списки имён для встроенной задачи варианта №1.

        Возвращает:
            Tuple[c, A, b, var_names, constraint_names]: Матрицы и списки для SimplexSolver.
        """
        # --- Имена переменных ---
        # x11, x12, x13, x21, ..., x53 (всего 15)
        var_names = [f"x{i}{j}" for i in range(1, 6) for j in range(1, 4)]

        # --- Вектор c (коэффициенты целевой функции) ---
        # c = [price_1, price_1, price_1, price_2, ..., price_5, price_5, price_5]
        # Для груза i=1..5, цена повторяется 3 раза (для j=1..3)
        c = np.array([cargo['price'] for cargo in self.cargos for _ in range(3)])

        # --- Матрица A и вектор b (ограничения) ---
        # m = 3 (вес) + 3 (объём) + 5 (наличие) = 11
        # n = 15 (x_ij, i=1..5, j=1..3)
        A = np.zeros((11, 15))
        b = np.zeros(11)
        constraint_names = []

        # Вспомогательная функция для вычисления индекса переменной x_ij в векторе x
        # x_11 -> idx 0, x_12 -> idx 1, x_13 -> idx 2, x_21 -> idx 3, ...
        def var_idx(i, j):
            return (i - 1) * 3 + (j - 1)

        # 1. Ограничения по весу отсеков (3 шт.)
        for j_idx, comp in enumerate(self.compartments): # j_idx = 0, 1, 2 -> отсеки 1, 2, 3
            comp_id = comp['id'] # 1, 2, 3
            for i_idx, cargo in enumerate(self.cargos): # i_idx = 0..4 -> грузы 1..5
                cargo_idx = cargo['index'] # 1..5
                A[j_idx, var_idx(cargo_idx, comp_id)] = cargo['weight']
            b[j_idx] = comp['max_weight']
            constraint_names.append(f"Вес_отсек{comp_id}")

        # 2. Ограничения по объёму отсеков (3 шт.)
        for j_idx, comp in enumerate(self.compartments):
            comp_id = comp['id']
            for i_idx, cargo in enumerate(self.cargos):
                cargo_idx = cargo['index']
                A[3 + j_idx, var_idx(cargo_idx, comp_id)] = cargo['volume']
            b[3 + j_idx] = comp['volume']
            constraint_names.append(f"Объем_отсек{comp_id}")

        # 3. Ограничения по наличию грузов (5 шт.)
        for i_idx, cargo in enumerate(self.cargos): # i_idx = 0..4
            cargo_idx = cargo['index'] # 1..5
            for j_idx, comp in enumerate(self.compartments): # j_idx = 0..2
                comp_id = comp['id'] # 1..3
                A[6 + i_idx, var_idx(cargo_idx, comp_id)] = 1.0
            b[6 + i_idx] = cargo['availability']
            constraint_names.append(f"Наличие_груз{cargo_idx}_{cargo['name']}")

        return c, A, b, var_names, constraint_names


    def _update_solution_display(self):
        """
        Обновляет Combobox с выбором итерации и отображает первую (или последнюю) таблицу.
        """
        # Получаем количество итераций из решателя
        iterations_count = len(self.solver.iterations) if self.solver else 0
        # Обновляем список значений Combobox
        self.iteration_combo['values'] = [str(i) for i in range(iterations_count)]
        # Устанавливаем значение на последнюю итерацию (если есть) или 0
        selected_iteration = str(iterations_count - 1) if iterations_count > 0 else "0"
        self.iteration_var.set(selected_iteration)
        # Отображаем выбранную таблицу
        self._show_selected_iteration(None)


    def _show_selected_iteration(self, event):
        """
        Отображает симплекс-таблицу для выбранной итерации в текстовом поле.
        """
        try:
            # Проверяем, существует ли решатель
            if not self.solver:
                return
            # Получаем номер итерации из Combobox
            iteration_num = int(self.iteration_var.get())
            # Получаем строковое представление таблицы
            table_str = self.solver.get_iteration_table(iteration_num)
            # Обновляем текстовое поле
            self.table_text.config(state=NORMAL)
            self.table_text.delete(1.0, END)
            self.table_text.insert(END, table_str)
            self.table_text.config(state=DISABLED)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при отображении таблицы: {str(e)}")
            logger.error(f"Ошибка в _show_selected_iteration: {e}")


    def _update_results_display(self, solution: np.ndarray, optimal_value: float):
        """
        Обновляет таблицу с оптимальным планом на вкладке "Результаты".
        """
        # Очищаем текущие строки таблицы
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)

        # Заполняем таблицу данными из решения
        for cargo in self.cargos:
            i = cargo['index'] # Индекс груза (1..5)
            row = [cargo['name']] # Первая колонка - название груза
            total = 0.0
            for j in range(1, 4): # Перебираем отсеки (1..3)
                var_name = f"x{i}{j}" # Имя переменной x_ij
                idx = (i-1)*3 + (j-1) # Индекс переменной в векторе решения
                amount = solution[idx] # Значение переменной из решения
                row.append(f"{amount:.2f}") # Добавляем значение для отсека j
                total += amount # Суммируем для итога
            row.append(f"{total:.2f}") # Добавляем итоговое количество
            self.plan_tree.insert("", "end", values=row) # Вставляем строку в таблицу

        # Обновляем метку с максимальной прибылью
        self.profit_label.config(text=f"Максимальная прибыль: {optimal_value:.2f} ден.ед.")


    def _generate_sensitivity_report(self):
        """
        Генерирует и отображает отчёт об устойчивости решения.
        """
        # Проверяем, существует ли анализатор
        if not self.analyzer:
            warning_msg = "Сначала решите задачу (встроенный)."
            messagebox.showwarning("Предупреждение", warning_msg)
            logger.warning(warning_msg)
            return
        try:
            # Рассчитываем минимальные цены для невыгодных грузов
            self.analyzer.calculate_min_price_for_unprofitable_cargos(self.cargos)
            # Генерируем полный отчёт
            report = self.analyzer.generate_stability_report()
            # Отображаем отчёт в текстовом поле
            self.report_text.config(state=NORMAL)
            self.report_text.delete(1.0, END)
            self.report_text.insert(END, report)
            self.report_text.config(state=DISABLED)
            logger.info("Отчёт об устойчивости сгенерирован.")
        except Exception as e:
            error_msg = f"Ошибка при генерации отчёта: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _calculate_scenarios(self):
        """
        Выполняет сценарный анализ для встроенного решения.
        """
        # Проверяем, существуют ли анализатор и исходный вектор b
        if not self.analyzer or self.original_b is None:
            warning_msg = "Сначала решите задачу (встроенный)."
            messagebox.showwarning("Предупреждение", warning_msg)
            logger.warning(warning_msg)
            return
        try:
            # Определяем сценарии
            scenarios = {
                "Сценарий 1: Пиломатериалы": {"cargo_index": 5, "new_availability": 400},
                "Сценарий 2: Бумага": {"cargo_index": 2, "new_availability": 900},
                "Сценарий 3: Контейнеры": {"cargo_index": 3, "new_availability": 100}
            }
            # Формируем заголовок отчёта
            report = "=" * 120 + "\nАНАЛИЗ СЦЕНАРИЕВ (встроенный)\n" + "=" * 120 + "\n"
            # Получаем текущую прибыль
            current_profit = self.solver.get_objective_value()
            report += f"Текущая прибыль: {current_profit:.2f}\n"

            for name, data in scenarios.items():
                cargo_idx = data["cargo_index"]
                # Находим имя ограничения по наличию для этого груза
                constraint_name = f"Наличие_груз{cargo_idx}"
                # Получаем теневую цену для этого ограничения
                shadow_price = self.analyzer.shadow_prices.get(constraint_name, 0.0)
                # Получаем исходное наличие из данных груза
                old_avail = self.cargos[cargo_idx-1]['availability']
                # Вычисляем изменение
                delta = data["new_availability"] - old_avail
                # Приблизительное изменение прибыли = теневая_цена * изменение_ресурса
                approx_change = shadow_price * delta
                # Для точного расчёта нужно решить задачу заново с изменённым b
                # Индекс ограничения в векторе b: 6 (вес) + 3 (объём) + (cargo_idx - 1) = 9 + (cargo_idx - 1)
                b_new = self.original_b.copy()
                b_idx = 6 + (cargo_idx - 1) # 6 - смещение после веса и объёма
                b_new[b_idx] = data["new_availability"] # Меняем значение
                # Решаем задачу с новым b
                success, _, new_val = self.solver.solve_with_modified_b(b_new)
                exact_change = new_val - current_profit if success else float('nan')
                # Добавляем строку в отчёт
                report += f"{name}: Δ≈{approx_change:+.2f}, Δточно={exact_change:+.2f}\n"

            # Отображаем отчёт в текстовом поле
            self.scenario_text.config(state=NORMAL)
            self.scenario_text.delete(1.0, END)
            self.scenario_text.insert(END, report)
            self.scenario_text.config(state=DISABLED)
            logger.info("Сценарный анализ выполнен.")

        except Exception as e:
            error_msg = f"Ошибка при расчёте сценариев: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _plot_results(self):
        """
        Строит графики на основе оптимального решения.
        """
        # Проверяем, существует ли решатель
        if not self.solver:
            warning_msg = "Сначала решите задачу (встроенный)."
            messagebox.showwarning("Предупреждение", warning_msg)
            logger.warning(warning_msg)
            return
        try:
            # Получаем оптимальный план
            optimal_plan = self.solver.get_optimal_plan()
            # Группируем количество по типам грузов (суммируем по отсекам)
            cargo_totals = {}
            for cargo in self.cargos:
                i = cargo['index']
                total = sum(optimal_plan.get(f"x{i}{j}", 0.0) for j in range(1, 4))
                cargo_totals[cargo['name']] = total

            # Очищаем текущий график
            self.ax.clear()
            # Подготовим данные для bar plot
            names = list(cargo_totals.keys())
            amounts = list(cargo_totals.values())
            # Построим столбчатую диаграмму
            self.ax.bar(names, amounts, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
            self.ax.set_title('Распределение грузов (встроенный)')
            self.ax.tick_params(axis='x', rotation=45)
            # Обновим canvas
            self.canvas_agg.draw()
            logger.info("График построен.")

        except Exception as e:
            error_msg = f"Ошибка при построении графика: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _save_plot(self):
        """
        Сохраняет текущий график в файл.
        """
        # Проверяем, существует ли фигура matplotlib
        if not hasattr(self, 'fig'):
            logger.warning("Попытка сохранить график до его построения.")
            return
        try:
            # Диалог выбора имени файла
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                # Сохраняем фигуру
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"График сохранён: {filename}")
                logger.info(f"График сохранён: {filename}")
        except Exception as e:
            error_msg = f"Ошибка при сохранении графика: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _export_to_csv(self):
        """
        Экспортирует результаты в файл CSV с кодировкой UTF-8.
        """
        # Проверяем, есть ли решённые задачи
        export_default = self.solver is not None
        export_manual = self.manual_solver is not None

        if not export_default and not export_manual:
            warning_msg = "Нет решённых задач для экспорта."
            messagebox.showwarning("Предупреждение", warning_msg)
            logger.warning(warning_msg)
            return

        # Диалог выбора имени файла
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filename:
            return # Пользователь отменил

        try:
            # Открываем файл на запись с кодировкой UTF-8
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # Создаём writer с точкой с запятой в качестве разделителя
                writer = csv.writer(f, delimiter=';')
                # Заголовок
                writer.writerow(["Лабораторная работа №5: Симплекс-метод"])
                # Экспорт результата встроенного решения
                if export_default:
                    writer.writerow(["Тип задачи: Встроенный вариант"])
                    optimal_plan = self.solver.get_optimal_plan()
                    optimal_value = self.solver.get_objective_value()
                    writer.writerow([f"Макс. прибыль (встроенный): {optimal_value:.2f}"])
                    for cargo in self.cargos:
                        i = cargo['index']
                        row = [cargo['name']]
                        total = 0.0
                        for j in range(1, 4):
                            var_name = f"x{i}{j}"
                            amount = optimal_plan.get(var_name, 0.0)
                            row.append(f"{amount:.2f}")
                            total += amount
                        row.append(f"{total:.2f}")
                        writer.writerow(row)
                # Экспорт результата ручного решения
                if export_manual:
                    writer.writerow(["Тип задачи: Ручной ввод"])
                    opt_plan = self.manual_solver.get_optimal_plan()
                    opt_val = self.manual_solver.get_objective_value()
                    writer.writerow([f"Макс. прибыль (ручной): {opt_val:.2f}"])
                    for i, cargo in enumerate(self.manual_cargos):
                        total = 0.0
                        row = [cargo['name']]
                        for j, comp in enumerate(self.manual_compartments):
                            var_name = f"x{i+1}{j+1}"
                            amount = opt_plan.get(var_name, 0.0)
                            row.append(f"{amount:.2f}")
                            total += amount
                        row.append(f"{total:.2f}")
                        writer.writerow(row)
            messagebox.showinfo("Успех", f"Результаты экспортированы в {filename}")
            logger.info(f"Результаты экспортированы в CSV: {filename}")

        except Exception as e:
            error_msg = f"Ошибка при экспорте в CSV: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _export_to_json(self):
        """
        Экспортирует результаты в файл JSON с кодировкой UTF-8.
        """
        # Проверяем, есть ли решённые задачи
        export_default = self.solver is not None
        export_manual = self.manual_solver is not None

        if not export_default and not export_manual:
            warning_msg = "Нет решённых задач для экспорта."
            messagebox.showwarning("Предупреждение", warning_msg)
            logger.warning(warning_msg)
            return

        # Диалог выбора имени файла
        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                               filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not filename:
            return # Пользователь отменил

        try:
            # Подготовим данные для экспорта
            export_data = {
                "metadata": {
                    "lab_number": 5,
                    "author": "Колосов Станислав",
                    "group": "ИВТ-3",
                    "year": 2026
                }
            }
            # Добавим результат встроенного решения
            if export_default:
                opt_plan = self.solver.get_optimal_plan()
                opt_val = self.solver.get_objective_value()
                export_data["solution_builtin"] = {
                    "optimal_value": opt_val,
                    "plan": {k: round(v, 2) for k, v in opt_plan.items()} # Округлим для читаемости
                }
            # Добавим результат ручного решения
            if export_manual:
                opt_plan_m = self.manual_solver.get_optimal_plan()
                opt_val_m = self.manual_solver.get_objective_value()
                export_data["solution_manual"] = {
                    "optimal_value": opt_val_m,
                    "plan": {k: round(v, 2) for k, v in opt_plan_m.items()}
                }
            # Запишем данные в файл с кодировкой UTF-8
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2) # ensure_ascii=False для кириллицы
            messagebox.showinfo("Успех", f"Результаты экспортированы в {filename}")
            logger.info(f"Результаты экспортированы в JSON: {filename}")

        except Exception as e:
            error_msg = f"Ошибка при экспорте в JSON: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            logger.error(error_msg)


    def _clear_results(self):
        """
        Очищает все результаты и сбрасывает состояния решателей.
        """
        # Подтверждение действия
        if not messagebox.askyesno("Подтверждение", "Очистить все результаты?"):
            return

        # Сбрасываем переменные
        self.solver = None
        self.analyzer = None
        self.original_b = None
        self.manual_solver = None
        self.manual_analyzer = None
        self.manual_original_b = None

        # Очищаем виджеты
        self.table_text.delete(1.0, END)
        self.report_text.delete(1.0, END)
        self.scenario_text.delete(1.0, END)
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)
        self.profit_label.config(text="Максимальная прибыль: —")
        self.status_label.config(text="Статус (встроенный): Результаты очищены", fg="#7f8c8d")
        self.progress_var.set(0)
        self.ax.clear()
        self.canvas_agg.draw()
        logger.info("Результаты очищены.")


    def _show_about(self):
        """
        Показывает окно "О программе".
        """
        about_text = """Лабораторная работа №5 по курсу "Основы искусственного интеллекта"
Тема: Решение прямой задачи линейного программирования симплексным методом

Вариант №1: Оптимизация загрузки грузопассажирского судна
Отсеки: №1, №2, №3
Грузы: №1-5

Реализация: Колосов Станислав
Группа: ИВТ-3
Год: 2026

Технологии: Python 3.13, tkinter, numpy, matplotlib

Функционал:
- Решение встроенного варианта задачи
- Ручной ввод параметров задачи и её решение
- Анализ устойчивости, сценарии, графики, экспорт для обоих режимов"""
        messagebox.showinfo("О программе", about_text)


def main():
    """
    Точка входа в приложение.
    """
    root = Tk()
    app = SimplexApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
