# main.py (working version, tested, with improvements and detailed comments)
"""
Main module for Lab Work #5 on the course "Fundamentals of Artificial Intelligence".
Solving a linear programming problem using the simplex method with a graphical interface.
Variant #1: Optimization of cargo loading for a cargo-passenger ship (compartments 1,2,3; cargos 1-5).
Manual input mode: user data.
Author: Stanislav Kolosov
Date: 2026
"""

import sys
import os
import csv
import json
import logging  # Adding logging
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Imports for error handling
try:
    from simplex_solver import SimplexSolver
    from sensitivity_analysis import SensitivityAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    messagebox.showerror("Error", f"Failed to import required modules: {e}\nCheck if simplex_solver.py and sensitivity_analysis.py are in the same folder.")
    sys.exit(1)

# --- ENCODING SETUP ---
# Python 3.13 uses UTF-8 for strings by default.
# When working with files, always specify encoding='utf-8' for correct Cyrillic handling.
# This is important for export, import, and data display in the interface.

## --- LOGGING SETUP ---
# Use logging for debugging and logging user actions and errors
logging.basicConfig(
    level=logging.INFO,  # Logging level (INFO, DEBUG, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.log", encoding='utf-8'),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

logger = logging.getLogger(__name__)  # Create a logger for this module

class SimplexApp:
    """
    Main application class managing the GUI and interaction logic with the solver.
    """

    def __init__(self, root: Tk):
        """
        Initializes the main window and all its components.
        """
        self.root = root
        self.root.title("Lab Work #5: Simplex Method (Variant #1)")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)

        ## --- DEFAULT DATA (Variant #1) ---
        # This data is used for built-in calculation
        self.default_compartments = [
            {'id': 1, 'volume': 500, 'max_weight': 700},
            {'id': 2, 'volume': 1000, 'max_weight': 800},
            {'id': 3, 'volume': 1500, 'max_weight': 1300}
        ]

        # Updated data for cargos (according to 9 фаза.pdf)
        self.default_cargos = [
            {'index': 1, 'name': 'Mini-tractors', 'weight': 0.35, 'volume': 3.0, 'price': 8.0, 'availability': 100},
            {'index': 2, 'name': 'Paper', 'weight': 1.6, 'volume': 1.0, 'price': 21.5, 'availability': 1000},
            {'index': 3, 'name': 'Containers', 'weight': 5.0, 'volume': 6.5, 'price': 51.0, 'availability': 200},
            {'index': 4, 'name': 'Rolled metal', 'weight': 35.0, 'volume': 6.0, 'price': 275.0, 'availability': 200},
            {'index': 5, 'name': 'Timber', 'weight': 4.0, 'volume': 6.0, 'price': 110.0, 'availability': 350}
        ]

        # --- VARIABLES FOR BUILT-IN VARIANT ---
        self.compartments = self.default_compartments
        self.cargos = self.default_cargos

        # Variables for storing solver and analyzer instances
        self.solver: SimplexSolver = None
        self.analyzer: SensitivityAnalyzer = None

        # Variable for storing the original vector b (for scenario analysis)
        self.original_b = None

        # --- VARIABLES FOR MANUAL INPUT ---
        # Lists for storing user-entered data
        self.manual_compartments = []
        self.manual_cargos = []

        # Variables for storing solver and analyzer instances for manual input
        self.manual_solver: SimplexSolver = None
        self.manual_analyzer: SensitivityAnalyzer = None

        # Variable for storing the original vector b for manual input
        self.manual_original_b = None

        # --- INTERFACE VARIABLES ---
        # Variables for storing references to widgets, tables, and plots
        self.figures = {}
        self.canvas_widgets = {}
        self.progress_var = DoubleVar()  # Variable for progress bar

        # Variables for storing references to StringVar for input fields in manual mode
        self.manual_comp_vars = []  # [{'id': StringVar, 'volume': StringVar, 'max_weight': StringVar, 'frame': Widget}, ...]
        self.manual_cargo_vars = []  # [{'name': StringVar, 'weight': StringVar, ..., 'frame': Widget}, ...]

        # --- CREATE INTERFACE ---
        # Call helper methods to build menu, tabs, and widgets
        self._create_menu()
        self._create_notebook()
        self._create_input_tab()
        self._create_solution_tab()
        self._create_sensitivity_tab()
        self._create_scenarios_tab()
        self._create_results_tab()
        self._create_plots_tab()
        self._create_manual_input_tab()

        # Apply styles to widgets
        self._style_widgets()

        # Display variant data on the "Input Data" tab
        self._display_variant_data()

        # Log successful initialization
        logger.info("SimplexApp initialized.")

    def _create_menu(self):
        """
        Creates the top menu of the application (File, Actions, Help).
        """
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File submenu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export results to CSV", command=self._export_to_csv)
        file_menu.add_command(label="Export results to JSON", command=self._export_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Actions submenu
        action_menu = Menu(menubar, tearoff=0)
        action_menu.add_command(label="Solve problem (built-in)", command=self._solve_problem)
        action_menu.add_command(label="Solve problem (manual input)", command=self._solve_manual_problem)
        action_menu.add_command(label="Clear results", command=self._clear_results)
        menubar.add_cascade(label="Actions", menu=action_menu)

        # Help submenu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _create_notebook(self):
        """
        Creates tabs (Notebook) to separate application functionality.
        """
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # Create frames for each tab
        self.tab_input = ttk.Frame(self.notebook)
        self.tab_solution = ttk.Frame(self.notebook)
        self.tab_sensitivity = ttk.Frame(self.notebook)
        self.tab_scenarios = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)
        self.tab_plots = ttk.Frame(self.notebook)
        self.tab_manual_input = ttk.Frame(self.notebook)

        # Add tabs with names
        self.notebook.add(self.tab_input, text="Input Data")
        self.notebook.add(self.tab_solution, text="Solution (Built-in)")
        self.notebook.add(self.tab_sensitivity, text="Sensitivity Analysis (Built-in)")
        self.notebook.add(self.tab_scenarios, text="Scenarios (Built-in)")
        self.notebook.add(self.tab_results, text="Results (Built-in)")
        self.notebook.add(self.tab_plots, text="Plots (Built-in)")
        self.notebook.add(self.tab_manual_input, text="Manual Input")

        # Create progress bar at the bottom of the main window
        self.progress_frame = Frame(self.root)
        self.progress_frame.pack(fill=X, padx=10, pady=5)
        Label(self.progress_frame, text="Progress:").pack(side=LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=LEFT, fill=X, expand=True, padx=5)

    def _create_input_tab(self):
        """
        Creates the "Input Data" tab with display of ship and cargo parameters.
        """
        header = Label(self.tab_input, text="Variant #1: Ship Loading Optimization", font=("Arial", 14, "bold"), fg="#2c3e50")
        header.pack(pady=10)

        # --- Compartments frame ---
        frame_compartments = LabelFrame(self.tab_input, text="Ship Compartments", font=("Arial", 10, "bold"))
        frame_compartments.pack(fill=X, padx=15, pady=5)
        tree_comp = ttk.Treeview(frame_compartments, columns=("id", "volume", "weight"), show="headings", height=3)
        tree_comp.heading("id", text="Compartment #")
        tree_comp.heading("volume", text="Volume (m³)")
        tree_comp.heading("weight", text="Max Weight (t)")
        tree_comp.column("id", width=100, anchor=CENTER)
        tree_comp.column("volume", width=150, anchor=CENTER)
        tree_comp.column("weight", width=150, anchor=CENTER)
        for comp in self.compartments:
            tree_comp.insert("", "end", values=(comp['id'], comp['volume'], comp['max_weight']))
        tree_comp.pack(fill=X, padx=5, pady=5)

        # --- Cargo frame ---
        frame_cargos = LabelFrame(self.tab_input, text="Cargo Types", font=("Arial", 10, "bold"))
        frame_cargos.pack(fill=X, padx=15, pady=15)
        tree_cargo = ttk.Treeview(frame_cargos, columns=("idx", "name", "weight", "volume", "price", "avail"), show="headings", height=5)
        tree_cargo.heading("idx", text="#")
        tree_cargo.heading("name", text="Cargo Type")
        tree_cargo.heading("weight", text="Weight (t)")
        tree_cargo.heading("volume", text="Volume (m³)")
        tree_cargo.heading("price", text="Price (monetary units)")
        tree_cargo.heading("avail", text="Availability (units)")
        for col, width, align in [("idx", 120, CENTER), ("name", 120, W), ("weight", 120, CENTER),
                                  ("volume", 120, CENTER), ("price", 120, CENTER), ("avail", 120, CENTER)]:
            tree_cargo.column(col, width=width, anchor=align)
        for cargo in self.cargos:
            tree_cargo.insert("", "end", values=(
                cargo['index'],
                cargo['name'],
                f"{cargo['weight']:.2f}", f"{cargo['volume']:.1f}", f"{cargo['price']:.1f}", cargo['availability']
            ))
        tree_cargo.pack(fill=X, padx=5, pady=5)

        # --- Mathematical model frame ---
        frame_model = LabelFrame(self.tab_input, text="Mathematical Model", font=("Arial", 10, "bold"))
        frame_model.pack(fill=BOTH, expand=True, padx=15, pady=5)
        model_text = scrolledtext.ScrolledText(frame_model, wrap=WORD, height=12, font=("Courier New", 10))
        model_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        model_content = """Objective function: max F = 8·(x11+x12+x13) + 21.5·(x21+x22+x23) + 51·(x31+x32+x33) + 275·(x41+x42+x43) + 110·(x51+x52+x53) → max

Constraints:

1. By compartment weight:
0.35x11 + 1.6x21 + 5x31 + 35x41 + 4x51 ≤ 700 (compartment 1)
0.35x12 + 1.6x22 + 5x32 + 35x42 + 4x52 ≤ 800 (compartment 2)
0.35x13 + 1.6x23 + 5x33 + 35x43 + 4x53 ≤ 1300 (compartment 3)

2. By compartment volume:
3x₁₁ + 1x₂₁ + 6.5x₃₁ + 6x₄₁ + 6x₅₁ ≤ 500  (compartment 1)
3x₁₂ + 1x₂₂ + 6.5x₃₂ + 6x₄₂ + 6x₅₂ ≤ 1000  (compartment 2)
3x₁₃ + 1x₂₃ + 6.5x₃₃ + 6x₄₃ + 6x₅₃ ≤ 1500  (compartment 3)

3. By cargo availability:
x₁₁ + x₁₂ + x₁₃ ≤ 100  (mini-tractors)
x₂₁ + x₂₂ + x₂₃ ≤ 1000  (paper)
x₃₁ + x₃₂ + x₃₃ ≤ 200  (containers)
x₄₁ + x₄₂ + x₄₃ ≤ 200  (rolled metal)
x₅₁ + x₅₂ + x₅₃ ≤ 350  (timber)

4. Non-negativity:
xᵢⱼ ≥ 0 for all i=1..5, j=1..3

Decision variables: xᵢⱼ — quantity of cargo type i in compartment j (total 15 variables)
"""
        model_text.insert(END, model_content)
        model_text.config(state=DISABLED)  # Make text read-only

    def _create_solution_tab(self):
        """
        Creates the "Solution (Built-in)" tab with buttons, table, and status.
        """
        control_frame = Frame(self.tab_solution)
        control_frame.pack(fill=X, padx=10, pady=5)
        Label(control_frame, text="Select iteration (built-in):", font=("Arial", 10)).pack(side=LEFT, padx=5)
        self.iteration_var = StringVar(value="0")
        self.iteration_combo = ttk.Combobox(control_frame, textvariable=self.iteration_var, width=10, state="readonly")
        self.iteration_combo.pack(side=LEFT, padx=5)
        # Bind selection event to table display method
        self.iteration_combo.bind("<<ComboboxSelected>>", self._show_selected_iteration)
        Button(control_frame, text="Solve problem (built-in)", command=self._solve_problem,
               bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=RIGHT, padx=5)

        table_frame = LabelFrame(self.tab_solution, text="Simplex Table (Built-in)", font=("Arial", 10, "bold"))
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.table_text = scrolledtext.ScrolledText(table_frame, wrap=NONE, font=("Courier New", 9))
        self.table_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.table_text.config(state=DISABLED)

        self.status_label = Label(self.tab_solution, text="Status (built-in): Waiting for solution...", font=("Arial", 10, "italic"), fg="#7f8c8d")
        self.status_label.pack(pady=5)

    def _create_sensitivity_tab(self):
        """
        Creates the "Sensitivity Analysis (Built-in)" tab.
        """
        btn_frame = Frame(self.tab_sensitivity)
        btn_frame.pack(fill=X, padx=10, pady=10)
        Button(btn_frame, text="Generate stability report (built-in)", command=self._generate_sensitivity_report,
               bg="#27ae60", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT)
        report_frame = LabelFrame(self.tab_sensitivity, text="Stability Report (Built-in)", font=("Arial", 10, "bold"))
        report_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=WORD, font=("Courier New", 9))
        self.report_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.report_text.config(state=DISABLED)

    def _create_scenarios_tab(self):
        """
        Creates the "Scenarios (Built-in)" tab.
        """
        desc_frame = LabelFrame(self.tab_scenarios, text="Scenario Descriptions (Built-in)", font=("Arial", 10, "bold"))
        desc_frame.pack(fill=X, padx=10, pady=5)
        desc_text = """Scenarios:
1. Timber: 400 units (instead of 350)
2. Paper: 900 units (instead of 1000)
3. Containers: 100 units (instead of 200)"""
        Label(desc_frame, text=desc_text, font=("Arial", 10), justify=LEFT, fg="#2c3e50").pack(padx=10, pady=10)
        btn_frame = Frame(self.tab_scenarios)
        btn_frame.pack(fill=X, padx=10, pady=10)
        Button(btn_frame, text="Calculate scenarios (built-in)", command=self._calculate_scenarios,
               bg="#e67e22", fg="white", font=("Arial", 10, "bold"), padx=15).pack()
        result_frame = LabelFrame(self.tab_scenarios, text="Results (Built-in)", font=("Arial", 10, "bold"))
        result_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        self.scenario_text = scrolledtext.ScrolledText(result_frame, wrap=WORD, font=("Arial", 10))
        self.scenario_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.scenario_text.config(state=DISABLED)

    def _create_results_tab(self):
        """
        Creates the "Results (Built-in)" tab with the optimal plan table.
        """
        header = Label(self.tab_results, text="Optimal Plan (Built-in)", font=("Arial", 14, "bold"), fg="#2c3e50")
        header.pack(pady=10)
        plan_frame = LabelFrame(self.tab_results, text="Cargo Distribution (Built-in)", font=("Arial", 10, "bold"))
        plan_frame.pack(fill=BOTH, expand=True, padx=15, pady=5)
        columns = ["Cargo", "Compartment 1", "Compartment 2", "Compartment 3", "Total"]
        self.plan_tree = ttk.Treeview(plan_frame, columns=columns, show="headings", height=6)
        for col, width in zip(columns, [150, 120, 120, 120, 120]):
            self.plan_tree.heading(col, text=col)
            self.plan_tree.column(col, width=width, anchor=CENTER if col != "Cargo" else W)
        self.plan_tree.pack(fill=BOTH, expand=True, padx=5, pady=5)
        profit_frame = Frame(self.tab_results)
        profit_frame.pack(fill=X, padx=15, pady=15)
        self.profit_label = Label(profit_frame, text="Maximum profit: —", font=("Arial", 12, "bold"), fg="#e74c3c")
        self.profit_label.pack(side=LEFT)

    def _create_plots_tab(self):
        """
        Creates the "Plots (Built-in)" tab with a plotting area.
        """
        plot_control_frame = Frame(self.tab_plots)
        plot_control_frame.pack(fill=X, padx=10, pady=5)
        Button(plot_control_frame, text="Plot results (built-in)", command=self._plot_results,
               bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT, padx=5)
        Button(plot_control_frame, text="Save plot (built-in)", command=self._save_plot,
               bg="#9b59b6", fg="white", font=("Arial", 10, "bold"), padx=15).pack(side=LEFT, padx=5)
        self.plot_canvas_frame = LabelFrame(self.tab_plots, text="Plots (Built-in)", font=("Arial", 10, "bold"))
        self.plot_canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        # Initialize matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # Create Canvas for embedding matplotlib in tkinter
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas_agg.get_tk_widget().pack(fill=BOTH, expand=True)
        # Draw initial (empty) plot
        self.canvas_agg.draw()

    def _create_manual_input_tab(self):
        """
        Creates the "Manual Input" tab with the ability to dynamically add compartments and cargos.
        """
        manual_frame = Frame(self.tab_manual_input)
        manual_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # --- Compartments frame ---
        comp_frame = LabelFrame(manual_frame, text="Compartments", font=("Arial", 10, "bold"))
        comp_frame.pack(fill=X, padx=5, pady=5)
        self.comp_entries_frame = Frame(comp_frame)
        self.comp_entries_frame.pack(fill=X, padx=5, pady=5)
        self.add_comp_btn = Button(comp_frame, text="Add Compartment", command=self._add_comp_entry)
        self.add_comp_btn.pack(side=TOP, padx=5, pady=5)

        # --- Cargo frame ---
        cargo_frame = LabelFrame(manual_frame, text="Cargo Types", font=("Arial", 10, "bold"))
        cargo_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.cargo_entries_frame = Frame(cargo_frame)
        self.cargo_entries_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        self.add_cargo_btn = Button(cargo_frame, text="Add Cargo", command=self._add_cargo_entry)
        self.add_cargo_btn.pack(side=TOP, padx=5, pady=5)

        # --- Solve button ---
        solve_manual_btn = Button(manual_frame, text="Calculate (Manual Input)", command=self._solve_manual_problem,
                                  bg="#3498db", fg="white", font=("Arial", 10, "bold"), padx=15)
        solve_manual_btn.pack(pady=10)

    def _add_comp_entry(self):
        """
        Adds a row of input fields for a new compartment to the frame self.comp_entries_frame.
        """
        frame = Frame(self.comp_entries_frame)
        frame.pack(fill=X, padx=2, pady=2)
        Label(frame, text="ID:").grid(row=0, column=0, sticky=W, padx=2)
        # ID is auto-generated, field is read-only
        id_var = StringVar(value=str(len(self.manual_comp_vars) + 1))
        Entry(frame, textvariable=id_var, width=5, state="disabled").grid(row=0, column=1, padx=2)
        Label(frame, text="Volume:").grid(row=0, column=2, sticky=W, padx=2)
        vol_var = StringVar()
        Entry(frame, textvariable=vol_var, width=10).grid(row=0, column=3, padx=2)
        Label(frame, text="Weight:").grid(row=0, column=4, sticky=W, padx=2)
        weight_var = StringVar()
        Entry(frame, textvariable=weight_var, width=10).grid(row=0, column=5, padx=2)
        remove_btn = Button(frame, text="Remove", command=lambda f=frame: self._remove_comp_entry(f))
        remove_btn.grid(row=0, column=6, padx=5)
        # Save references to StringVar and frame
        self.manual_comp_vars.append({'id': id_var, 'volume': vol_var, 'max_weight': weight_var, 'frame': frame})

    def _remove_comp_entry(self, frame):
        """
        Removes a compartment input row and clears the corresponding entry from the list.
        """
        frame.destroy()
        # Update the list, removing the entry corresponding to the deleted frame
        self.manual_comp_vars = [v for v in self.manual_comp_vars if v['frame'] != frame]

    def _add_cargo_entry(self):
        """
        Adds a row of input fields for a new cargo to the frame self.cargo_entries_frame.
        """
        frame = Frame(self.cargo_entries_frame)
        frame.pack(fill=X, padx=2, pady=2)
        Label(frame, text="Name:").grid(row=0, column=0, sticky=W, padx=2)
        name_var = StringVar()
        Entry(frame, textvariable=name_var, width=15).grid(row=0, column=1, padx=2)
        Label(frame, text="Weight:").grid(row=0, column=2, sticky=W, padx=2)
        weight_var = StringVar()
        Entry(frame, textvariable=weight_var, width=10).grid(row=0, column=3, padx=2)
        Label(frame, text="Volume:").grid(row=0, column=4, sticky=W, padx=2)
        volume_var = StringVar()
        Entry(frame, textvariable=volume_var, width=10).grid(row=0, column=5, padx=2)
        Label(frame, text="Price:").grid(row=0, column=6, sticky=W, padx=2)
        price_var = StringVar()
        Entry(frame, textvariable=price_var, width=10).grid(row=0, column=7, padx=2)
        Label(frame, text="Availability:").grid(row=0, column=8, sticky=W, padx=2)
        avail_var = StringVar()
        Entry(frame, textvariable=avail_var, width=10).grid(row=0, column=9, padx=2)
        remove_btn = Button(frame, text="Remove", command=lambda f=frame: self._remove_cargo_entry(f))
        remove_btn.grid(row=0, column=10, padx=5)
        # Save references to StringVar and frame
        self.manual_cargo_vars.append({
            'name': name_var, 'weight': weight_var, 'volume': volume_var,
            'price': price_var, 'availability': avail_var, 'frame': frame
        })

    def _remove_cargo_entry(self, frame):
        """
        Removes a cargo input row and clears the corresponding entry from the list.
        """
        frame.destroy()
        # Update the list, removing the entry corresponding to the deleted frame
        self.manual_cargo_vars = [v for v in self.manual_cargo_vars if v['frame'] != frame]

    def _solve_manual_problem(self):
        """
        Collects data from input fields on the "Manual Input" tab, builds an LP problem, solves it, and displays results.
        """
        logger.info("Starting manual input problem solution.")
        try:
            # --- Collect compartment data ---
            compartments_data = []
            for var_dict in self.manual_comp_vars:
                # Validate input: try to convert to float
                vol_str = var_dict['volume'].get().strip()
                weight_str = var_dict['max_weight'].get().strip()
                if not vol_str or not weight_str:
                    messagebox.showerror("Input Error", f"'Volume' or 'Weight' field for compartment {var_dict['id'].get()} is empty.")
                    return
                try:
                    vol = float(vol_str)
                    weight = float(weight_str)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid number in 'Volume' or 'Weight' for compartment {var_dict['id'].get()}.")
                    return
                if vol <= 0 or weight <= 0:
                    raise ValueError("Volume and weight must be positive.")
                compartments_data.append({'id': int(var_dict['id'].get()), 'volume': vol, 'max_weight': weight})

            # --- Collect cargo data ---
            cargos_data = []
            for i, var_dict in enumerate(self.manual_cargo_vars):
                name = var_dict['name'].get().strip()
                if not name:
                    raise ValueError("Name cannot be empty.")
                # Validate input: try to convert to numbers
                weight_str = var_dict['weight'].get().strip()
                volume_str = var_dict['volume'].get().strip()
                price_str = var_dict['price'].get().strip()
                avail_str = var_dict['availability'].get().strip()
                if not all([weight_str, volume_str, price_str, avail_str]):
                    messagebox.showerror("Input Error", f"One of the fields for cargo '{name}' is empty.")
                    return
                try:
                    weight = float(weight_str)
                    volume = float(volume_str)
                    price = float(price_str)
                    availability = int(avail_str)  # Availability is usually an integer
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid number in one of the fields for cargo '{name}'.")
                    return
                if weight <= 0 or volume <= 0 or price <= 0 or availability < 0:
                    raise ValueError("Weight, volume, price must be positive. Availability >= 0.")
                cargos_data.append({
                    'index': i+1, 'name': name, 'weight': weight, 'volume': volume,
                    'price': price, 'availability': availability
                })

            # Check that at least one compartment and one cargo are entered
            if not compartments_data or not cargos_data:
                messagebox.showwarning("Warning", "Enter at least one compartment and one cargo.")
                return

            # --- Build and solve the problem ---
            c, A, b, var_names, constraint_names = self._build_manual_problem_matrices(compartments_data, cargos_data)
            self.manual_original_b = b.copy()  # Save the original vector b
            self.manual_solver = SimplexSolver(c, A, b, var_names, constraint_names)
            success, solution, optimal_value = self.manual_solver.solve(max_iterations=100)
            if not success:
                messagebox.showerror("Error", "Failed to find an optimal solution for the manual input problem!")
                logger.error("No solution found for manual input problem.")
                return

            # --- Display result ---
            result_text = f"Manual input problem solved!\nMaximum profit: {optimal_value:.2f} monetary units.\n\nOptimal plan:\n"
            for i, cargo in enumerate(cargos_data):
                cargo_total_amount = 0.0
                cargo_result_line = f"{cargo['name']}: "
                for j, comp in enumerate(compartments_data):
                    var_idx = i * len(compartments_data) + j
                    amount = solution[var_idx]
                    if amount > 1e-5:  # Show only non-zero values
                        cargo_result_line += f"{comp['id']}({amount:.2f}), "
                        cargo_total_amount += amount
                if cargo_total_amount > 1e-5:
                    result_text += f"{cargo_result_line[:-2]} (Total: {cargo_total_amount:.2f})\n"
                else:
                    result_text += f"{cargo['name']}: 0 (not used)\n"
            messagebox.showinfo("Result (Manual Input)", result_text)
            logger.info(f"Manual input problem solved. Profit: {optimal_value:.2f}")
            # --- Create analyzer for manual solution ---
            self.manual_analyzer = SensitivityAnalyzer(self.manual_solver)

        except ValueError as ve:
            # Catch errors related to data validation (non-numbers, negatives, etc.)
            messagebox.showerror("Input Error", f"Data error: {str(ve)}")
            logger.error(f"Input error in _solve_manual_problem: {ve}")
        except Exception as e:
            # Catch any other errors
            messagebox.showerror("Error", f"Error solving manual input problem:\n{str(e)}")
            logger.error(f"Error in _solve_manual_problem: {e}")

    def _build_manual_problem_matrices(self, compartments_data: List[Dict], cargos_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Builds matrices c, A, b, and lists of names for the LP problem based on manual data.

        Args:
            compartments_data (List[Dict]): Compartment data.
            cargos_data (List[Dict]): Cargo data.

        Returns:
            Tuple[c, A, b, var_names, constraint_names]: Matrices and lists for SimplexSolver.
        """
        num_compartments = len(compartments_data)
        num_cargos = len(cargos_data)
        n = num_cargos * num_compartments  # Total number of decision variables x_ij
        # m = num_compartments (weight) + num_compartments (volume) + num_cargos (availability)
        m = num_compartments * 2 + num_cargos

        # --- Variable names ---
        # x_11, x_12, ..., x_1C, x_21, ..., x_R1, ..., x_RC
        var_names = [f"x{i+1}{j+1}" for i in range(num_cargos) for j in range(num_compartments)]

        # --- Vector c (objective function coefficients) ---
        # c = [price_1, price_1, ..., price_1 (C times), price_2, ..., price_R (C times)]
        c = np.array([cargo['price'] for cargo in cargos_data for _ in range(num_compartments)])

        # --- Matrix A and vector b (constraints) ---
        A = np.zeros((m, n))
        b = np.zeros(m)
        constraint_names = []

        # 1. Weight constraints for compartments
        for j, comp in enumerate(compartments_data):
            for i, cargo in enumerate(cargos_data):
                # Index of variable x_ij in vector x: i * num_compartments + j
                var_idx = i * num_compartments + j
                # Coefficient for x_ij in the weight constraint for compartment j: weight of cargo i
                A[j, var_idx] = cargo['weight']
                # Right-hand side: max_weight of compartment j
                b[j] = comp['max_weight']
                constraint_names.append(f"Weight_Comp{comp['id']}")

        # 2. Volume constraints for compartments
        offset_vol = num_compartments  # Offset for volume constraint rows
        for j, comp in enumerate(compartments_data):
            for i, cargo in enumerate(cargos_data):
                var_idx = i * num_compartments + j
                # Coefficient for x_ij in the volume constraint for compartment j: volume of cargo i
                A[offset_vol + j, var_idx] = cargo['volume']
                b[offset_vol + j] = comp['volume']
                constraint_names.append(f"Volume_Comp{comp['id']}")

        # 3. Availability constraints for cargos
        offset_avail = 2 * num_compartments  # Offset for availability constraint rows
        for i, cargo in enumerate(cargos_data):
            # For cargo i, sum x_i1 + x_i2 + ... + x_iC
            for j in range(num_compartments):
                var_idx = i * num_compartments + j
                # Coefficient for x_ij in the availability constraint for cargo i: 1
                A[offset_avail + i, var_idx] = 1.0
                # Right-hand side: availability of cargo i
                b[offset_avail + i] = cargo['availability']
                constraint_names.append(f"Availability_Cargo{cargo['index']}_{cargo['name']}")

        return c, A, b, var_names, constraint_names

    def _style_widgets(self):
        """
        Configures the appearance of widgets using ttk.Style.
        """
        style = ttk.Style()
        style.theme_use('clam')  # Choose theme
        style.configure('TNotebook.Tab', padding=[12, 8], font=("Arial", 10))
        style.configure('TFrame', background="#f5f5f5")
        style.configure('TLabelframe', background="#f5f5f5", font=("Arial", 10, "bold"))

    def _display_variant_data(self):
        """
        Displays the current variant data (default).
        Can be extended for dynamic data changes.
        """
        # In the current implementation, data is displayed in _create_input_tab
        pass

    def _solve_problem(self):
        """
        Solves the built-in LP problem for variant #1.
        """
        logger.info("Starting built-in problem solution.")
        try:
            # --- Build the problem ---
            c, A, b, var_names, constraint_names = self._build_problem_matrices()
            self.original_b = b.copy()  # Save the original vector b
            self.solver = SimplexSolver(c, A, b, var_names, constraint_names)

            # --- Solve ---
            # Add debug output to check problem correctness
            print("\n=== CHECKING PROBLEM BUILDING (Built-in) ===")
            print(f"Vector c (objective function, first 10): {c[:10]}...")
            print(f"Vector b (constraints): {b}")
            print(f"Matrix A (first 3 rows, first 10 columns): \n{A[:3, :10]}")
            print(f"Variable names (first 10): {var_names[:10]}")
            print(f"Constraint names: {constraint_names}")
            print("==============================>\n")

            success, solution, optimal_value = self.solver.solve(max_iterations=100)
            if not success:
                error_msg = "Failed to find an optimal solution for the built-in variant."
                messagebox.showerror("Error", error_msg)
                self.status_label.config(text="Status (built-in): Solution error", fg="#e74c3c")
                logger.error(error_msg)
                return

            print("\n=== SOLUTION RESULTS (Built-in) ===")
            print(f"Solution (first 10): {solution[:10]}...")
            print(f"Profit (F): {optimal_value}")
            print("==============================\n")

            # --- Update interface ---
            self._update_solution_display()
            self._update_results_display(solution, optimal_value)
            # Create analyzer for the found solution
            self.analyzer = SensitivityAnalyzer(self.solver)
            status_text = f"Status (built-in): Solution found! Profit = {optimal_value:.2f} monetary units."
            self.status_label.config(text=status_text, fg="#27ae60")
            messagebox.showinfo("Success", f"Built-in variant solved! Profit: {optimal_value:.2f} monetary units.")
            logger.info(f"Built-in problem solved. Profit: {optimal_value:.2f}")

        except Exception as e:
            error_msg = f"Error (built-in): {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Status (built-in): Error", fg="#e74c3c")
            logger.error(error_msg)

    def _build_problem_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Builds matrices c, A, b, and lists of names for the built-in problem of variant #1.

        Returns:
            Tuple[c, A, b, var_names, constraint_names]: Matrices and lists for SimplexSolver.
        """
        # --- Variable names ---
        # x11, x12, x13, x21, ..., x53 (total 15)
        var_names = [f"x{i}{j}" for i in range(1, 6) for j in range(1, 4)]

        # --- Vector c (objective function coefficients) ---
        # c = [price_1, price_1, price_1, price_2, ..., price_5, price_5, price_5]
        # For cargo i=1..5, the price is repeated 3 times (for j=1..3)
        c = np.array([cargo['price'] for cargo in self.cargos for _ in range(3)])

        # --- Matrix A and vector b (constraints) ---
        # m = 3 (weight) + 3 (volume) + 5 (availability) = 11
        # n = 15 (x_ij, i=1..5, j=1..3)
        A = np.zeros((11, 15))
        b = np.zeros(11)
        constraint_names = []

        # Helper function to calculate the index of variable x_ij in vector x
        # x_11 -> idx 0, x_12 -> idx 1, x_13 -> idx 2, x_21 -> idx 3, ...
        def var_idx(i, j):
            return (i - 1) * 3 + (j - 1)

        # 1. Weight constraints for compartments (3 pcs.)
        for j_idx, comp in enumerate(self.compartments):  # j_idx = 0, 1, 2 -> compartments 1, 2, 3
            comp_id = comp['id']  # 1, 2, 3
            for i_idx, cargo in enumerate(self.cargos):  # i_idx = 0..4 -> cargos 1..5
                cargo_idx = cargo['index']  # 1..5
                A[j_idx, var_idx(cargo_idx, comp_id)] = cargo['weight']
                b[j_idx] = comp['max_weight']
                constraint_names.append(f"Weight_Comp{comp_id}")

        # 2. Volume constraints for compartments (3 pcs.)
        for j_idx, comp in enumerate(self.compartments):
            comp_id = comp['id']
            for i_idx, cargo in enumerate(self.cargos):
                cargo_idx = cargo['index']
                A[3 + j_idx, var_idx(cargo_idx, comp_id)] = cargo['volume']
                b[3 + j_idx] = comp['volume']
                constraint_names.append(f"Volume_Comp{comp_id}")

        # 3. Availability constraints for cargos (5 pcs.)
        for i_idx, cargo in enumerate(self.cargos):  # i_idx = 0..4
            cargo_idx = cargo['index']  # 1..5
            for j_idx, comp in enumerate(self.compartments):  # j_idx = 0..2
                comp_id = comp['id']  # 1..3
                A[6 + i_idx, var_idx(cargo_idx, comp_id)] = 1.0
                b[6 + i_idx] = cargo['availability']
                constraint_names.append(f"Availability_Cargo{cargo_idx}_{cargo['name']}")

        return c, A, b, var_names, constraint_names

    def _update_solution_display(self):
        """
        Updates the Combobox with iteration selection and displays the first (or last) table.
        """
        # Get the number of iterations from the solver
        iterations_count = len(self.solver.iterations) if self.solver else 0
        # Update the list of values in the Combobox
        self.iteration_combo['values'] = [str(i) for i in range(iterations_count)]
        # Set the value to the last iteration (if any) or 0
        selected_iteration = str(iterations_count - 1) if iterations_count > 0 else "0"
        self.iteration_var.set(selected_iteration)
        # Display the selected table
        self._show_selected_iteration(None)

    def _show_selected_iteration(self, event):
        """
        Displays the simplex table for the selected iteration in the text field.
        """
        try:
            # Check if the solver exists
            if not self.solver:
                return
            # Get the iteration number from the Combobox
            iteration_num = int(self.iteration_var.get())
            # Get the string representation of the table
            table_str = self.solver.get_iteration_table(iteration_num)
            # Update the text field
            self.table_text.config(state=NORMAL)
            self.table_text.delete(1.0, END)
            self.table_text.insert(END, table_str)
            self.table_text.config(state=DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Error displaying table: {str(e)}")
            logger.error(f"Error in _show_selected_iteration: {e}")

    def _update_results_display(self, solution: np.ndarray, optimal_value: float):
        """
        Updates the table with the optimal plan on the "Results" tab.
        """
        # Clear current table rows
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)

        # Fill the table with data from the solution
        for cargo in self.cargos:
            i = cargo['index']  # Cargo index (1..5)
            row = [cargo['name']]  # First column - cargo name
            total = 0.0
            for j in range(1, 4):  # Iterate over compartments (1..3)
                var_name = f"x{i}{j}"  # Variable name x_ij
                idx = (i-1)*3 + (j-1)  # Variable index in the solution vector
                amount = solution[idx]  # Variable value from the solution
                row.append(f"{amount:.2f}")  # Add value for compartment j
                total += amount  # Sum for total
            row.append(f"{total:.2f}")  # Add total quantity
            self.plan_tree.insert("", "end", values=row)  # Insert row into the table

        # Update the label with maximum profit
        self.profit_label.config(text=f"Maximum profit: {optimal_value:.2f} monetary units.")

    def _generate_sensitivity_report(self):
        """
        Generates and displays a stability report for the solution.
        """
        # Check if the analyzer exists
        if not self.analyzer:
            warning_msg = "First solve the problem (built-in)."
            messagebox.showwarning("Warning", warning_msg)
            logger.warning(warning_msg)
            return

        try:
            # Calculate minimum prices for unprofitable cargos
            self.analyzer.calculate_min_price_for_unprofitable_cargos(self.cargos)
            # Generate full report
            report = self.analyzer.generate_stability_report()
            # Display report in the text field
            self.report_text.config(state=NORMAL)
            self.report_text.delete(1.0, END)
            self.report_text.insert(END, report)
            self.report_text.config(state=DISABLED)
            logger.info("Stability report generated.")
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _calculate_scenarios(self):
        """
        Performs scenario analysis for the built-in solution.
        """
        # Check if analyzer and original vector b exist
        if not self.analyzer or self.original_b is None:
            warning_msg = "First solve the problem (built-in)."
            messagebox.showwarning("Warning", warning_msg)
            logger.warning(warning_msg)
            return

        try:
            # Define scenarios
            scenarios = {
                "Scenario 1: Timber": {"cargo_index": 5, "new_availability": 400},
                "Scenario 2: Paper": {"cargo_index": 2, "new_availability": 900},
                "Scenario 3: Containers": {"cargo_index": 3, "new_availability": 100}
            }
            # Form report header
            report = "=" * 120 + "\nSCENARIO ANALYSIS (Built-in)\n" + "=" * 120 + "\n"
            # Get current profit
            current_profit = self.solver.get_objective_value()
            report += f"Current profit: {current_profit:.2f}\n"

            for name, data in scenarios.items():
                cargo_idx = data["cargo_index"]
                # Find the name of the availability constraint for this cargo
                constraint_name = f"Availability_Cargo{cargo_idx}"
                # Get the shadow price for this constraint
                shadow_price = self.analyzer.shadow_prices.get(constraint_name, 0.0)
                # Get the original availability from cargo data
                old_avail = self.cargos[cargo_idx-1]['availability']
                # Calculate change
                delta = data["new_availability"] - old_avail
                # Approximate change in profit = shadow_price * resource_change
                approx_change = shadow_price * delta
                # For exact calculation, solve the problem again with modified b
                # Index of constraint in vector b: 6 (weight) + 3 (volume) + (cargo_idx - 1) = 9 + (cargo_idx - 1)
                b_new = self.original_b.copy()
                b_idx = 6 + (cargo_idx - 1)  # Offset after weight and volume
                b_new[b_idx] = data["new_availability"]  # Change value
                # Solve the problem with new b
                success, _, new_val = self.solver.solve_with_modified_b(b_new)
                exact_change = new_val - current_profit if success else float('nan')
                # Add line to report
                report += f"{name}: Δ={approx_change:+.2f}, Δexact={exact_change:+.2f}\n"

            # Display report in the text field
            self.scenario_text.config(state=NORMAL)
            self.scenario_text.delete(1.0, END)
            self.scenario_text.insert(END, report)
            self.scenario_text.config(state=DISABLED)
            logger.info("Scenario analysis completed.")

        except Exception as e:
            error_msg = f"Error calculating scenarios: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _plot_results(self):
        """
        Plots graphs based on the optimal solution.
        """
        # Check if the solver exists
        if not self.solver:
            warning_msg = "First solve the problem (built-in)."
            messagebox.showwarning("Warning", warning_msg)
            logger.warning(warning_msg)
            return

        try:
            # Get the optimal plan
            optimal_plan = self.solver.get_optimal_plan()
            # Group quantities by cargo types (sum by compartments)
            cargo_totals = {}
            for cargo in self.cargos:
                i = cargo['index']
                total = sum(optimal_plan.get(f"x{i}{j}", 0.0) for j in range(1, 4))
                cargo_totals[cargo['name']] = total

            # Clear current plot
            self.ax.clear()
            # Prepare data for bar plot
            names = list(cargo_totals.keys())
            amounts = list(cargo_totals.values())
            # Plot bar chart
            self.ax.bar(names, amounts, color=['skyblue', 'lightgreen', 'salmon', 'gold', 'plum'])
            self.ax.set_title('Cargo Distribution (Built-in)')
            self.ax.tick_params(axis='x', rotation=45)
            # Update canvas
            self.canvas_agg.draw()
            logger.info("Plot drawn.")

        except Exception as e:
            error_msg = f"Error plotting: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _save_plot(self):
        """
        Saves the current plot to a file.
        """
        # Check if matplotlib figure exists
        if not hasattr(self, 'fig'):
            logger.warning("Attempt to save plot before drawing.")
            return
        try:
            # File name selection dialog
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if filename:
                # Save figure
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved: {filename}")
                logger.info(f"Plot saved: {filename}")
        except Exception as e:
            error_msg = f"Error saving plot: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _export_to_csv(self):
        """
        Exports results to a CSV file with UTF-8 encoding.
        """
        # Check if there are solved problems
        export_default = self.solver is not None
        export_manual = self.manual_solver is not None

        if not export_default and not export_manual:
            warning_msg = "No solved problems to export."
            messagebox.showwarning("Warning", warning_msg)
            logger.warning(warning_msg)
            return

        # File name selection dialog
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filename:
            return  # User canceled

        try:
            # Open file for writing with UTF-8 encoding
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # Create writer with semicolon as delimiter
                writer = csv.writer(f, delimiter=';')
                # Header
                writer.writerow(["Lab Work #5: Simplex Method"])
                # Export built-in solution result
                if export_default:
                    writer.writerow(["Problem Type: Built-in Variant"])
                    optimal_plan = self.solver.get_optimal_plan()
                    optimal_value = self.solver.get_objective_value()
                    writer.writerow([f"Max. profit (built-in): {optimal_value:.2f}"])
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
                # Export manual solution result
                if export_manual:
                    writer.writerow(["Problem Type: Manual Input"])
                    opt_plan = self.manual_solver.get_optimal_plan()
                    opt_val = self.manual_solver.get_objective_value()
                    writer.writerow([f"Max. profit (manual): {opt_val:.2f}"])
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
            messagebox.showinfo("Success", f"Results exported to {filename}")
            logger.info(f"Results exported to CSV: {filename}")

        except Exception as e:
            error_msg = f"Error exporting to CSV: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _export_to_json(self):
        """
        Exports results to a JSON file with UTF-8 encoding.
        """
        # Check if there are solved problems
        export_default = self.solver is not None
        export_manual = self.manual_solver is not None

        if not export_default and not export_manual:
            warning_msg = "No solved problems to export."
            messagebox.showwarning("Warning", warning_msg)
            logger.warning(warning_msg)
            return

        # File name selection dialog
        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not filename:
            return  # User canceled

        try:
            # Prepare data for export
            export_data = {
                "metadata": {
                    "lab_number": 5,
                    "author": "Stanislav Kolosov",
                    "group": "IVT-3",
                    "year": 2026
                }
            }
            # Add built-in solution result
            if export_default:
                opt_plan = self.solver.get_optimal_plan()
                opt_val = self.solver.get_objective_value()
                export_data["solution_builtin"] = {
                    "optimal_value": opt_val,
                    "plan": {k: round(v, 2) for k, v in opt_plan.items()}  # Round for readability
                }

            # Add manual solution result
            if export_manual:
                opt_plan_m = self.manual_solver.get_optimal_plan()
                opt_val_m = self.manual_solver.get_objective_value()
                export_data["solution_manual"] = {
                    "optimal_value": opt_val_m,
                    "plan": {k: round(v, 2) for k, v in opt_plan_m.items()}
                }

            # Write data to file with UTF-8 encoding
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)  # ensure_ascii=False for Cyrillic
            messagebox.showinfo("Success", f"Results exported to {filename}")
            logger.info(f"Results exported to JSON: {filename}")

        except Exception as e:
            error_msg = f"Error exporting to JSON: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)

    def _clear_results(self):
        """
        Clears all results and resets solver states.
        """
        # Confirm action
        if not messagebox.askyesno("Confirmation", "Clear all results?"):
            return

        # Reset variables
        self.solver = None
        self.analyzer = None
        self.original_b = None
        self.manual_solver = None
        self.manual_analyzer = None
        self.manual_original_b = None

        # Clear widgets
        self.table_text.delete(1.0, END)
        self.report_text.delete(1.0, END)
        self.scenario_text.delete(1.0, END)
        for item in self.plan_tree.get_children():
            self.plan_tree.delete(item)
        self.profit_label.config(text="Maximum profit: —")
        self.status_label.config(text="Status (built-in): Results cleared", fg="#7f8c8d")
        self.progress_var.set(0)
        self.ax.clear()
        self.canvas_agg.draw()
        logger.info("Results cleared.")

    def _show_about(self):
        """
        Shows the "About" window.
        """
        about_text = """Lab Work #5 on the course "Fundamentals of Artificial Intelligence"
Topic: Solving a direct linear programming problem using the simplex method

Variant #1: Optimization of cargo loading for a cargo-passenger ship
Compartments: #1, #2, #3
Cargos: #1-5

Implementation: Stanislav Kolosov
Group: IVT-3
Year: 2026

Technologies: Python 3.13, tkinter, numpy, matplotlib

Features:
- Solving the built-in variant of the problem
- Manual input of problem parameters and solving
- Stability analysis, scenarios, plots, export for both modes"""
        messagebox.showinfo("About", about_text)

def main():
    """
    Application entry point.
    """
    root = Tk()
    app = SimplexApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
