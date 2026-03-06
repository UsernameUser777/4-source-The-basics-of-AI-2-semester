# -*- coding: utf-8 -*-
"""GUI module for the Hierarchy Analysis application."""

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

from mcdm_methods import MatrixProcessor  # Using the updated module

class MCDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laboratory Work #8 - Hierarchy Analysis")
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
        self.add_to_history("Initialization")

    def setup_ui(self):
        # Menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.load_matrix)
        file_menu.add_command(label="Save", command=self.save_matrix)
        file_menu.add_separator()
        file_menu.add_command(label="Export to CSV", command=self.export_to_csv)
        file_menu.add_command(label="Export to JSON", command=self.export_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Dark Theme", command=lambda: self.toggle_theme(dark=True))
        view_menu.add_command(label="Light Theme", command=lambda: self.toggle_theme(dark=False))
        view_menu.add_command(label="Presentation Mode", command=self.start_presentation_mode)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="Methodical Examples", command=self.show_methodical_examples)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Data Input Tab ---
        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Data Input")
        self.create_input_tab()

        # --- Results Tab ---
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.create_results_tab()

        # --- Rank Reversal Tab ---
        self.reversal_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reversal_frame, text="Rank Reversal")
        self.create_reversal_tab()

        # --- Sensitivity Tab ---
        self.sensitivity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sensitivity_frame, text="Sensitivity")
        self.create_sensitivity_tab()

        # --- Stability Tab ---
        self.stability_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stability_frame, text="Stability")
        self.create_stability_tab()

        # --- Consistency Tab ---
        self.consistency_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.consistency_frame, text="Consistency")
        self.create_consistency_tab()

        # --- Visualization Tab ---
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Visualization")
        self.create_visualization_tab()

        # --- Report Tab ---
        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="Report")
        self.create_report_tab()

        # --- Help Tab ---
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="Help")
        self.create_help_tab()

    def create_input_tab(self):
        # History
        history_frame = ttk.LabelFrame(self.input_frame, text="Change History")
        history_frame.pack(fill=tk.X, padx=10, pady=5)
        self.history_text = tk.Text(history_frame, height=4, state='disabled')
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        hist_btn_frame = ttk.Frame(history_frame)
        hist_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(hist_btn_frame, text="<< Back", command=self.history_back).pack(side=tk.LEFT)
        ttk.Button(hist_btn_frame, text="Forward >>", command=self.history_forward).pack(side=tk.LEFT, padx=(5, 0))

        # Matrix
        matrix_frame = ttk.LabelFrame(self.input_frame, text="Pairwise Comparison Matrix (4x4)")
        matrix_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Column headers
        for j in range(4):
            ttk.Label(matrix_frame, text=f"A{j+1}").grid(row=0, column=j + 1, padx=5, pady=5)

        # Row headers and cells
        for i in range(4):
            ttk.Label(matrix_frame, text=f"A{i+1}").grid(row=i + 1, column=0, padx=5, pady=5)
            for j in range(4):
                var = tk.StringVar(value=str(self.current_matrix[i, j]))
                entry = ttk.Entry(matrix_frame, textvariable=var, width=8)
                entry.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                self.matrix_entries[i][j] = var
                var.trace_add("write", lambda *args, i=i, j=j: self.on_matrix_entry_change(i, j))

        # Control buttons
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Update", command=self.update_from_entries).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Check Symmetry", command=self.check_symmetry).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Compare with Ideal", command=self.compare_with_ideal).pack(side=tk.LEFT, padx=5)

    def create_results_tab(self):
        # Text field for results
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Calculation buttons
        btn_frame = ttk.Frame(self.results_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Calculate Weights", command=self.calculate_and_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Compare Methods", command=self.compare_methods).pack(side=tk.LEFT, padx=5)

    def create_reversal_tab(self):
        # Setup for adding a new alternative
        setup_frame = ttk.LabelFrame(self.reversal_frame, text="Add New Alternative")
        setup_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(setup_frame, text="Enter comparison weights of the new alternative with existing ones (A1-A4):").grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.new_alt_entries = []
        for i in range(4):
            ttk.Label(setup_frame, text=f"Comparison with A{i+1}:").grid(row=i+1, column=0, padx=5, pady=3, sticky=tk.W)
            var = tk.StringVar(value="1.0")
            entry = ttk.Entry(setup_frame, textvariable=var, width=8)
            entry.grid(row=i+1, column=1, padx=5, pady=3)
            self.new_alt_entries.append(var)

        ttk.Button(setup_frame, text="Add Alternative and Analyze Reversal",
                   command=self.add_alternative_and_analyze).grid(row=5, column=0, columnspan=2, pady=10)

        # Results field
        results_frame = ttk.LabelFrame(self.reversal_frame, text="Rank Reversal Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.reversal_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.reversal_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_sensitivity_tab(self):
        # Sensitivity analysis description
        desc_frame = ttk.LabelFrame(self.sensitivity_frame, text="Description")
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        desc_text = ("Sensitivity analysis shows how alternative weights change "
                     "with small changes in the pairwise comparison matrix.")
        ttk.Label(desc_frame, text=desc_text, wraplength=1100, justify=tk.LEFT).pack(padx=10, pady=5, anchor=tk.W)

        # Button to run analysis
        ttk.Button(self.sensitivity_frame, text="Perform Sensitivity Analysis",
                   command=self.analyze_sensitivity).pack(pady=10)

        # Results field
        results_frame = ttk.LabelFrame(self.sensitivity_frame, text="Sensitivity Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.sensitivity_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.sensitivity_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_stability_tab(self):
        # Stability analysis description
        desc_frame = ttk.LabelFrame(self.stability_frame, text="Description")
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        desc_text = ("Stability analysis evaluates the robustness of alternative rankings "
                     "to small changes in the input data.\n"
                     "Wider intervals indicate a more stable solution.")
        ttk.Label(desc_frame, text=desc_text, wraplength=1100, justify=tk.LEFT).pack(padx=10, pady=5, anchor=tk.W)

        # Button to run analysis
        ttk.Button(self.stability_frame, text="Perform Stability Analysis", command=self.analyze_stability).pack(pady=10)

        # Results field
        results_frame = ttk.LabelFrame(self.stability_frame, text="Stability Intervals")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stability_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.stability_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_consistency_tab(self):
        # Consistency metrics
        info_frame = ttk.LabelFrame(self.consistency_frame, text="Consistency Metrics")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        self.cr_label = ttk.Label(info_frame, text="", font=("Arial", 10, "bold"))
        self.cr_label.pack(pady=5)
        self.ci_label = ttk.Label(info_frame, text="")
        self.ci_label.pack(pady=5)
        self.ri_label = ttk.Label(info_frame, text="")
        self.ri_label.pack(pady=5)
        self.rec_label = ttk.Label(info_frame, text="", wraplength=1100, justify=tk.LEFT)
        self.rec_label.pack(pady=10)

        # Actions to improve consistency
        action_frame = ttk.LabelFrame(self.consistency_frame, text="Actions to Improve Consistency")
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(action_frame, text="Check Transitivity", command=self.check_transitivity).pack(pady=5)
        ttk.Button(action_frame, text="Show Inconsistent Pairs", command=self.show_inconsistent_pairs).pack(pady=5)

    def create_visualization_tab(self):
        # Visualization controls
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(control_frame, text="Update All Plots", command=self.update_all_visualizations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Plot as SVG", command=self.save_current_plot_svg).pack(side=tk.LEFT, padx=5)

        # Plots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_report_tab(self):
        # Report controls
        controls_frame = ttk.Frame(self.report_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(controls_frame, text="Generate Report (PDF)", command=self.generate_pdf_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Generate Report (LaTeX)", command=self.generate_latex_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Generate Report (HTML)", command=self.generate_html_report).pack(side=tk.LEFT, padx=5)

        # Report display field
        self.report_text = scrolledtext.ScrolledText(self.report_frame, wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_help_tab(self):
        # User guide
        text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.show_step_by_step_guide(text_widget=text)

    # --- Event Handling Methods ---
    def on_matrix_entry_change(self, i, j):
        """Update matrix when a cell is changed."""
        try:
            val = float(self.matrix_entries[i][j].get())
            self.current_matrix[i, j] = val
            if i != j:  # Update the symmetric element
                self.matrix_entries[j][i].set(str(1.0 / val))
                self.current_matrix[j, i] = 1.0 / val
        except ValueError:
            pass  # Ignore invalid input

    def update_from_entries(self):
        """Update matrix from input fields."""
        try:
            for i in range(4):
                for j in range(4):
                    self.current_matrix[i, j] = float(self.matrix_entries[i][j].get())
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_consistency_info()
            self.add_to_history("Matrix Updated")
            messagebox.showinfo("Success", "Matrix updated.")
        except ValueError:
            messagebox.showerror("Error", "Invalid data in cells.")

    def reset_matrix(self):
        """Reset matrix to default state."""
        self.current_matrix = np.array([
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 5.0, 3.0],
            [1 / 3, 1 / 5, 1.0, 1 / 5],
            [1.0, 1 / 3, 5.0, 1.0]
        ])
        self.matrix_processor = MatrixProcessor(self.current_matrix)
        self.update_matrix_entries()
        self.update_consistency_info()
        self.add_to_history("Matrix Reset")

    def update_matrix_entries(self):
        """Update input fields from current matrix."""
        for i in range(4):
            for j in range(4):
                self.matrix_entries[i][j].set(f"{self.current_matrix[i, j]:.4f}")

    def check_symmetry(self):
        """Check for reciprocal symmetry in the matrix."""
        is_symmetric = True
        violations = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    if abs(self.current_matrix[i, j] * self.current_matrix[j, i] - 1.0) > 1e-9:
                        is_symmetric = False
                        violations.append((i + 1, j + 1))
        if is_symmetric:
            messagebox.showinfo("Symmetry", "Matrix has reciprocal symmetry!")
        else:
            messagebox.showwarning("Warning", f"Reciprocal symmetry violations found: {violations}")

    def compare_with_ideal(self):
        """Compare current matrix with the ideal matrix."""
        ideal_matrix = np.array([
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 5.0, 3.0],
            [1 / 3, 1 / 5, 1.0, 1 / 5],
            [1.0, 1 / 3, 5.0, 1.0]
        ])
        current_matrix = self.current_matrix

        compare_window = tk.Toplevel(self.root)
        compare_window.title("Comparison with Ideal Matrix")
        compare_window.geometry("800x600")

        frame_current = ttk.LabelFrame(compare_window, text="Current Matrix")
        frame_current.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_ideal = ttk.LabelFrame(compare_window, text="Ideal Matrix")
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

    # --- History Methods ---
    def add_to_history(self, action):
        """Add action to history."""
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
        """Update history display."""
        self.history_text.config(state='normal')
        self.history_text.delete(1.0, tk.END)
        for idx, entry in enumerate(self.history):
            marker = ">>>" if idx == self.history_position else ""
            self.history_text.insert(tk.END, f"{marker}[{entry['timestamp']}] {entry['action']}\n")
        self.history_text.config(state='disabled')

    def history_back(self):
        """Go back in history."""
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
        """Go forward in history."""
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self.current_matrix = self.history[self.history_position]['matrix'].copy()
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_matrix_entries()
            self.calculate_and_display()
            self.update_all_visualizations()
            self.update_history_display()

    # --- Update Information Methods ---
    def update_consistency_info(self):
        """Update consistency information."""
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        self.cr_label.config(text=f"Consistency Ratio (CR): {cr:.4f}")
        self.ci_label.config(text=f"Consistency Index (CI): {ci:.4f}")
        self.ri_label.config(text=f"Random Index (RI): {ri:.4f}")
        if cr < 0.1:
            rec_text = "Matrix is consistent. Results can be considered reliable."
            self.rec_label.config(text=rec_text, foreground="green")
        else:
            rec_text = ("Matrix is inconsistent (CR > 0.1). It is recommended to revise "
                        "pairwise comparisons to improve consistency.")
            self.rec_label.config(text=rec_text, foreground="red")

    def calculate_and_display(self):
        """Calculate and display weights and ranks."""
        methods, weights = self.matrix_processor.get_weights_comparison()
        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Alternative {i + 1}" for i in range(4)]
        )
        ranks = np.argsort(-weights, axis=1) + 1
        df_ranks = pd.DataFrame(
            ranks,
            index=methods,
            columns=[f"Alternative {i + 1}" for i in range(4)]
        )
        result_str = "=== Weight Calculation Results ===\n\n"
        result_str += str(df_weights.round(4))
        result_str += "\n\n=== Alternative Ranks ===\n\n"
        result_str += str(df_ranks)

        # Kendall coefficients
        tau_df, _, _, _, _ = self.matrix_processor.calculate_kendall_tau()
        result_str += "\n\n=== Kendall Coefficients Between Methods ===\n\n"
        result_str += str(tau_df.round(4))

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state='disabled')

    def compare_methods(self):
        """Compare weight calculation methods."""
        methods, weights = self.matrix_processor.get_weights_comparison()
        tau_df, tau_matrix, _, _, _ = self.matrix_processor.calculate_kendall_tau()

        result_str = "=== Method Comparison ===\n\n"
        for i, method in enumerate(methods):
            result_str += f"{method}: {weights[i].round(4)}\n"
        result_str += "\n=== Kendall Coefficients ===\n\n"
        result_str += str(tau_df.round(4))

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state='disabled')

    # --- Analysis Methods ---
    def add_alternative_and_analyze(self):
        """Add a new alternative and analyze rank reversal."""
        try:
            new_row = [float(v.get()) for v in self.new_alt_entries]
            if len(new_row) != 4:
                raise ValueError("Four values are required.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid data for new alternative: {e}")
            return

        result = self.matrix_processor.add_alternative_and_analyze(new_row + [1.0])
        self.reversal_results = result

        report_str = "=== Results of Adding a New Alternative ===\n\n"
        report_str += "New Matrix (5x5):\n"
        report_str += str(result["new_matrix"].round(4)) + "\n\n"
        report_str += "Weights by Methods (new):\n"
        for method, w in result["new_weights"].items():
            report_str += f"{method}: {np.array(w).round(4)} \n"
        report_str += "\nRanks by Methods (new):\n"
        for i, method in enumerate(["Distributive", "Ideal", "Multiplicative", "GUBOPA", "MAI"]):
            report_str += f"{method}: {result['new_ranks'][i]} \n"

        report_str += "\n=== Rank Reversal Analysis ===\n"
        for res in result["reversal_results"]:
            report_str += f"\nMethod: {res['method']} \n"
            report_str += f"Original Ranks (old): {res['original_ranks']} \n"
            report_str += f"New Ranks (old alternatives): {res['new_ranks']} \n"
            if res["reversal_detected"]:
                report_str += f"Reversal detected! Pairs: {res['reversal_pairs']} \n"
            else:
                report_str += "No reversal detected.\n"

        self.reversal_results_text.config(state='normal')
        self.reversal_results_text.delete(1.0, tk.END)
        self.reversal_results_text.insert(tk.END, report_str)
        self.reversal_results_text.config(state='disabled')

    def analyze_sensitivity(self):
        """Analyze sensitivity of weights to changes in matrix elements."""
        baseline_weights = self.matrix_processor.get_weights_comparison()[1]
        changes = np.linspace(-0.2, 0.2, 5)
        element_to_vary = (0, 1)  # a12
        results = []
        original_val = self.current_matrix[element_to_vary]

        for delta in changes:
            temp_matrix = self.current_matrix.copy()
            new_val = max(0.1, original_val + delta)  # Limit range
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

        report_str = "=== Sensitivity Analysis Results ===\n\n"
        for r in results:
            report_str += (f"Value a{element_to_vary[0] + 1},{element_to_vary[1] + 1}: {r['element_value']:.2f}, "
                           f"Average Weight Change: {r['avg_weight_change']:.4f}\n")

        self.sensitivity_results_text.config(state='normal')
        self.sensitivity_results_text.delete(1.0, tk.END)
        self.sensitivity_results_text.insert(tk.END, report_str)
        self.sensitivity_results_text.config(state='disabled')

    def analyze_stability(self):
        """Analyze stability of ranks when noise is added to the matrix."""
        baseline_weights = self.matrix_processor.get_weights_comparison()[1]
        noise_levels = np.linspace(0.0, 0.1, 6)
        results = []

        for level in noise_levels:
            rank_changes = []
            for _ in range(10):  # 10 repetitions for each noise level
                noise = np.random.uniform(-level, level, size=self.current_matrix.shape)
                perturbed_matrix = self.current_matrix + noise
                perturbed_matrix = np.abs(perturbed_matrix)  # Ensure positivity

                for i in range(4):
                    for j in range(4):
                        if i != j:
                            perturbed_matrix[i, j] = max(0.1, perturbed_matrix[i, j])
                            perturbed_matrix[j, i] = 1.0 / perturbed_matrix[i, j]

                test_proc = MatrixProcessor(perturbed_matrix)
                _, weights = test_proc.get_weights_comparison()
                ranks = np.argsort(-weights, axis=1) + 1
                rank_changes.append(ranks[0])  # Use the first method

            results.append({
                "noise_level": level,
                "avg_std": np.mean(np.std(rank_changes, axis=0))
            })

        report_str = "=== Stability Analysis Results ===\n\n"
        for r in results:
            report_str += (f"Noise Level: {r['noise_level']:.2f}, "
                           f"Average Standard Deviation of Ranks: {r['avg_std']:.4f}\n")

        self.stability_results_text.config(state='normal')
        self.stability_results_text.delete(1.0, tk.END)
        self.stability_results_text.insert(tk.END, report_str)
        self.stability_results_text.config(state='disabled')

    def check_transitivity(self):
        """Check matrix transitivity."""
        inconsistencies = self.matrix_processor.check_transitivity()
        if inconsistencies:
            msg = f"Transitivity violations found: {inconsistencies}"
        else:
            msg = "No transitivity violations found."
        messagebox.showinfo("Transitivity Check", msg)

    def show_inconsistent_pairs(self):
        """Display inconsistent pairs."""
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        if cr >= 0.1:
            inconsistent_pairs = self.matrix_processor.find_inconsistent_pairs()
            msg = (f"Matrix is inconsistent (CR={cr:.4f}).\n"
                   f"Most inconsistent pairs:\n{inconsistent_pairs}")
        else:
            msg = "Matrix is consistent. No inconsistent pairs found."
        messagebox.showinfo("Inconsistent Pairs", msg)

    # --- Visualization Methods ---
    def update_all_visualizations(self):
        """Update all visualizations."""
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()
        self.axs[1, 0].clear()
        self.axs[1, 1].clear()

        # 1. Weights plot
        methods, weights = self.matrix_processor.get_weights_comparison()
        x = np.arange(len(methods))
        width = 0.15
        for i in range(4):
            self.axs[0, 0].bar(x + i * width, weights[:, i], width, label=f'A{i + 1}')
        self.axs[0, 0].set_xlabel('Method')
        self.axs[0, 0].set_ylabel('Weight')
        self.axs[0, 0].set_title('Weight Comparison by Methods')
        self.axs[0, 0].set_xticks(x + width * 1.5)
        self.axs[0, 0].set_xticklabels(methods, rotation=45)
        self.axs[0, 0].legend()

        # 2. Heatmap of the matrix
        im = self.axs[0, 1].imshow(self.current_matrix, cmap='viridis', aspect='auto')
        self.axs[0, 1].set_title('Matrix Heatmap')
        self.axs[0, 1].set_xticks(range(4))
        self.axs[0, 1].set_yticks(range(4))
        self.axs[0, 1].set_xticklabels([f'A{i + 1}' for i in range(4)])
        self.axs[0, 1].set_yticklabels([f'A{i + 1}' for i in range(4)])
        self.fig.colorbar(im, ax=self.axs[0, 1])

        # 3. Ranks
        ranks = np.argsort(-weights, axis=1) + 1
        for i in range(4):
            self.axs[1, 0].plot(methods, ranks[:, i], marker='o', label=f'A{i + 1}')
        self.axs[1, 0].set_xlabel('Method')
        self.axs[1, 0].set_ylabel('Rank')
        self.axs[1, 0].set_title('Alternative Ranks by Methods')
        self.axs[1, 0].legend()
        self.axs[1, 0].tick_params(axis='x', rotation=45)

        # 4. CR histogram
        cr, _, _ = self.matrix_processor.calculate_consistency()
        self.axs[1, 1].bar(['CR'], [cr], color=['green'] if cr < 0.1 else 'red')
        self.axs[1, 1].axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
        self.axs[1, 1].set_ylabel('Value')
        self.axs[1, 1].set_title('Consistency Ratio (CR)')
        self.axs[1, 1].legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def save_current_plot_svg(self):
        """Save current plot as SVG."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            self.fig.savefig(filename, format='svg')

    # --- Report Methods ---
    def generate_pdf_report(self):
        """Generate report in PDF format."""
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
        story.append(Paragraph("Hierarchy Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Input matrix
        story.append(Paragraph("1. Input Matrix", styles['Heading2']))
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

        # Weight calculation results
        story.append(Paragraph("2. Weight Calculation Results", styles['Heading2']))
        methods, weights = self.matrix_processor.get_weights_comparison()
        df_weights = pd.DataFrame(
            weights,
            index=methods,
            columns=[f"Alternative {i+1}" for i in range(4)]
        )
        data_w = [df_weights.columns.tolist()]
        data_w += df_weights.round(4).values.tolist()
        data_w[0] = ["Method"] + data_w[0]
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

        # Consistency
        story.append(Paragraph("3. Consistency Check", styles['Heading2']))
        cr, ci, ri = self.matrix_processor.calculate_consistency()
        cons_text = (f"Consistency Ratio (CR): {cr:.4f}<br/>"
                     f"Consistency Index (CI): {ci:.4f}<br/>"
                     f"Random Index (RI): {ri:.4f}")
        story.append(Paragraph(cons_text, styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        messagebox.showinfo("Report", f"PDF report successfully saved to {filename}")

    def generate_latex_report(self):
        """Generate report in LaTeX format."""
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
            f.write("\\title{Hierarchy Analysis Report}\n")
            f.write("\\author{Kolosov S.V.}\n")
            f.write("\\date{\\today}\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\\n")
            f.write("\\section{Input Matrix}\n")
            f.write("\\begin{longtable}{|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("& Alternative 1 & Alternative 2 & Alternative 3 & Alternative 4 \\\\\n")
            f.write("\\hline\n")
            for i, row in enumerate(self.current_matrix):
                f.write(f"Alternative {i + 1} & ")
                f.write(" & ".join([f"{val:.4f}" for val in row]))
                f.write("\\\\\n")
                f.write("\\hline\n")
            f.write("\\end{longtable}\n")
            f.write("\\n")
            f.write("\\section{Weight Calculation Results}\n")
            methods, weights = self.matrix_processor.get_weights_comparison()
            f.write("\\begin{longtable}{|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Method & Alternative 1 & Alternative 2 & Alternative 3 & Alternative 4 \\\\\n")
            f.write("\\hline\n")
            for i, method in enumerate(methods):
                f.write(f"{method} & ")
                f.write(" & ".join([f"{w:.4f}" for w in weights[i]]))
                f.write("\\\\\n")
                f.write("\\hline\n")
            f.write("\\end{longtable}\n")
            f.write("\\n")
            f.write("\\section{Consistency Check}\n")
            cr, ci, ri = self.matrix_processor.calculate_consistency()
            f.write(f"Consistency Ratio (CR): {cr:.4f} \\\\\n")
            f.write(f"Consistency Index (CI): {ci:.4f} \\\\\n")
            f.write(f"Random Index (RI): {ri:.4f} \\\\\n")
            f.write("\\end{document}\n")

        messagebox.showinfo("Report", f"LaTeX report successfully saved to {filename}")

    def generate_html_report(self):
        """Generate report in HTML format."""
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
<title>Hierarchy Analysis Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h1>Hierarchy Analysis Report</h1>
<h2>Input Matrix</h2>
<table>
<tr><th></th><th>Alternative 1</th><th>Alternative 2</th><th>Alternative 3</th><th>Alternative 4</th></tr>
"""
        for i, row in enumerate(self.current_matrix):
            html_content += f"<tr><td>Alternative {i + 1}</td>"
            for val in row:
                html_content += f"<td>{val:.4f}</td>"
            html_content += "</tr>\n"
        html_content += """
</table>
<h2>Consistency Check</h2>
<p>Consistency Ratio (CR): {cr:.4f}</p>
<p>Consistency Index (CI): {ci:.4f}</p>
<p>Random Index (RI): {ri:.4f}</p>
</body>
</html>
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        messagebox.showinfo("Report", f"HTML report successfully saved to {filename}")

    # --- Help Methods ---
    def show_step_by_step_guide(self, text_widget=None):
        """Display step-by-step guide."""
        guide_content = """LABORATORY WORK #8.
MULTI-CRITERIA HIERARCHY ANALYSIS

Step 1: Data Input
- Go to the "Data Input" tab.
- Enter values into the pairwise comparison matrix (4x4).
- Click "Update".

Step 2: Consistency Check
- Go to the "Consistency" tab.
- Check the CR value (should be < 0.1).
- Adjust the matrix if necessary.

Step 3: Weight Calculation
- Go to the "Results" tab.
- Click "Calculate Weights".
- Analyze the obtained weights and ranks by methods.

Step 4: Rank Reversal Analysis
- Go to the "Rank Reversal" tab.
- Enter weights for the new alternative.
- Click "Add Alternative and Analyze Reversal".
- Analyze if the rank of the first alternative has changed.

Step 5: Generate Report
- Go to the "Report" tab.
- Click "Generate Report".
- Save the results.
"""
        if text_widget:
            text_widget.config(state='normal')
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, guide_content)
            text_widget.config(state='disabled')
        else:
            guide_window = tk.Toplevel(self.root)
            guide_window.title("Step-by-Step Guide")
            guide_window.geometry("700x500")
            text = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, guide_content)
            text.config(state='disabled')

    def show_methodical_examples(self):
        """Display examples from the manual."""
        examples_window = tk.Toplevel(self.root)
        examples_window.title("Methodical Examples")
        examples_window.geometry("800x600")
        text = scrolledtext.ScrolledText(examples_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        examples_content = """EXAMPLES FROM METHODICAL GUIDELINES
The Analytic Hierarchy Process (AHP) involves building a hierarchy of goals, criteria, and alternatives.
For pairwise comparisons, the Saaty scale (1, 3, 5, 7, 9) is used.
The matrix must be consistent (CR < 0.1).
Weights are calculated, for example, as normalized row sums (distributive method) or via the eigenvector (MAI).

Example of filling the matrix:
If alternative A1 is "preferred" over A2, then a12 = 3, a21 = 1/3.
"""
        text.insert(tk.END, examples_content)
        text.config(state='disabled')

    def show_help(self):
        """Display help."""
        self.show_step_by_step_guide(text_widget=None)

    def show_about(self):
        """Display about information."""
        about_text = """Laboratory Work #8 on Fundamentals of AI
Variant 1

Author: Kolosov S.V.
Group: IVT-3
Course: 4

(c) 2026"""
        messagebox.showinfo("About", about_text)

    # --- Theme Methods ---
    def toggle_theme(self, dark=True):
        """Toggle dark/light theme."""
        if dark:
            self.style.theme_use('clam')
            self.root.configure(bg='#2b2b2b')
            self.style.configure('TFrame', background='#3c3f41')
            self.style.configure('TLabel', background='#3c3f41', foreground='white')
            self.style.configure('TButton', background='#4a4d52', foreground='white')
            self.style.map('TButton', background=[('active', '#5a5d62')])
            messagebox.showinfo("Theme", "Dark theme activated")
        else:
            self.style.theme_use('default')
            self.root.configure(bg='SystemButtonFace')
            self.style.configure('TFrame', background="")
            self.style.configure('TLabel', background="", foreground='black')
            self.style.configure('TButton', background="", foreground='black')
            messagebox.showinfo("Theme", "Light theme activated")

    def start_presentation_mode(self):
        """Start presentation mode."""
        pres_window = tk.Toplevel(self.root)
        pres_window.attributes('-fullscreen', True)
        pres_window.configure(bg='black')
        label = tk.Label(pres_window, text="PRESENTATION MODE\nPress ESC to exit",
                         fg='white', bg='black', font=("Arial", 24))
        label.pack(expand=True)
        pres_window.bind('<Escape>', lambda e: pres_window.destroy())

    # --- Save/Load Methods ---
    def load_matrix(self):
        """Load matrix from file."""
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
                messagebox.showerror("Error", "Unsupported file format.")
                return

            if matrix.shape != (4, 4):
                raise ValueError("Matrix must be 4x4.")

            self.current_matrix = matrix
            self.matrix_processor = MatrixProcessor(self.current_matrix)
            self.update_matrix_entries()
            self.update_consistency_info()
            self.add_to_history("Matrix Loaded")
            messagebox.showinfo("Success", f"Matrix loaded from {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def save_matrix(self):
        """Save matrix to file."""
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
                messagebox.showerror("Error", "Unsupported file format.")
                return

            messagebox.showinfo("Success", f"Matrix saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def export_to_csv(self):
        """Export data to CSV."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            df = pd.DataFrame(self.current_matrix)
            df.to_csv(filename, index=False)
            messagebox.showinfo("Export", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def export_to_json(self):
        """Export data to JSON."""
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
            messagebox.showinfo("Export", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

def main():
    root = tk.Tk()
    app = MCDAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
