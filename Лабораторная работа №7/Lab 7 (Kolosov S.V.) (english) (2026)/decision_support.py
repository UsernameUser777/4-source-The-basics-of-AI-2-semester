# -*- coding: utf-8 -*-

"""
Decision Support Module and Expert Evaluation Consistency Analysis.

Contains functions for working with pairwise comparison matrices and calculating alternative weights.
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2
import pandas as pd
import json
from datetime import datetime

def normalize_matrix(matrix):
    """
    Normalize the pairwise comparison matrix.

    Each element is divided by the sum of the elements in its column.

    Args:
        matrix (np.array): Original pairwise comparison matrix

    Returns:
        np.array: Normalized matrix
    """
    col_sums = matrix.sum(axis=0)
    return matrix / col_sums

def calculate_weights_eigenvector(matrix):
    """
    Calculate weights using the eigenvector method (Analytic Hierarchy Process).

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        tuple: (alternative weights, eigenvalue, consistency index CI, consistency ratio CR)
    """
    norm_matrix = normalize_matrix(matrix)
    weights = np.mean(norm_matrix, axis=1)
    Aw = np.dot(matrix, weights)
    lambda_max = np.mean(Aw / weights)
    n = matrix.shape[0]
    CI = (lambda_max - n) / (n - 1)
    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = random_index.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0
    return weights, lambda_max, CI, CR

def calculate_weights_log_least_squares(matrix):
    """
    Calculate weights using the logarithmic least squares method.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        np.array: Alternative weights
    """
    n = matrix.shape[0]
    log_matrix = np.log(matrix)
    row_sums = np.sum(log_matrix, axis=1)
    weights = np.exp(row_sums / n)
    return weights / np.sum(weights)

def calculate_weights_geometric_mean(matrix):
    """
    Calculate weights using the geometric mean method.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        np.array: Alternative weights
    """
    geometric_means = np.power(np.prod(matrix, axis=1), 1 / matrix.shape[1])
    return geometric_means / np.sum(geometric_means)

def calculate_weights_line_method(matrix):
    """
    Calculate weights using the "line" method.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        np.array: Alternative weights
    """
    n = matrix.shape[0]
    weights = np.ones(n)

    for i in range(n):
        product = 1.0
        for j in range(n):
            if i != j:
                product *= matrix[i, j]
        weights[i] = product ** (1 / n)

    return weights / np.sum(weights)

def calculate_weights_ahp(matrix):
    """
    Calculate weights using the Analytic Hierarchy Process (AHP) method with additional checks.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        tuple: (alternative weights, eigenvalue, consistency index CI, consistency ratio CR)
    """
    weights, lambda_max, CI, CR = calculate_weights_eigenvector(matrix)

    # Consistency check
    if CR > 0.1:
        print("Warning: Consistency ratio CR > 0.1. The matrix may be inconsistent.")

    return weights, lambda_max, CI, CR

def analyze_consistency(matrix):
    """
    Full consistency analysis of the pairwise comparison matrix.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        dict: Consistency analysis results
    """
    try:
        weights_eig, lambda_max, CI, CR = calculate_weights_eigenvector(matrix)
    except Exception:
        weights_eig, lambda_max, CI, CR = np.zeros(matrix.shape[0]), 0, 0, 0

    try:
        weights_log = calculate_weights_log_least_squares(matrix)
    except Exception:
        weights_log = np.zeros(matrix.shape[0])

    try:
        weights_geo = calculate_weights_geometric_mean(matrix)
    except Exception:
        weights_geo = np.zeros(matrix.shape[0])

    try:
        weights_line = calculate_weights_line_method(matrix)
    except Exception:
        weights_line = np.zeros(matrix.shape[0])

    try:
        corr_matrix = np.corrcoef(matrix)
        mean_corr = np.mean(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
    except Exception:
        corr_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
        mean_corr = 0.0

    try:
        expected = np.sqrt(matrix * matrix.T)
        chi2_stat = 0
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                if expected[i, j] > 0:
                    chi2_stat += ((matrix[i, j] - expected[i, j]) ** 2) / expected[i, j]
                    chi2_stat += ((matrix[j, i] - expected[j, i]) ** 2) / expected[j, i]
        df = (matrix.shape[0] - 1) * (matrix.shape[0] - 2) // 2
        p_value = 1 - chi2.cdf(chi2_stat, df) if df > 0 else 1.0
    except Exception:
        chi2_stat, p_value, df = 0.0, 1.0, 0

    try:
        inconsistent_pairs = []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                for k in range(matrix.shape[0]):
                    if k != i and k != j:
                        expected = matrix[i, j] * matrix[j, k]
                        actual = matrix[i, k]
                        if actual > 0:
                            inconsistency = abs(expected - actual) / actual
                            inconsistent_pairs.append((i, j, k, inconsistency))

        inconsistent_pairs.sort(key=lambda x: x[3], reverse=True)
    except Exception:
        inconsistent_pairs = []

    return {
        'weights_eigenvector': weights_eig,
        'weights_log_least_squares': weights_log,
        'weights_geometric_mean': weights_geo,
        'weights_line_method': weights_line,
        'lambda_max': lambda_max,
        'CI': CI,
        'CR': CR,
        'correlation_matrix': corr_matrix,
        'mean_correlation': mean_corr,
        'chi2_statistic': chi2_stat,
        'chi2_p_value': p_value,
        'chi2_df': df,
        'inconsistent_pairs': inconsistent_pairs[:5],
        'consistent_matrix': generate_consistent_matrix(weights_eig)
    }

def generate_consistent_matrix(weights):
    """
    Generate a consistent pairwise comparison matrix based on weights.

    Args:
        weights (list or np.array): Alternative weights

    Returns:
        np.array: Consistent pairwise comparison matrix
    """
    n = len(weights)
    consistent_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                consistent_matrix[i, j] = weights[i] / weights[j]

    return consistent_matrix

def check_transitivity(matrix):
    """
    Check the transitivity of the pairwise comparison matrix.

    Args:
        matrix (np.array): Pairwise comparison matrix

    Returns:
        tuple: (transitivity flag, list of inconsistent triplets)
    """
    n = matrix.shape[0]
    transitivity_violations = []

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if matrix[i, j] * matrix[j, k] != matrix[i, k]:
                    transitivity_violations.append((i, j, k))

    return len(transitivity_violations) == 0, transitivity_violations

def create_sensitivity_analysis(matrix, results, perturbation_factor=0.1):
    """
    Sensitivity analysis to changes in the matrix.

    Args:
        matrix (np.array): Original matrix
        results (dict): Analysis results
        perturbation_factor (float): Perturbation factor

    Returns:
        dict: Sensitivity analysis results
    """
    n = matrix.shape[0]
    sensitivity_results = {}
    try:
        original_weights = results['weights_eigenvector'].copy()
        max_sensitivity = 0
        most_sensitive_pair = (0, 1)

        for i in range(n):
            for j in range(i + 1, n):
                modified_matrix = matrix.copy()
                modified_matrix[i, j] *= (1 + perturbation_factor)
                modified_matrix[j, i] = 1 / modified_matrix[i, j]
                try:
                    modified_results = analyze_consistency(modified_matrix)
                    modified_weights = modified_results['weights_eigenvector']
                    sensitivity = np.linalg.norm(modified_weights - original_weights)

                    if sensitivity > max_sensitivity:
                        max_sensitivity = sensitivity
                        most_sensitive_pair = (i, j)
                except Exception:
                    continue

        sensitivity_results['max_sensitivity'] = max_sensitivity
        sensitivity_results['most_sensitive_pair'] = most_sensitive_pair
        sensitivity_results['perturbation_factor'] = perturbation_factor
    except Exception:
        sensitivity_results['max_sensitivity'] = 0
        sensitivity_results['most_sensitive_pair'] = (0, 1)
        sensitivity_results['perturbation_factor'] = perturbation_factor

    return sensitivity_results

def adjust_inconsistent_pairs(matrix, inconsistent_pairs, threshold=0.1):
    """
    Automatic correction of inconsistent pairwise comparisons.

    Args:
        matrix (np.array): Original matrix
        inconsistent_pairs (list): List of inconsistent pairs
        threshold (float): Inconsistency threshold

    Returns:
        tuple: (corrected matrix, list of changed pairs)
    """
    corrected_matrix = matrix.copy()
    changes_made = []

    for a, b, c, inc in inconsistent_pairs:
        if inc > threshold:
            recommended = matrix[a, b] * matrix[b, c]
            current = matrix[a, c]
            correction_factor = 0.5
            new_value = current + correction_factor * (recommended - current)

            if new_value > 0:
                corrected_matrix[a, c] = new_value
                corrected_matrix[c, a] = 1 / new_value
                changes_made.append({
                    'pair': (a, c),
                    'original': current,
                    'corrected': new_value,
                    'recommendation': recommended,
                    'inconsistency': inc
                })

    return corrected_matrix, changes_made

def export_to_latex_table(df, caption="Table", label="tab:example"):
    """
    Export DataFrame to LaTeX table.

    Args:
        df (pd.DataFrame): DataFrame to export
        caption (str): Table caption
        label (str): Table label

    Returns:
        str: LaTeX table code
    """
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += f"\\caption{{{caption}}}\n"
    latex_code += f"\\label{{{label}}}\n"
    latex_code += "\\begin{tabular}{" + "|c" * len(df.columns) + "|}\n"
    latex_code += "\\hline\n"
    headers = " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\\hline\n"
    latex_code += headers

    for _, row in df.iterrows():
        row_data = " & ".join([f"{val}" for val in row.values]) + " \\\\\hline\n"
        latex_code += row_data

    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{table}\n"

    return latex_code

def save_session(session_data, filename):
    """
    Save analysis session to file.

    Args:
        session_data (dict): Session data
        filename (str): File name for saving

    Returns:
        bool: Success of saving
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False)

        return True
    except Exception as e:
        print(f"Error saving session: {e}")
        return False

def load_session(filename):
    """
    Load analysis session from file.

    Args:
        filename (str): File name for loading

    Returns:
        dict: Session data or None if error
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        return session_data
    except Exception as e:
        print(f"Error loading session: {e}")
        return None
