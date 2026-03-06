# Author: Kolosov S.V., IVT-3, 4th year
# Lab work №6, variant №1, 2026
# Full visualization implementation with error handling and compliance with requirements

import matplotlib
matplotlib.use('TkAgg')  # Important for working with Tkinter in main.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, List, Dict, Any, Set
import logging

# Import logger
try:
    from utils.logger import logger
except ImportError:
    logger = logging.getLogger("Visualization")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

# Set default style (dark mode)
sns.set_style("darkgrid")

def safe_save_fig(fig: Optional[plt.Figure], filepath: str) -> bool:
    """
    Safely save a figure — check if fig is not None.
    """
    if fig is None:
        logger.error(f"Failed to save figure: fig == None (path: {filepath})")
        return False
    try:
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving figure {filepath}: {e}")
        return False

def plot_decision_tree(
    options: List[Dict[str, Any]],
    utility_type: str = "exponential",
    risk_params: Optional[Dict[str, float]] = None,
    title: str = "Decision Tree"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Plot a decision tree with decision nodes and outcomes.
    """
    if not options:
        logger.error("plot_decision_tree: options list is empty")
        return None, None

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        y_start = len(options) * 1.0
        y_step = 1.2
        risk_params = risk_params or {}
        colors = sns.color_palette("husl", len(options))

        for i, option in enumerate(options):
            y = y_start - i * y_step
            outcomes = option.get("outcomes", [])
            probabilities = option.get("probabilities", [])

            if len(outcomes) != len(probabilities):
                logger.warning(f"Mismatch in lengths of outcomes and probabilities in option '{option.get('name', '?')}'")
                outcomes = outcomes[:len(probabilities)] if len(outcomes) > len(probabilities) else outcomes
                probabilities = probabilities[:len(outcomes)] if len(probabilities) > len(outcomes) else probabilities

            ev = np.sum(np.array(outcomes) * np.array(probabilities))
            utility = 0.0

            if utility_type == "exponential":
                a = risk_params.get("a", 1.0)
                utility = np.sum(probabilities * (1 - np.exp(-a * np.array(outcomes))))
            elif utility_type == "logarithmic":
                a = risk_params.get("a", 1.0)
                utility = np.sum(probabilities * np.log(a * np.array(outcomes) + 1e-6))
            elif utility_type == "power":
                a = risk_params.get("a", 1.0)
                b = risk_params.get("b", 0.5)
                utility = np.sum(probabilities * (a * np.array(outcomes) ** b))
            else:
                utility = ev

            node_text = f"{option.get('name', 'Option')}\nEV: {ev:.2f}\nU: {utility:.2f}"
            ax.text(3, y, node_text, bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round, pad=0.5'),
                    ha='center', va='center', fontsize=9)

            for j, (outcome, prob) in enumerate(zip(outcomes, probabilities)):
                y_child = y - (j + 1) * 0.5
                ax.plot([3, 5], [y, y_child], 'k-', linewidth=0.5)
                outcome_text = f"Outcome: {outcome:.2f}\nP: {prob:.2f}"
                ax.text(5, y_child, outcome_text,
                        bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round, pad=0.3'),
                        ha='center', va='center', fontsize=8)

        ax.set_xlim(2, 6)
        ax.set_ylim(y_start - len(options) * y_step - 1, y_start + 1)
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=20)
        plt.tight_layout()
        return fig, ax
    except Exception as e:
        logger.error(f"Error in plot_decision_tree: {e}")
        return None, None

def plot_probability_distribution(
    options: List[Dict[str, Any]],
    title: str = "Probability Distribution"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Plot probability distribution for options.
    """
    if not options:
        logger.error("plot_probability_distribution: options list is empty")
        return None, None

    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = sns.color_palette("tab10", len(options))
        has_valid_data = False

        for i, opt in enumerate(options):
            outcomes = np.array(opt.get("outcomes", []))
            probs = np.array(opt.get("probabilities", []))

            if outcomes.size == 0 or probs.size == 0:
                logger.warning(f"Skipping option '{opt.get('name', '?')}' — empty outcomes/probabilities")
                continue

            if outcomes.size != probs.size:
                logger.warning(f"Skipping option '{opt.get('name', '?')}' — size mismatch")
                continue

            total_prob = np.sum(probs)
            if not np.isclose(total_prob, 1.0, rtol=1e-3):
                probs = probs / (total_prob + 1e-12)
                logger.info(f"Normalized probabilities for '{opt.get('name', '?')}'")

            ev = np.sum(outcomes * probs)
            var = np.sum(probs * (outcomes - ev) ** 2)
            std = np.sqrt(var)

            # Plot
            ax.stem(outcomes, probs, linefmt=f'{colors[i]}-', markerfmt=f'{colors[i]}o',
                    basefmt="", label=f'{opt.get("name", f"Option {i + 1}")} (EV={ev:.2f})')

            ax.axvline(ev, color=colors[i], linestyle='--', linewidth=1.2)
            has_valid_data = True

        if not has_valid_data:
            logger.error("plot_probability_distribution: no valid data to plot")
            plt.close(fig)
            return None, None

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        return fig, ax
    except Exception as e:
        logger.error(f"Error in plot_probability_distribution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def plot_weibull_analysis(
    shape: float = 1.0,
    scale: float = 1.0,
    title: str = "Weibull Distribution Analysis"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Plot PDF and CDF of the Weibull distribution.
    """
    try:
        if shape <= 0 or scale <= 0:
            logger.error(f"Invalid Weibull parameters: shape={shape}, scale={scale}")
            return None, None

        x = np.linspace(0, scale * 5, 300)
        pdf = (shape / scale) * (x / scale) ** (shape - 1) * np.exp(-(x / scale) ** shape)
        cdf = 1 - np.exp(-(x / scale) ** shape)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(x, pdf, 'b-', lw=2, label=f'PDF (k={shape:.2f}, λ={scale:.2f})')
        ax[0].set_title('Probability Density Function')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('f(x)')
        ax[0].legend()
        ax[0].grid(True, linestyle='--', alpha=0.6)

        ax[1].plot(x, cdf, 'r-', lw=2, label=f'CDF (k={shape:.2f}, λ={scale:.2f})')
        ax[1].set_title('Cumulative Distribution Function')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('F(x)')
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig, ax
    except Exception as e:
        logger.error(f"Error in plot_weibull_analysis: {e}")
        return None, None

def plot_expected_shortfall(
    mean: float = 0.0,
    std: float = 1.0,
    confidence_level: float = 0.95,
    title: str = "Expected Shortfall"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Plot ES for a normal distribution.
    """
    try:
        if not (0 < confidence_level < 1):
            logger.error(f"Invalid confidence level: {confidence_level}")
            return None, None

        alpha = 1.0 - confidence_level
        z_score = stats.norm.ppf(alpha)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
        y = stats.norm.pdf(x, mean, std)

        ax.plot(x, y, 'k-', lw=1.5, label='PDF')
        x_es = np.linspace(mean + z_score * std, mean - 4 * std, 200)
        y_es = stats.norm.pdf(x_es, mean, std)

        ax.fill_between(x_es, y_es, color='purple', alpha=0.3, label=f'ES ({confidence_level:.0%})')
        es_value = mean - std * stats.norm.pdf(z_score) / alpha

        ax.axvline(es_value, color='purple', linestyle='--', label=f'ES = {es_value:.2f}')
        ax.set_title(f"{title} — {confidence_level:.0%}", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        return fig, ax
    except Exception as e:
        logger.error(f"Error in plot_expected_shortfall: {e}")
        return None, None

def plot_risk_attitude(
    utility_type: str = "exponential",
    risk_params: Optional[Dict[str, float]] = None
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Plot utility function and second derivative → risk attitude.
    """
    try:
        risk_params = risk_params or {}
        x = np.linspace(0, 100, 200)

        if utility_type == "exponential":
            a = risk_params.get("a", 0.05)
            u = 1 - np.exp(-a * x)
            second_deriv = -a ** 2 * np.exp(-a * x)
        elif utility_type == "logarithmic":
            a = risk_params.get("a", 1.0)
            u = np.log(a * x + 1e-6)
            second_deriv = -a ** 2 / (a * x + 1e-6) ** 2
        elif utility_type == "power":
            a = risk_params.get("a", 1.0)
            b = risk_params.get("b", 0.5)
            u = a * (x ** b)
            second_deriv = a * b * (b - 1) * (x ** (b - 2))
        else:
            u = x
            second_deriv = np.zeros_like(x)

        fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax[0].plot(x, u, 'b-', label='Utility')
        ax[0].set_ylabel("U(x)")
        ax[0].legend()
        ax[0].grid(True, linestyle='--', alpha=0.6)

        ax[1].plot(x, second_deriv, 'r-', label="U''(x)")
        ax[1].axhline(0, color='gray', lw=0.8, linestyle=':')
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("U''(x)")
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        avg_sec = np.mean(second_deriv)
        if avg_sec < 0:
            attitude = "Risk Aversion"
        elif avg_sec > 0:
            attitude = "Risk Seeking"
        else:
            attitude = "Risk Neutral"

        fig.suptitle(f"Risk Attitude: {attitude}", fontsize=14)
        plt.tight_layout()
        return fig, ax
    except Exception as e:
        logger.error(f"Error in plot_risk_attitude: {e}")
        return None, None

def plot_venn_diagram(
    sets: List[Set[Any]],
    labels: Optional[List[str]] = None,
    title: str = "Venn Diagram"
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Simplified Venn diagram for 2–3 sets.
    """
    try:
        has_venn = False
        try:
            from matplotlib_venn import venn2, venn3
            has_venn = True
        except ImportError:
            logger.warning("matplotlib_venn not installed. Using simplified diagram.")

        if len(sets) == 2 and has_venn:
            fig, ax = plt.subplots(figsize=(6, 6))
            venn2(subsets=(len(sets[0] - sets[1]), len(sets[1] - sets[0]), len(sets[0] & sets[1])),
                  set_labels=labels or ("Set A", "Set B"), ax=ax)
            ax.set_title(title)
            return fig, ax
        elif len(sets) == 3 and has_venn:
            fig, ax = plt.subplots(figsize=(6, 6))
            venn3(subsets=(
                len(sets[0] - sets[1] - sets[2]),
                len(sets[1] - sets[0] - sets[2]),
                len(sets[0] & sets[1] - sets[2]),
                len(sets[2] - sets[0] - sets[1]),
                len(sets[0] & sets[2] - sets[1]),
                len(sets[1] & sets[2] - sets[0]),
                len(sets[0] & sets[1] & sets[2])
            ), set_labels=labels or ("A", "B", "C"), ax=ax)
            ax.set_title(title)
            return fig, ax
        else:
            # Fallback: 2 circles manually
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
                return fig, ax
            else:
                logger.error(f'plot_venn_diagram: supports 2 or 3 sets, got {len(sets)}')
                return None, None
    except Exception as e:
        logger.error(f"Error in plot_venn_diagram: {e}")
        return None, None
