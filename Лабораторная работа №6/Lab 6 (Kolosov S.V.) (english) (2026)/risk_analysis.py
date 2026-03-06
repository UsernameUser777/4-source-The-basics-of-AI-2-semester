# Author: Kolosov S.V., IVT-3, 4th year
# Lab work №6, variant №1, 2026
# Full risk analysis implementation with error handling and compliance with requirements

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy import stats

# Import logger
try:
    from utils.logger import logger
except ImportError:
    logger = logging.getLogger("RiskAnalysis")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def calculate_expected_value(outcomes: List[float], probabilities: List[float]) -> float:
    """
    Calculate the expected value (Expected Value) for given outcomes and probabilities.
    """
    try:
        if len(outcomes) != len(probabilities):
            logger.error(f"Length mismatch: outcomes={len(outcomes)}, probabilities={len(probabilities)})")
            return 0.0

        outcomes_arr = np.asarray(outcomes, dtype=np.float64)
        probs_arr = np.asarray(probabilities, dtype=np.float64)

        total_prob = np.sum(probs_arr)
        if not np.isclose(total_prob, 1.0):
            probs_arr = probs_arr / (total_prob + 1e-12)

        ev = np.sum(outcomes_arr * probs_arr)
        return float(ev)
    except Exception as e:
        logger.error(f"Error in calculate_expected_value: {e}")
        return 0.0

def calculate_variance_and_std(outcomes: List[float], probabilities: List[float],
                                expected_value: Optional[float] = None) -> Tuple[float, float]:
    """
    Calculate variance and standard deviation.
    """
    try:
        if len(outcomes) != len(probabilities):
            logger.error(f"Length mismatch: outcomes={len(outcomes)}, probabilities={len(probabilities)}")
            return 0.0, 0.0

        outcomes_arr = np.asarray(outcomes, dtype=np.float64)
        probs_arr = np.asarray(probabilities, dtype=np.float64)

        if expected_value is None:
            expected_value = calculate_expected_value(outcomes, probabilities)

        variance = np.sum(probs_arr * (outcomes_arr - expected_value) ** 2)
        std_dev = np.sqrt(variance)

        return float(variance), float(std_dev)
    except Exception as e:
        logger.error(f"Error in calculate_variance_and_std: {e}")
        return 0.0, 0.0

def calculate_utility(outcomes: List[float], probabilities: List[float],
                      utility_type: str = "linear",
                      risk_params: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate expected utility using different models.
    """
    try:
        risk_params = risk_params or {}

        outcomes_arr = np.asarray(outcomes, dtype=np.float64)
        probs_arr = np.asarray(probabilities, dtype=np.float64)

        if utility_type == "exponential":
            a = risk_params.get("a", 1.0)
            utilities = 1 - np.exp(-a * outcomes_arr)
        elif utility_type == "logarithmic":
            a = risk_params.get("a", 1.0)
            adjusted_outcomes = np.where(outcomes_arr <= 0, 1e-6, outcomes_arr)
            utilities = np.log(a * adjusted_outcomes)
        elif utility_type == "power":
            a = risk_params.get("a", 1.0)
            b = risk_params.get("b", 0.5)
            adjusted_outcomes = np.where(outcomes_arr < 0, 0.0, outcomes_arr)
            utilities = a * (adjusted_outcomes ** b)
        else:  # linear
            utilities = outcomes_arr

        expected_utility = np.sum(probs_arr * utilities)
        return float(expected_utility)
    except Exception as e:
        logger.error(f"Error in calculate_utility: {e}")
        return 0.0

def value_at_risk(returns: List[float], confidence_level: float = 0.95,
                  method: str = "historical") -> float:
    """
    Calculate Value at Risk (VaR) using historical data or normal distribution.
    """
    try:
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.size == 0:
            logger.error("Empty returns array for VaR")
            return 0.0

        if method == "historical":
            if confidence_level <= 0 or confidence_level >= 1:
                logger.error(f"Invalid confidence level for VaR: {confidence_level}")
                return 0.0

            alpha = 1.0 - confidence_level
            var = np.quantile(returns_arr, alpha)
            return float(var)

        elif method == "normal":
            mean = np.mean(returns_arr)
            std = np.std(returns_arr, ddof=1)

            if std == 0:
                logger.warning("Standard deviation = 0, cannot calculate normal VaR")
                return float(mean)

            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean - std * z_score
            return float(var)

        else:
            logger.error(f"Unknown VaR method: {method}")
            return 0.0
    except Exception as e:
        logger.error(f"Error in value_at_risk: {e}")
        return 0.0

def expected_shortfall(returns: List[float], confidence_level: float = 0.95, method: str = "historical") -> float:
    """
    Calculate Expected Shortfall (Conditional Value at Risk).
    """
    try:
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.size == 0:
            logger.error("Empty returns array for ES")
            return 0.0

        if method == "historical":
            if confidence_level <= 0 or confidence_level >= 1:
                logger.error(f"Invalid confidence level for ES: {confidence_level}")
                return 0.0

            alpha = 1.0 - confidence_level
            var_threshold = np.quantile(returns_arr, alpha)
            tail_returns = returns_arr[returns_arr <= var_threshold]

            if tail_returns.size == 0:
                logger.warning("No data in tail for ES")
                return float(var_threshold)

            es = np.mean(tail_returns)
            return float(es)

        elif method == "normal":
            mean = np.mean(returns_arr)
            std = np.std(returns_arr, ddof=1)

            if std == 0:
                logger.warning("Standard deviation = 0, cannot calculate normal ES")
                return float(mean)

            z_score = stats.norm.ppf(1 - confidence_level)
            es = mean - std * (stats.norm.pdf(z_score) / (1 - confidence_level))
            return float(es)

        else:
            logger.error(f"Unknown ES method: {method}")
            return 0.0
    except Exception as e:
        logger.error(f"Error in expected_shortfall: {e}")
        return 0.0

def monte_carlo_simulation(initial_value: float, mean_return: float, volatility: float, time_horizon: int, n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Perform Monte Carlo simulation for price/return forecasting.
    """
    try:
        if n_simulations <= 0:
            logger.error(f"n_simulations must be > 0, got: {n_simulations}")
            n_simulations = 1000

        dt = 1.0 / time_horizon
        Z = np.random.normal(size=(time_horizon, n_simulations))
        returns_paths = (mean_return - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z
        cumulative_returns = np.cumsum(returns_paths, axis=0)
        price_paths = initial_value * np.exp(cumulative_returns)

        final_prices = price_paths[-1, :]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        min_final_price = np.min(final_prices)
        max_final_price = np.max(final_prices)
        ci_lower = np.percentile(final_prices, 2.5)
        ci_upper = np.percentile(final_prices, 97.5)

        return {
            "final_prices": final_prices.tolist(),
            "mean": mean_final_price,
            "std_dev": std_final_price,
            "min": min_final_price,
            "max": max_final_price,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    except Exception as e:
        logger.error(f"Error in monte_carlo_simulation: {e}")
        return {
            "final_prices": [],
            "mean": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0
        }

def fuzzy_set_membership(x: float, params: Dict[str, float]) -> float:
    """
    Calculate membership degree to a fuzzy set (trapezoidal).
    """
    try:
        a = params.get("a", 0.0)
        b = params.get("b", 1.0)
        c = params.get("c", 2.0)
        d = params.get("d", 3.0)

        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error in fuzzy_set_membership: {e}")
        return 0.0

def pairwise_comparison_matrix(
    criteria_names: List[str],
    comparisons: List[Tuple[int, int, float]]
) -> np.ndarray:
    """
    Create a pairwise comparison matrix from a list of comparisons (i, j, value).
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
        logger.error(f"Error in pairwise_comparison_matrix: {e}")
        return np.ones((len(criteria_names), len(criteria_names)))

def ahp_analyze(matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Perform Analytic Hierarchy Process (AHP) analysis.
    Returns weights, consistency index, and consistency ratio.
    """
    try:
        n = matrix.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        max_eigenvalue = np.real(eigenvalues).max()
        principal_eigenvector = np.real(eigenvectors[:, np.argmax(eigenvalues)])
        weights = np.abs(principal_eigenvector) / np.sum(np.abs(principal_eigenvector))

        ci = (max_eigenvalue - n) / (n - 1) if n > 1 else 0.0

        ri_values = {
            1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24,
            7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }
        ri = ri_values.get(n, 1.49)

        cr = ci / ri if ri != 0 else 0.0

        return weights, ci, cr
    except Exception as e:
        logger.error(f"Error in ahp_analyze: {e}")
        n = matrix.shape[0]
        return np.ones(n) / n, 0.0, 0.0

def topsis_analyze(decision_matrix: np.ndarray, weights: np.ndarray,
                   criteria_types: List[bool]) -> np.ndarray:
    """
    Perform TOPSIS analysis.
    """
    try:
        squared_sums = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
        normalized_matrix = decision_matrix / (squared_sums + 1e-12)
        weighted_matrix = normalized_matrix * weights

        ideal_best = np.where(criteria_types, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
        ideal_worst = np.where(criteria_types, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))

        dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

        scores = dist_worst / (dist_best + dist_worst + 1e-12)
        return scores
    except Exception as e:
        logger.error(f"Error in topsis_analyze: {e}")
        return np.zeros(decision_matrix.shape[0])

def scenario_analysis(base_value: float, optimistic_factor: float, pessimistic_factor: float) -> Dict[str, float]:
    """
    Perform scenario analysis (pessimistic, base, optimistic).
    """
    try:
        optimistic_value = base_value * (1 + optimistic_factor)
        pessimistic_value = base_value * (1 - pessimistic_factor)

        return {
            "optimistic": optimistic_value,
            "base": base_value,
            "pessimistic": pessimistic_value
        }
    except Exception as e:
        logger.error(f"Error in scenario_analysis: {e}")
        return {"optimistic": 0.0, "base": 0.0, "pessimistic": 0.0}

def generate_recommendations(monte_carlo_results: Dict[str, Any]) -> str:
    """
    Generate recommendations based on analysis results.
    """
    try:
        std_dev = monte_carlo_results.get("std_dev", 0.0)
        ci_lower = monte_carlo_results.get("ci_lower", 0.0)
        ci_upper = monte_carlo_results.get("ci_upper", 0.0)

        recommendations = "===== RECOMMENDATIONS ====\n\n"
        if std_dev > 20:
            recommendations += "- Diversification or hedging is recommended.\n"
        else:
            recommendations += "- Risk level is acceptable.\n"

        if ci_lower < 0:
            recommendations += "- Loss is possible in the worst-case scenario (lower bound < 0).\n"

        if ci_upper - ci_lower > 100:
            recommendations += "- High forecast uncertainty (wide confidence interval).\n"

        return recommendations
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {e}")
        return "===== RECOMMENDATIONS ====\nFailed to generate recommendations.\n"

def sensitivity_analysis(outcomes: List[float], probabilities: List[float],
                          probability_change: float = 0.05) -> Dict[str, Any]:
    """
    Calculate sensitivity of expected value to probability change.
    """
    try:
        base_ev = calculate_expected_value(outcomes, probabilities)

        # Increase probability of the first outcome
        probs_up = probabilities.copy()
        if len(probs_up) > 1:
            probs_up[0] = min(1.0, probs_up[0] + probability_change)
            probs_up[1] = max(0.0, probs_up[1] - probability_change)
            ev_up = calculate_expected_value(outcomes, probs_up)

        # Decrease probability of the first outcome
        probs_down = probabilities.copy()
        if len(probs_down) > 1:
            probs_down[0] = max(0.0, probs_down[0] - probability_change)
            probs_down[1] = min(1.0, probs_down[1] + probability_change)
            ev_down = calculate_expected_value(outcomes, probs_down)

        sensitivity = (ev_up - ev_down) / (2 * probability_change) if probability_change > 0 else 0.0

        return {
            "base_ev": base_ev,
            "ev_up": ev_up,
            "ev_down": ev_down,
            "sensitivity": sensitivity,
            "probability_change": probability_change
        }
    except Exception as e:
        logger.error(f"Error in sensitivity_analysis: {e}")
        return {
            "base_ev": 0.0,
            "ev_up": 0.0,
            "ev_down": 0.0,
            "sensitivity": 0.0,
            "probability_change": probability_change
        }

if __name__ == "__main__":
    print("--- Testing risk_analysis.py ---")

    outcomes = [100, 200, -50]
    probs = [0.3, 0.5, 0.2]
    ev = calculate_expected_value(outcomes, probs)
    print(f"Expected value: {ev}")

    var, std = calculate_variance_and_std(outcomes, probs, ev)
    print(f"Variance: {var}, Std Dev: {std}")

    util = calculate_utility(outcomes, probs, "exponential", {"a": 0.01})
    print(f"Utility (exp): {util}")

    returns = np.random.normal(0.01, 0.05, 1000).tolist()
    var_hist = value_at_risk(returns, 0.95, "historical")
    print(f"VaR (historical): {var_hist}")

    es_hist = expected_shortfall(returns, 0.95, "historical")
    print(f"ES (historical): {es_hist}")

    mc_res = monte_carlo_simulation(100, 0.05, 0.2, 252, 1000)
    print(f"MC Mean: {mc_res['mean']:.2f}, Std: {mc_res['std_dev']:.2f}")

    recs = generate_recommendations(mc_res)
    print(recs)

    sens = sensitivity_analysis(outcomes, probs, 0.05)
    print(f"Sensitivity analysis: {sens}")

    print("File risk_analysis.py is working correctly.")
