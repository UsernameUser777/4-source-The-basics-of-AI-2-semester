# sensitivity_analysis.py
"""
Module for analyzing the stability (sensitivity) of the optimal solution to an LP problem.
Implements calculation of shadow prices, allowable changes in resources, and minimum prices for cargos.
Author: Stanislav Kolosov
Date: 2026
"""

import numpy as np
from typing import Dict, List, Tuple

# --- IMPORT FOR LOGGING ---
import logging
# ---
from simplex_solver import SimplexSolver

# --- CREATE LOGGER FOR THIS MODULE ---
logger = logging.getLogger(__name__)
# ---

class SensitivityAnalyzer:
    """
    Class for performing stability analysis of the optimal solution to an LP problem.

    Analysis includes:
    1. Calculation of shadow prices (dual estimates) for each constraint
    2. Determination of allowable intervals for changing resources without changing the basis
    3. Calculation of the minimum allowable price for unprofitable cargos
    4. Analysis of the impact of changing cargo availability on total profit
    """

    def __init__(self, solver: SimplexSolver):
        """
        Initialization of the stability analyzer.

        Args:
            solver (SimplexSolver): An instance of the solver with the found optimal solution.
            Requires that solver.solve() has been called successfully.
        """
        self.solver = solver
        # Dictionaries for storing analysis results
        self.shadow_prices: Dict[str, float] = {}  # {constraint_name: shadow_price}
        self.allowable_increase: Dict[str, float] = {}  # {constraint_name: max_increase}
        self.allowable_decrease: Dict[str, float] = {}  # {constraint_name: max_decrease}

        # List for storing information about unprofitable cargos
        self.unprofitable_cargos: List[Tuple[str, float, float]] = []  # (cargo, current_price, min_price)

        # Check that a solution has been found (there is iteration history)
        if not self.solver.iterations:
            raise ValueError("Problem solution not found. First call the solve() method.")

    def calculate_shadow_prices(self) -> Dict[str, float]:
        """
        Calculation of shadow prices (dual estimates) for each constraint.

        The shadow price shows how much the objective function will change when the right-hand side of the constraint increases by one unit (assuming the optimal basis is preserved).

        For a maximization problem, shadow prices are found in the index row F(X) in the positions corresponding to the slack variables (columns after n).

        Since the coefficients for the slack variables in the F(X) row already account for the sign (they are negative for maximization), the actual shadow price is equal to -(coefficient from the F(X) row).

        Returns:
            Dict[str, float]: Dictionary {constraint_name: shadow_price}.
        """
        # Get the final tableau from the last iteration
        last_tableau = self.solver.iterations[-1]['tableau']
        m = self.solver.m  # Number of constraints

        # Extract coefficients from the F(X) row for slack variables (columns n+1 to n+m)
        # These coefficients are denoted as r_s (reduced costs for s)
        # The shadow price for constraint i is equal to -r_s[i]
        # Indices of slack variable columns in tableau: from n+1 to n+m
        # Extract slice [n+1 : n+m+1]
        reduced_costs_slack = last_tableau[m, self.solver.n + 1:self.solver.n + m + 1]

        # Calculate shadow prices
        shadow_prices_raw = -reduced_costs_slack

        # Form a dictionary mapping constraint names and calculated shadow prices
        self.shadow_prices = {
            self.solver.constraint_names[i]: shadow_prices_raw[i]
            for i in range(m)
        }
        return self.shadow_prices

    def calculate_allowable_changes(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculation of allowable changes in the right-hand sides of constraints without changing the basis.

        This analysis shows how much a resource (b_i) can be increased or decreased so that the current optimal basis remains feasible and optimal.

        The dual simplex method or sensitivity analysis is used.

        Algorithm:
        1. Consider the final simplex tableau.
        2. For each constraint (row i) and the corresponding column of the slack variable s_i, analyze how changing b_i affects the values of the variables in the basis.
        3. Use the formula: Delta_b_i_max = min(B^{-1}e_i)j / |a_jk| for a_jk < 0 (increase)
            Delta_b_i_min = min(B^{-1}e_i)j / |a_jk| for a_jk > 0 (decrease)
            where B is the basis matrix, e_i is the unit vector.

        In practice, this reduces to analyzing the coefficients in the s_i column of the final tableau.

        Returns:
            Tuple[Dict, Dict]:
            - allowable increase for each constraint
            - allowable decrease for each constraint
        """
        # Get the final tableau and basis
        last_tableau = self.solver.iterations[-1]['tableau']
        m = self.solver.m
        n = self.solver.n

        # Clear old data
        self.allowable_increase = {}
        self.allowable_decrease = {}

        # Iterate over all constraints (each associated with one slack variable s_i)
        for i in range(m):
            constraint_name = self.solver.constraint_names[i]
            # b_i - current value of the right-hand side
            b_i = last_tableau[i, 0]

            # Index of the slack variable s_i column in the tableau: n + 1 + i
            # (n+1 - start of s columns, i - offset)
            slack_col = n + 1 + i

            # Extract coefficients from *all* rows (including F(X)) for this column
            col_coeffs = last_tableau[:, slack_col]

            # --- Calculation of allowable increase (Delta b_i+) ---
            # Determines how much b_i can be *increased* so that b_i >= 0
            # This depends on how b_i changes as a result of changing b_j in the system B*x = b_new
            # In terms of the simplex tableau: b_new_i = b_i + sum(T_jk * delta_b_k) for k
            # where T_jk is the element of the tableau in row j, column k (for slack variables k).
            # We consider the change of only one b_i.
            # b_new_i = b_i + T_ii * delta_b_i
            # To have b_new_i >= 0, we need delta_b_i >= -b_i / T_ii (if T_ii > 0)
            # delta_b_i <= -b_i / T_ii (if T_ii < 0)

            # However, to analyze the change of *one* b_i by delta_b_i,
            # we need to see how this change affects *all* b_j in the basis.
            # b_new = b_old + delta_b_i * B^{-1} * e_i
            # where e_i is the unit vector. In the simplex tableau, the column B^{-1} * e_i is the s_i column.
            # New value b_j = b_old_j + delta_b_i * T_ji
            # Condition: b_new_j >= 0 for all j
            # b_old_j + delta_b_i * T_ji >= 0
            # delta_b_i * T_ji >= -b_old_j
            # If T_ji > 0: delta_b_i >= -b_old_j / T_ji
            # If T_ji < 0: delta_b_i <= -b_old_j / T_ji
            # We are interested in max delta_b_i (increase) and min delta_b_i (decrease).
            # max delta_b_i = min_j (-b_old_j / T_ji) for j where T_ji < 0
            # min delta_b_i = max_j (-b_old_j / T_ji) for j where T_ji > 0
            # i.e., max_delta = min((-b_j / T_ji) for j where T_ji < 0)
            # min_delta = max(-b_j / T_ji) for j where T_ji > 0
            # allowable_increase = max_delta - current_b_i
            # allowable_decrease = current_b_i - min_delta

            # Simplified approach, often used in sensitivity analysis:
            # allowable_increase_i = min_j (b_j / |T_ji|) for j where T_ji < 0
            # allowable_decrease_i = min_j (b_j / |T_ji|) for j where T_ji > 0
            # This is an approximation, but often gives reasonable estimates.
            # The correct way: solve the system of inequalities b_old + delta_b_i * col_coeffs >= 0
            # for delta_b_i.

            # Implement a simplified approach, as in many educational examples
            increases = []
            decreases = []

            # Iterate over rows of basic variables (excluding F(X))
            for j in range(m):
                coeff = col_coeffs[j]  # T_ji - coefficient in row j, column s_i
                if coeff < -1e-10:  # a_ji < 0 (with numerical tolerance)
                    # delta_b_i <= b_j / |coeff| = b_j / (-coeff)
                    # This is a potential max increase
                    potential_inc = (b_j / (-coeff)) if (b_j := last_tableau[j, 0]) >= 0 else float('inf')
                    if potential_inc >= 0:  # Only if the result makes sense
                        increases.append(potential_inc)
                elif coeff > 1e-10:  # a_ji > 0
                    # delta_b_i >= -b_j / coeff
                    # This is a potential max decrease (in the negative direction)
                    # allowable_decrease = min (b_j / coeff)
                    potential_dec = (b_j / coeff) if (b_j := last_tableau[j, 0]) >= 0 else float('inf')
                    if potential_dec >= 0:
                        decreases.append(potential_dec)

            # Determine max increase and max decrease
            # If there are no constraints, the change is unlimited
            self.allowable_increase[constraint_name] = min(increases) if increases else float('inf')
            self.allowable_decrease[constraint_name] = min(decreases) if decreases else float('inf')

        return self.allowable_increase, self.allowable_decrease

    def calculate_min_price_for_unprofitable_cargos(self, cargo_data: List[Dict]) -> List[Tuple[str, float, float]]:
        """
        Calculation of the minimum allowable price for cargos that are not included in the optimal plan.

        Algorithm:
        1. Identify cargos with zero quantity in the optimal plan.
        2. For each such cargo, calculate the "reduction" (reduced cost):
           rc = c_j - sum(y_i * a_ij), where:
           - c_j is the current price (coefficient in the objective function) of cargo j
           - y_i is the shadow price of the i-th constraint
           - a_ij is the coefficient of variable x_j in the i-th constraint
        3. If rc < 0, the cargo is potentially profitable. Its minimum price for inclusion in the plan is c_j_new = c_j - rc (so that rc becomes >= 0).
           If rc >= 0, the cargo is not profitable even at the current price.

        Args:
            cargo_data (List[Dict]): List of cargo data with keys:
                - 'name': cargo name
                - 'price': current price (coefficient c_j)
                - 'weight': weight per unit of cargo (coefficient in weight constraint)
                - 'volume': volume per unit of cargo (coefficient in volume constraint)
                - 'index': cargo index (1-5), to match with variable names x_ij

        Returns:
            List[Tuple[str, float, float]]: List of tuples (cargo, current_price, min_price).
            Includes only cargos with rc < 0 (potentially unprofitable).
        """
        # Get the optimal plan and shadow prices
        optimal_plan = self.solver.get_optimal_plan()
        if not self.shadow_prices:
            self.calculate_shadow_prices()
        # Determine which cargos are not used (total quantity = 0)
        unused_cargo_indices = set()
        for cargo_info in cargo_data:
            cargo_idx = cargo_info['index']
            # Sum the quantity of this cargo across all compartments
            total_amount = sum(optimal_plan.get(f'x{cargo_idx}{j}', 0.0) for j in range(1, 4))
            if total_amount < 1e-5:  # Check for "zero" with tolerance
                unused_cargo_indices.add(cargo_idx)
        # Calculate minimum price for unprofitable cargos
        self.unprofitable_cargos = []
        for cargo_info in cargo_data:
            cargo_idx = cargo_info['index']
            if cargo_idx not in unused_cargo_indices:
                continue  # Skip if cargo is used
            # Get current price and cargo characteristics
            current_price = cargo_info['price']
            weight = cargo_info['weight']
            volume = cargo_info['volume']
            # Calculate "reduction" (reduced cost) rc = c_j - sum(y_i * a_ij)
            # a_ij - coefficient of x_ij in constraints
            # For cargo i and compartment j:
            # - in the weight constraint for compartment j: a_ij = weight
            # - in the volume constraint for compartment j: a_ij = volume
            # - in the availability constraint for cargo i: a_ij = 1 (for any j)
            # rc_j = current_price - [(sum over j (shadow_price_weight_j * weight)) +
            # (sum over j (shadow_price_volume_j * volume)) +
            # (shadow_price_availability_i * 1)]
            # Simplification: Calculate rc for x_i1 (cargo i in compartment 1), this will be the same for all x_ij
            # since the coefficients a_ij for different j (compartments) are the same for a fixed i (cargo).
            # rc_i = c_i - [ sum_j(shadow_price_weight_j * weight_i) +
            # sum_j(shadow_price_volume_j * volume_i) +
            # shadow_price_availability_i * 1 ]
            # More accurate calculation: sum rc_sum = 0.0
            rc_sum = 0.0  # Initialize rc_sum
            for j in range(1, 4):  # For each compartment (1, 2, 3)
                # Coefficient in the weight constraint for compartment j
                weight_constraint_name = f'Weight_Comp{j}'
                y_weight = self.shadow_prices.get(weight_constraint_name, 0.0)
                rc_sum += y_weight * weight
                # Coefficient in the volume constraint for compartment j
                volume_constraint_name = f'Volume_Comp{j}'
                y_volume = self.shadow_prices.get(volume_constraint_name, 0.0)
                rc_sum += y_volume * volume
                # Coefficient in the availability constraint for cargo i
                # Name of constraint: assumed format 'Availability_Cargo{idx}_{name}'
                # Find exact name by index
                availability_constraint_name = next((name for name in self.solver.constraint_names if f'Availability_Cargo{cargo_idx}_' in name), None)
                if availability_constraint_name:
                    y_availability = self.shadow_prices.get(availability_constraint_name, 0.0)
                    rc_sum += y_availability * 1.0  # Coefficient of x_ij in availability constraint = 1

            # Calculate reduction
            reduced_cost = current_price - rc_sum

            # Minimum price for profitability: c_new = c_old - rc
            # If rc < 0, then c_new > c_old, meaning the price needs to be increased
            # to compensate for the lack of profit due to constraints.
            # If rc >= 0, the cargo is not profitable anyway.
            min_price = current_price - reduced_cost if reduced_cost < 0 else current_price

            # Add to result if the cargo is truly unprofitable (rc < 0)
            if reduced_cost < 0:
                self.unprofitable_cargos.append((
                    cargo_info['name'],
                    current_price,
                    max(min_price, 0.0)  # Price cannot be negative
                ))

        return self.unprofitable_cargos

    def analyze_scenario(self, scenario_changes: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Analysis of profit change when parameters (cargo availability) change.
        This is a simplified scenario analysis, assuming that the structure of the optimal plan (which cargos in which compartments) does not change radically.

        Args:
            scenario_changes (Dict[str, float]): Dictionary of changes in variable values in the *already found* optimal plan.
            Key - variable name (e.g., 'x51'), value - new quantity (expected to be >= 0 and not violate constraints).

        Returns:
            Tuple[float, Dict[str, float]]:
            - delta (or loss) of profit (delta) when applying the scenario.
            - new "plan" with changed values (other values as in the optimal).
        """
        # Get the current optimal plan
        current_plan = self.solver.get_optimal_plan()
        current_profit = self.solver.get_objective_value()

        # Apply changes to the plan
        new_plan = current_plan.copy()
        delta_profit = 0.0

        # Iterate over scenario changes
        for var_name, new_value in scenario_changes.items():
            if var_name in new_plan:
                old_value = new_plan[var_name]
                delta = new_value - old_value

                # Determine cargo index from variable name (e.g., 'x51' -> 5)
                try:
                    cargo_idx_str = var_name[1]  # Second character (after 'x')
                    # If variable name is xIJ, where IJ is a two-digit index, use slice
                    # cargo_idx_str = var_name[1] # For now, assume single-digit index
                    cargo_idx = int(cargo_idx_str)
                except (ValueError, IndexError):
                    # --- FIXED: use local logger ---
                    logger.warning(f'Failed to determine cargo index from variable "{var_name}". Skipping.')
                    # ---
                    continue

                # Find the price of this cargo (coefficient in the objective function)
                # This can be done by knowing how the vector c was built in _build_problem_matrices
                # In this case, the price depends only on the cargo index
                # Use helper method or assume structure
                price = self._get_cargo_price(cargo_idx)

                # Change in profit = (new quantity - old quantity) * price
                delta_profit += delta * price
                # Update value in new plan
                new_plan[var_name] = new_value

        return delta_profit, new_plan

    def _get_cargo_price(self, cargo_index: int) -> float:
        """
        Helper method to get the price of a cargo by its index.

        Args:
            cargo_index (int): Cargo index (1-5).

        Returns:
            float: Price per unit of cargo.
        """
        # Mapping cargo index to its price (taken from _build_problem_matrices)
        cargo_prices = {
            1: 8.0,  # Mini-tractors
            2: 21.5,  # Paper
            3: 51.0,  # Containers
            4: 275.0,  # Rolled metal
            5: 110.0  # Timber
        }
        return cargo_prices.get(cargo_index, 0.0)

    def generate_stability_report(self) -> str:
        """
        Generates a text report on the stability of the solution.

        Fixed: protection against string formatting errors instead of numbers.
        """
        # Calculate all analysis indicators if not already calculated
        if not self.shadow_prices:
            self.calculate_shadow_prices()
        if not self.allowable_increase or not self.allowable_decrease:
            self.calculate_allowable_changes()

        report = "=" * 80 + "\n"
        report += "STABILITY REPORT OF THE OPTIMAL SOLUTION\n"
        report += "(SENSITIVITY ANALYSIS)\n"
        report += "=" * 80 + "\n\n"

        # --- Helper: function for safe number formatting ---
        def safe_float_format(value, fmt=".4f"):
            """
            Converts value to float and formats it, or returns 'N/A'.
            """
            try:
                if isinstance(value, (int, float)):
                    return f"{value:{fmt}}"
                elif isinstance(value, str):
                    return f"{float(value):{fmt}}"
                else:
                    return "N/A"
            except (ValueError, TypeError):
                return "N/A"

        # 1. Shadow prices
        report += "1. SHADOW PRICES (DUAL ESTIMATES)\n"
        report += "-" * 80 + "\n"
        report += f"{'Constraint':<25} | {'Shadow Price':>15} | {'Economic Meaning':<35}\n"
        report += "-" * 80 + "\n"

        for constraint, price in self.shadow_prices.items():
            if 'Weight' in constraint:
                meaning = "Income from +1 t of load capacity"
            elif 'Volume' in constraint:
                meaning = "Income from +1 m³ of volume"
            elif 'Availability' in constraint:
                meaning = "Income from +1 unit of cargo"
            else:
                meaning = "—"

            price_str = safe_float_format(price)
            report += f"{constraint:<25} | {price_str:>15} | {meaning:<35}\n"
        report += "\n"

        # 2. Allowable changes
        report += "2. ALLOWABLE CHANGES IN RESOURCES WITHOUT CHANGING THE BASIS\n"
        report += "-" * 80 + "\n"
        report += f"{'Constraint':<25} | {'Current':>10} | {'+Δ':>10} | {'-Δ':>10} | {'New Range':<20}\n"
        report += "-" * 80 + "\n"

        # Original values of the right-hand sides of constraints (from the initial problem)
        original_b = self.solver.b

        # Ensure we do not exceed the bounds of the array
        for i in range(min(len(self.solver.constraint_names), len(original_b))):
            constraint = self.solver.constraint_names[i]
            current = original_b[i]
            inc = self.allowable_increase.get(constraint, float('inf'))
            dec = self.allowable_decrease.get(constraint, float('inf'))

            current_str = safe_float_format(current)
            inc_str = safe_float_format(inc, ".1f") if inc != float('inf') else "∞"
            dec_str = safe_float_format(dec, ".1f") if dec != float('inf') else "∞"
            range_start = safe_float_format(current - dec, ".1f") if dec != float('inf') else "-∞"
            range_end = safe_float_format(current + inc, ".1f") if inc != float('inf') else "+∞"
            range_str = f"[{range_start}; {range_end}]"
            report += f"{constraint:<25} | {current_str:>10} | {inc_str:>10} | {dec_str:>10} | {range_str:<20}\n"
        report += "\n"

        # 3. Unprofitable cargos
        if self.unprofitable_cargos:
            report += "3. UNPROFITABLE CARGOS AND MINIMUM ALLOWABLE PRICE\n"
            report += "-" * 80 + "\n"
            report += f"{'Cargo':<20} | {'Current Price':>15} | {'Min. Price':>15} | {'Difference':>15}\n"
            report += "-" * 80 + "\n"

            for cargo, current_price, min_price in self.unprofitable_cargos:
                diff = min_price - current_price
                curr_str = safe_float_format(current_price)
                min_str = safe_float_format(min_price)
                diff_str = safe_float_format(diff)
                report += f"{cargo:<20} | {curr_str:>15} | {min_str:>15} | {diff_str:>15}\n"

            report += "\nNote: The cargo will become profitable when the price is increased by the difference.\n"
        else:
            report += "3. UNPROFITABLE CARGOS AND MINIMUM ALLOWABLE PRICE\n"
            report += "-" * 80 + "\n"
            report += "All cargos available in the specified quantity are included in the optimal plan.\n"
        report += "\n" + "=" * 80 + "\n"
        return report
