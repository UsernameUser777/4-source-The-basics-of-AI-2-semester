# simplex_solver.py
"""
Implementation of the simplex method for solving a linear programming problem.
Variant #1: Optimization of cargo loading for a cargo-passenger ship.
Author: Stanislav Kolosov
Date: 2026
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# --- Logging setup ---
# Use standard INFO level.
# For debugging, you can set level=logging.DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class SimplexSolver:
    """
    Class for solving a linear programming problem using the simplex method.

    Supports:
    - Solving the problem of maximizing profit from cargo transportation
    - Output of all iterations of simplex tables
    - Determination of the optimal cargo distribution plan
    - Analysis of degeneracy and cycling
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                 var_names: List[str], constraint_names: List[str]):
        """
        Initialization of the simplex solver.

        Args:
            c (np.ndarray): Vector of objective function coefficients (dimension n).
                For a maximization problem, each element c[j] represents the coefficient of variable x_j in the objective function F = sum(c[j] * x[j]).
            A (np.ndarray): Matrix of coefficients of the left-hand sides of constraints (dimension m x n).
                Each element A[i][j] is the coefficient of variable x_j in the i-th constraint.
            b (np.ndarray): Vector of right-hand sides of constraints (dimension m).
                Each element b[i] is the right-hand side of the i-th constraint.
                All elements b[i] must be >= 0.
            var_names (List[str]): Names of decision variables (e.g., ['x11', 'x12', ...]).
                Must be unique and correspond to the order in vector c.
            constraint_names (List[str]): Names of constraints (e.g., ['Weight_Comp1', 'Volume_Comp1', ...]).
                Must be unique and correspond to the order in vector b.

        Note:
            The problem is assumed to be in canonical form:
            max c^T * x
            subject to A * x <= b, x >= 0.

            The solver automatically converts it to standard form by adding slack variables s_i (s >= 0), turning inequalities into equalities:
            A * x + s = b.

            Thus, the total number of variables in the simplex tableau will be n + m.
        """
        # --- Check correctness of input data ---
        # Check that the number of objective function coefficients equals the number of columns in A
        if len(c) != A.shape[1]:
            raise ValueError(
                f"Dimension mismatch: length of vector c ({len(c)}) != number of columns in A ({A.shape[1]})"
            )

        # Check that the number of right-hand sides equals the number of rows in A
        if len(b) != A.shape[0]:
            raise ValueError(
                f"Dimension mismatch: length of vector b ({len(b)}) != number of rows in A ({A.shape[0]})"
            )

        # Check that all elements of b are non-negative (canonical form requirement)
        if any(bi < 0 for bi in b):
            raise ValueError("Right-hand sides of constraints (b) cannot be negative.")

        # Save input data as object attributes, converting to float
        self.c = c.astype(float)
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.var_names = var_names
        self.constraint_names = constraint_names

        # Problem dimensions
        self.m, self.n = A.shape  # m - number of constraints, n - number of decision variables x

        # --- Initialize simplex tableau ---
        # Tableau structure:
        # | Basis | B | x1 | x2 | ... | xn | s1 | s2 | ... | sm |
        # | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
        # | s1 | b1 | a11 | a12 | ... | a1n | 1 | 0 | ... | 0 |
        # | s2 | b2 | a21 | a22 | ... | a2n | 0 | 1 | ... | 0 |
        # | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
        # | sm | bm | am1 | am2 | ... | amn | 0 | 0 | ... | 1 |
        # | F(X) | 0 | -c1 | -c2 | ... | -cn | 0 | 0 | ... | 0 |

        # Create a zero tableau of size (m+1) x (n+m+1)
        # +1 for the objective function row, +1 for the B column
        self.tableau = np.zeros((self.m + 1, self.n + self.m + 1))

        # Fill the free terms column (B)
        self.tableau[:self.m, 0] = self.b

        # Fill the coefficients of decision variables x
        self.tableau[:self.m, 1:self.n + 1] = self.A

        # Fill the identity matrix for slack variables s (left of the diagonal)
        self.tableau[:self.m, self.n + 1:self.n + self.m + 1] = np.eye(self.m)

        # Fill the objective function row F(X) with the opposite sign (-c)
        # This is because we are solving a maximization problem, and the simplex method is usually implemented for minimization.
        # max F = - min(-F), so the coefficients become -c_j
        self.tableau[self.m, 1:self.n + 1] = -self.c

        # Coefficients of slack variables in the F(X) row are 0
        # Names of all variables (main x + slack s)
        self.all_var_names = var_names + [f's{i + 1}' for i in range(self.m)]

        # Current basis (indices of variables in the basis).
        # Initially, the basis consists of slack variables s1..sm.
        # Their indices in the extended tableau: n (for s1), n+1 (for s2), ..., n+m-1 (for sm)
        self.basis = list(range(self.n, self.n + self.m))

        # History of iterations for reporting
        self.iterations: List[Dict] = []

        # Log the start of work
        logger.info(f"Simplex solver initialized: {self.n} variables, {self.m} constraints")
        logger.info(f"Objective function: max F = {' + '.join([f'{c_i:.1f}*{name}' for c_i, name in zip(c, var_names)])}")

        # Save initial state (iteration 0)
        self._save_iteration(0)

    def solve(self, max_iterations: int = 100) -> Tuple[bool, np.ndarray, float]:
        """
        Main method for solving the problem using the simplex method.

        Args:
            max_iterations (int): Maximum number of iterations to prevent cycling.

        Returns:
            Tuple[bool, np.ndarray, float]:
            - success: True if an optimal plan is found, False if the problem is unsolvable or the iteration limit is exceeded.
            - solution: Vector of the optimal solution (dimension n, only for variables x).
            - optimal_value: Value of the objective function at the optimal point.
        """
        iteration = 0
        logger.info("Starting simplex method solution...")

        # Main loop of the simplex method
        while iteration < max_iterations:
            # --- Step 1: Check for optimality ---
            # Extract the index row (objective function row F(X))
            # It is located in the last row of the tableau (self.m)
            # Consider coefficients for all variables (x and s)
            index_row = self.tableau[self.m, 1:self.n + self.m + 1]

            # For a maximization problem, the optimal plan is reached when all
            # coefficients in the index row <= 0 (with numerical tolerance).
            # Check if there are any positive ones.
            positive_coeffs_mask = index_row > 1e-10  # Threshold for numerical error

            # If there are no positive coefficients, the plan is optimal
            if not np.any(positive_coeffs_mask):
                logger.info(f"Optimal plan found at iteration {iteration}")
                # Save the final state, which is optimal
                self._save_iteration(iteration)
                break  # Exit the loop

            # --- Step 2: Select the pivot column (entering variable) ---
            # Select the variable that will enter the basis.
            # Usually, the variable with the largest positive coefficient
            # in the index row is selected (largest coefficient rule).
            pivot_col_idx = np.argmax(index_row)  # Column index (excluding B column)
            pivot_col = pivot_col_idx + 1  # Column index in the tableau (considering B column)

            # --- Step 3: Select the pivot row (leaving variable) ---
            # Select the variable that will leave the basis.
            # Calculate simplex ratios (bi / aij) for positive elements aij
            # in the pivot column. Select the row with the minimum positive ratio.
            ratios = []
            for i in range(self.m):  # Iterate over constraint rows
                if self.tableau[i, pivot_col] > 1e-10:  # Only if element > 0
                    ratio = self.tableau[i, 0] / self.tableau[i, pivot_col]  # bi / aij
                    ratios.append((ratio, i))  # Save ratio and row index
                else:
                    ratios.append((float('inf'), i))  # If aij <= 0, ratio is inf

            # If all ratios are inf, the problem is unbounded
            if all(ratio == float('inf') for ratio, _ in ratios):
                logger.error("Problem is unbounded (objective function can be increased to infinity)")
                unbounded_var = self.all_var_names[pivot_col_idx]
                logger.error(f"Unbounded variable: {unbounded_var}")
                return False, np.zeros(self.n), float('inf')

            # Find the row with the minimum positive ratio
            pivot_row = min(ratios, key=lambda x: x[0])[1]  # Take the row index
            pivot_element = self.tableau[pivot_row, pivot_col]  # Get the pivot element

            # Check for degeneracy (if bi = 0, this can lead to cycling)
            if abs(self.tableau[pivot_row, 0]) < 1e-10:
                logger.warning(f"Degeneracy detected at iteration {iteration} (b_{pivot_row + 1} = 0)")

            # Save information about the current iteration before transforming the tableau
            entering_var = self.all_var_names[pivot_col_idx]
            leaving_var = self.all_var_names[self.basis[pivot_row]]
            logger.info(f"Iteration {iteration + 1}: entering {entering_var}, leaving {leaving_var}, "
                        f"pivot element = {pivot_element:.4f} (row {pivot_row + 1}, column {pivot_col_idx + 1})")

            # --- Step 4: Transform the tableau (Jordan-Gauss method) ---
            # Normalize the pivot row by dividing by the pivot element
            self.tableau[pivot_row, :] = self.tableau[pivot_row, :] / pivot_element

            # Zero out the other elements of the pivot column
            for i in range(self.m + 1):  # Iterate over all rows (including F(X))
                if i != pivot_row:  # Skip the pivot row
                    factor = self.tableau[i, pivot_col]  # Current element of the pivot column
                    # Subtract the pivot row multiplied by the factor from the current row
                    self.tableau[i, :] -= factor * self.tableau[pivot_row, :]

            # Update the basis: replace the variable leaving the basis with the entering one
            self.basis[pivot_row] = pivot_col_idx

            # Save the results of the current iteration (after transformation)
            self._save_iteration(iteration, pivot_row, pivot_col, pivot_element, entering_var, leaving_var)

            # Increment iteration counter
            iteration += 1

            # Check for exceeding the iteration limit (possible cycling)
            if iteration >= max_iterations:
                logger.warning(f'Iteration limit reached ({max_iterations}). Possible degeneracy or cycling in the problem.')

                # Even if the limit is exceeded, the last saved tableau may be useful
                # But logically, the solution is not found successfully.
                return False, np.zeros(self.n), 0.0

        # --- Form the result ---
        # After the loop ends (if there was no iteration limit exceeded), the tableau is optimal
        # Form the solution vector only for variables x (dimension n)
        solution = np.zeros(self.n)
        for i, basis_idx in enumerate(self.basis):
            # If the basis variable is one of the original variables x (not s)
            if basis_idx < self.n:
                # Its value is equal to the element in the B column in the corresponding row
                solution[basis_idx] = self.tableau[i, 0]

        # The value of the objective function is in the bottom right corner of the tableau (F(X), B)
        optimal_value = self.tableau[self.m, 0]

        # Log the result
        logger.info(f'Solution completed. Optimal value F = {optimal_value:.2f}')
        logger.info(f'Optimal plan: {dict(zip(self.var_names, solution.round(2)))}')

        # Return success, found solution, and function value
        return True, solution, optimal_value

    def _save_iteration(self, iteration_num: int, pivot_row: Optional[int] = None,
                        pivot_col: Optional[int] = None, pivot_element: Optional[float] = None,
                        entering_var: Optional[str] = None, leaving_var: Optional[str] = None):
        """
        Saves the current state of the simplex tableau to the iteration history.

        Args:
            iteration_num (int): Iteration number.
            pivot_row (int, optional): Index of the pivot row.
            pivot_col (int, optional): Index of the pivot column (considering B column).
            pivot_element (float, optional): Value of the pivot element.
            entering_var (str, optional): Name of the variable entering the basis.
            leaving_var (str, optional): Name of the variable leaving the basis.
        """
        # Create a dictionary with data of the current iteration
        iteration_data = {
            'iteration': iteration_num,
            'tableau': self.tableau.copy(),  # IMPORTANT: make a copy of the tableau
            'basis': self.basis.copy(),      # IMPORTANT: make a copy of the basis
            'pivot_row': pivot_row,
            'pivot_col': pivot_col,
            'pivot_element': pivot_element,
            'entering_var': entering_var,
            'leaving_var': leaving_var
        }

        # Add the dictionary to the history list
        self.iterations.append(iteration_data)

    def get_iteration_table(self, iteration_num: int) -> str:
        """
        Generates a text representation of the simplex tableau for the specified iteration.

        Args:
            iteration_num (int): Iteration number (0 - initial tableau).

        Returns:
            str: Formatted string with the simplex tableau.
        """
        # Check if the requested iteration exists
        if iteration_num >= len(self.iterations):
            return f"Iteration {iteration_num} not found. Available iterations: {len(self.iterations)}"

        # Get data for the specific iteration
        iter_data = self.iterations[iteration_num]
        tableau = iter_data['tableau']
        basis = iter_data['basis']

        # Form the tableau header
        header = f"{'Basis':<12} | {'B':>10} | "
        for name in self.all_var_names:
            header += f"{name:>10} | "
        header += "\n" + "-" * (12 + 13 + (13 * len(self.all_var_names)))

        # Form the tableau rows (constraints)
        rows = []
        for i in range(self.m):
            # Name of the basis variable for row i
            basis_var = self.all_var_names[basis[i]]
            # Start forming the row with the basis variable and B value
            row_str = f"{basis_var:<12} | {tableau[i, 0]:>10.4f} | "
            # Add coefficients for all variables (x and s)
            for j in range(1, self.n + self.m + 1):
                row_str += f"{tableau[i, j]:>10.4f} | "
            rows.append(row_str)

        # Form the objective function row (F(X))
        f_row = f"{'F(X)':<12} | {tableau[self.m, 0]:>10.4f} | "
        for j in range(1, self.n + self.m + 1):
            f_row += f"{tableau[self.m, j]:>10.4f} | "

        # Form information about the pivot element (if it exists for this iteration)
        pivot_info = ""
        if iter_data['pivot_row'] is not None:
            pivot_info = (f"\nPivot element: {iter_data['pivot_element']:.4f} "
                          f"(row {iter_data['pivot_row'] + 1}, column {iter_data['pivot_col']})\n"
                          f"Entering: {iter_data['entering_var']}, Leaving: {iter_data['leaving_var']}")

        # Assemble the final string
        result = f"\nIteration #{iteration_num}{pivot_info}\n{header}\n"
        result += "\n".join(rows) + "\n" + f_row

        return result

    def get_optimal_plan(self) -> Dict[str, float]:
        """
        Returns the optimal plan as a dictionary {variable_name: value}.

        Returns:
            Dict[str, float]: Dictionary with optimal values of variables x.
            If the solution is not yet found, returns an empty dictionary.
        """
        # Check if there is iteration history
        if not self.iterations:
            logger.warning("get_optimal_plan: Solution not yet found, returning empty dictionary.")
            return {}

        # Take data from the last saved iteration (it should be optimal)
        last_iter = self.iterations[-1]
        basis = last_iter['basis']
        tableau = last_iter['tableau']

        # Create a dictionary initialized with zeros for all variables x
        plan = {name: 0.0 for name in self.var_names}

        # Iterate over basis variables and their values
        for i, basis_idx in enumerate(basis):
            # If the basis variable is one of the original variables x
            if basis_idx < self.n:
                # Assign the value from the B column
                plan[self.var_names[basis_idx]] = tableau[i, 0]

        return plan

    def get_objective_value(self) -> float:
        """
        Returns the value of the objective function in the optimal plan.

        Returns:
            float: Value of the objective function F(X).
            If the solution is not yet found, returns 0.0.
        """
        # Check if there is iteration history
        if not self.iterations:
            logger.warning("get_objective_value: Solution not yet found, returning 0.0.")
            return 0.0
        # Take the value of F(X) from the last saved tableau
        return self.iterations[-1]['tableau'][self.m, 0]

    def solve_with_modified_b(self, b_new: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Solves the problem with a modified vector b (right-hand sides of constraints).
        Used for scenario analysis.

        Args:
            b_new (np.ndarray): New vector of right-hand sides of constraints (dimension m).

        Returns:
            Tuple[bool, np.ndarray, float]:
            - success: True if an optimal plan is found, False if the problem is unsolvable.
            - solution: Vector of the optimal solution (dimension n).
            - optimal_value: Value of the objective function at the optimal point.
        """
        # Check the dimension of the new vector
        if len(b_new) != self.m:
            raise ValueError(f'Dimension of b_new ({len(b_new)}) does not match the number of constraints ({self.m})')

        # Create a *new* instance of the solver with the same A, c, names, but new b
        # This is necessary because the simplex method algorithm changes the internal state of the object
        solver_copy = SimplexSolver(self.c, self.A, b_new, self.var_names, self.constraint_names)

        # Call the solve method for the new instance
        return solver_copy.solve()
