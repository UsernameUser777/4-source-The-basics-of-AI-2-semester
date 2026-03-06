LABORATORY WORK #5 ON THE COURSE "FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE"
===============================================================

Topic: Solving a direct linear programming problem using the simplex method

Variant #1: Optimization of cargo loading for a cargo-passenger ship
---

Objective:
---
To develop software in Python for solving a linear programming (LP) problem using the simplex method, applied to the problem of optimizing the loading of a cargo-passenger ship to maximize profit. The program should not only find the optimal cargo distribution plan but also perform stability (sensitivity) analysis of the solution, including calculation of shadow prices, allowable changes in resources, and minimum allowable prices for unprofitable cargos, as well as analyze various scenarios of changing problem parameters.

Problem Description (Variant #1):

- The ship has 3 cargo compartments:
  - Compartment 1: volume 500 m³, max. load capacity 700 t
  - Compartment 2: volume 1000 m³, max. load capacity 800 t
  - Compartment 3: volume 1500 m³, max. load capacity 1300 t
- 5 types of cargos are transported:
  - Cargo 1: Mini-tractors (0.35 t, 3 m³, 8 monetary units/unit, 100 units available)
  - Cargo 2: Paper (1.6 t, 1 m³, 21.5 monetary units/unit, 1000 units available)
  - Cargo 3: Containers (5 t, 6.5 m³, 51 monetary units/unit, 200 units available)
  - Cargo 4: Rolled metal (35 t, 6 m³, 275 monetary units/unit, 200 units available)
  - Cargo 5: Timber (4 t, 6 m³, 110 monetary units/unit, 350 units available)

Objective function:
$$ F = 8 \times (x11 + x12 + x13) + 21.5 \times (x21 + x22 + x23) + 51 \times (x31 + x32 + x33) + 275 \times (x41 + x42 + x43) + 110 \times (x51 + x52 + x53) \rightarrow \max $$
where xij — quantity of cargo type i in compartment j.

Constraints:
- By weight of each compartment.
- By volume of each compartment.
- By availability of each cargo type.
- Non-negativity of variables.

Project Composition:
- main.py — main application with graphical interface (tkinter)
- simplex_solver.py — core of the simplex method
- sensitivity_analysis.py — stability analysis of the solution
- requirements.txt — list of dependencies
- README.txt — this file

Technologies:
- Python 3.13
- tkinter (graphical interface)
- numpy (numerical computations)
- matplotlib (visualization)

Installing Dependencies:

pip install -r requirements.txt

Running the Program:

python main.py

Implementation Features:

- Full support for UTF-8 encoding for correct work with Cyrillic (output of reports, interface, exported files).
(Note: Initially, Windows-1251 was planned, but the code was adapted for UTF-8 for better compatibility and modern standards).
- Detailed output of all iterations of the simplex tableau with pivot elements.
- Automatic calculation of shadow prices (dual estimates) and allowable changes in resources (upper and lower bounds).
- Identification of unprofitable cargos (with zero quantity in the optimal plan) and calculation of the minimum allowable price for their profitability.
- Scenario analysis (change in profit with changes in cargo availability):
  - Timber: 400 units (instead of 350 units)
  - Paper: 900 units (instead of 1000 units)
  - Containers: 100 units (instead of 200 units)
- Analysis is performed both approximately (via shadow prices) and precisely (by re-solving the problem with modified constraints).
- Visualization of results in the form of a bar chart of cargo distribution by types.
- Ability to save plots to a file (PNG).
- Progress bar for tracking the status of long operations (implemented in the interface).
- Export of results to CSV and JSON formats with a detailed data structure and UTF-8 support.
- Detailed comments for all functions and methods in Russian.
- 7 interface tabs: Input Data, Solution (Built-in), Stability Analysis (Built-in), Scenarios (Built-in), Final Results (Built-in), Plots (Built-in), Manual Input.
- Built-in validator for input data in "Manual Input" mode to prevent errors.
- "Clear Results" function to reset the program state.
- Logging of events and errors to the file app_log.log.
- Fixed syntax errors and formatting errors that caused crashes.
- Fixed logic for saving simplex tableau iterations for correct display of the optimal solution.
- Fixed error in the method calculate_min_price_for_unprofitable_cargos in sensitivity_analysis.py.
- Fixed error in the method generate_stability_report related to formatting strings instead of numbers.
- Fixed NameError for 'logger' in the method analyze_scenario in sensitivity_analysis.py.

Manual Input Function:

- The user can enter their own parameters for compartments (volume, weight) and cargos (name, weight, volume, price, availability).
- The program solves the optimization problem for the entered parameters.
- Results of manual calculation can be compared with the built-in variant.
- Stability analysis, scenarios, plots, and export functions are available for both modes.

Conclusions:

1. The program successfully solves the linear programming problem using the simplex method.
2. The optimal ship loading plan that maximizes profit has been found.
3. Stability analysis of the solution has been performed, confirming its reliability within the specified limits of resource changes.
4. Shadow prices have been established, showing the economic value of each resource.
5. Allowable ranges for changing resources without losing the optimality of the basis have been determined.
6. Potentially unprofitable cargos have been identified, and the minimum prices for their inclusion in the optimal plan have been calculated.
7. Scenario analysis has been performed, demonstrating the sensitivity of profit to changes in problem parameters (cargo availability). Comparing the approximate and exact methods showed their differences, which emphasizes the importance of precise recalculation for significant changes.
8. Visualization of results (plots) improves the perception and understanding of the optimal plan.
9. Manual input mode allows analyzing arbitrary loading problems.
10. The program meets all the requirements of the methodological guidelines and can be used to analyze similar optimization problems.
11. The code is structured, well-documented, and corrected for errors, ensuring reliable and stable operation.

Author: Stanislav Kolosov
Group: IVT-3
Year: 2026
