# simplex_solver.py
"""
Реализация симплекс-метода для решения задачи линейного программирования.
Вариант №1: Оптимизация загрузки грузопассажирского судна.
Автор: Колосов Станислав
Дата: 2026
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# --- Настройка логирования ---
# Используем стандартный уровень INFO.
# Для отладки можно установить level=logging.DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplexSolver:
    """
    Класс для решения задачи линейного программирования симплекс-методом.

    Поддерживает:
    - Решение задачи максимизации прибыли от перевозки грузов
    - Вывод всех итераций симплекс-таблиц
    - Определение оптимального плана распределения грузов по отсекам
    - Анализ вырожденности и зацикливания
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                 var_names: List[str], constraint_names: List[str]):
        """
        Инициализация симплекс-решателя.

        Аргументы:
            c (np.ndarray): Вектор коэффициентов целевой функции (размерность n).
                            Для задачи максимизации, каждый элемент c[j] представляет
                            коэффициент при переменной x_j в целевой функции F = sum(c[j] * x[j]).
            A (np.ndarray): Матрица коэффициентов левых частей ограничений (размерность m x n).
                            Каждый элемент A[i][j] - это коэффициент при переменной x_j
                            в i-ом ограничении.
            b (np.ndarray): Вектор правых частей ограничений (размерность m).
                            Каждый элемент b[i] - это правая часть i-го ограничения.
                            Все элементы b[i] должны быть >= 0.
            var_names (List[str]): Имена переменных решения (например, ['x11', 'x12', ...]).
                                   Должны быть уникальными и соответствовать порядку в векторе c.
            constraint_names (List[str]): Имена ограничений (например, ['Вес_отсек1', 'Объем_отсек1', ...]).
                                          Должны быть уникальными и соответствовать порядку в векторе b.

        Примечание:
            Задача предполагается в канонической форме:
            max c^T * x
            при условиях A * x <= b, x >= 0.
            Решатель автоматически приводит её к стандартной форме, добавляя
            дополнительные переменные s_i (s >= 0), превращая неравенства в равенства:
            A * x + s = b.
            Таким образом, общее количество переменных в симплекс-таблице будет n + m.
        """
        # --- Проверка корректности входных данных ---
        # Проверяем, что количество коэффициентов целевой функции равно числу столбцов A
        if len(c) != A.shape[1]:
            raise ValueError(
                f"Несоответствие размерностей: длина вектора c ({len(c)}) != количество столбцов A ({A.shape[1]})"
            )
        # Проверяем, что количество правых частей равно числу строк A
        if len(b) != A.shape[0]:
            raise ValueError(
                f"Несоответствие размерностей: длина вектора b ({len(b)}) != количество строк A ({A.shape[0]})"
            )
        # Проверяем, что все элементы b неотрицательны (требование канонической формы)
        if any(bi < 0 for bi in b):
            raise ValueError("Правые части ограничений (b) не могут быть отрицательными.")

        # Сохраняем входные данные как атрибуты объекта, преобразуя в float
        self.c = c.astype(float)
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.var_names = var_names
        self.constraint_names = constraint_names

        # Размерности задачи
        self.m, self.n = A.shape  # m - количество ограничений, n - количество переменных решения x

        # --- Инициализация симплекс-таблицы ---
        # Структура таблицы:
        #   | Базис | B  | x1 | x2 | ... | xn | s1 | s2 | ... | sm |
        #   |-------|----|----|----|-----|----|----|----|-----|----|
        #   | s1    | b1 | a11| a12| ... | a1n| 1  | 0  | ... | 0  |
        #   | s2    | b2 | a21| a22| ... | a2n| 0  | 1  | ... | 0  |
        #   | ...   | .. | .. | .. | ... | .. | .. | .. | ... | .. |
        #   | sm    | bm | am1| am2| ... | amn| 0  | 0  | ... | 1  |
        #   | F(X)  | 0  | -c1| -c2| ... | -cn| 0  | 0  | ... | 0  |

        # Создаём нулевую таблицу размером (m+1) x (n+m+1)
        # +1 для строки целевой функции, +1 для столбца B
        self.tableau = np.zeros((self.m + 1, self.n + self.m + 1))

        # Заполняем столбец свободных членов (B)
        self.tableau[:self.m, 0] = self.b

        # Заполняем коэффициенты переменных решения x
        self.tableau[:self.m, 1:self.n + 1] = self.A

        # Заполняем единичную матрицу для дополнительных переменных s (слева от диагонали)
        self.tableau[:self.m, self.n + 1:self.n + self.m + 1] = np.eye(self.m)

        # Заполняем строку целевой функции F(X) с противоположным знаком (-c)
        # Это потому, что мы решаем задачу максимизации, а симплекс-метод обычно реализуется для минимизации.
        # max F = - min(-F), поэтому коэффициенты становятся -c_j
        self.tableau[self.m, 1:self.n + 1] = -self.c
        # Коэффициенты при дополнительных переменных в строке F(X) равны 0

        # Имена всех переменных (основные x + дополнительные s)
        self.all_var_names = var_names + [f's{i + 1}' for i in range(self.m)]

        # Текущий базис (индексы переменных, находящихся в базисе).
        # Изначально базисными являются дополнительные переменные s1..sm.
        # Их индексы в расширенной таблице: n (для s1), n+1 (для s2), ..., n+m-1 (для sm)
        self.basis = list(range(self.n, self.n + self.m))

        # История итераций для отчёта
        self.iterations: List[Dict] = []

        # Логируем начало работы
        logger.info(f"Инициализирован симплекс-решатель: {self.n} переменных, {self.m} ограничений")
        logger.info(f"Целевая функция: max F = {' + '.join([f'{c_i:.1f}*{name}' for c_i, name in zip(c, var_names)])}")

        # Сохраняем начальное состояние (итерация 0)
        self._save_iteration(0)


    def solve(self, max_iterations: int = 100) -> Tuple[bool, np.ndarray, float]:
        """
        Основной метод решения задачи симплекс-методом.

        Аргументы:
            max_iterations (int): Максимальное количество итераций для предотвращения зацикливания.

        Возвращает:
            Tuple[bool, np.ndarray, float]:
                - success: True если найден оптимальный план, False если задача неразрешима или превышен лимит итераций.
                - solution: Вектор оптимального решения (размерность n, только для переменных x).
                - optimal_value: Значение целевой функции в оптимальной точке.
        """
        iteration = 0
        logger.info("Начало решения симплекс-методом...")

        # Основной цикл симплекс-метода
        while iteration < max_iterations:
            # --- Шаг 1: Проверка оптимальности ---
            # Извлекаем индексную строку (строку целевой функции F(X))
            # Она находится в последней строке таблицы (self.m)
            # Рассматриваем коэффициенты при всех переменных (x и s)
            index_row = self.tableau[self.m, 1:self.n + self.m + 1]

            # Для задачи максимизации оптимальный план достигнут, когда все
            # коэффициенты в индексной строке <= 0 (с учётом погрешности вычислений).
            # Проверяем, есть ли среди них положительные.
            positive_coeffs_mask = index_row > 1e-10  # Порог для численной погрешности

            # Если нет положительных коэффициентов, план оптимален
            if not np.any(positive_coeffs_mask):
                logger.info(f"Оптимальный план найден на итерации {iteration}")
                # Сохраняем финальное состояние, которое является оптимальным
                self._save_iteration(iteration)
                break # Выходим из цикла

            # --- Шаг 2: Выбор ведущего столбца (вводимая переменная) ---
            # Выбираем переменную, которая будет введена в базис.
            # Обычно выбирается переменная с наибольшим положительным коэффициентом
            # в индексной строке (правило наибольшего коэффициента).
            pivot_col_idx = np.argmax(index_row) # Индекс столбца (без учёта столбца B)
            pivot_col = pivot_col_idx + 1 # Индекс столбца в таблице (учитывая столбец B)

            # --- Шаг 3: Выбор ведущей строки (выводимая переменная) ---
            # Выбираем переменную, которая будет выведена из базиса.
            # Вычисляем симплексные отношения (bi / aij) для положительных элементов aij
            # ведущего столбца. Выбираем строку с минимальным положительным отношением.
            ratios = []
            for i in range(self.m): # Перебираем строки ограничений
                if self.tableau[i, pivot_col] > 1e-10: # Только если элемент > 0
                    ratio = self.tableau[i, 0] / self.tableau[i, pivot_col] # bi / aij
                    ratios.append((ratio, i)) # Сохраняем отношение и индекс строки
                else:
                    ratios.append((float('inf'), i)) # Если aij <= 0, отношение inf

            # Если все отношения равны inf, задача неограничена
            if all(ratio == float('inf') for ratio, _ in ratios):
                logger.error("Задача неограничена (целевая функция может быть увеличена до бесконечности)")
                unbounded_var = self.all_var_names[pivot_col_idx]
                logger.error(f"Неограниченная переменная: {unbounded_var}")
                return False, np.zeros(self.n), float('inf')

            # Находим строку с минимальным положительным отношением
            pivot_row = min(ratios, key=lambda x: x[0])[1] # Берём индекс строки
            pivot_element = self.tableau[pivot_row, pivot_col] # Получаем разрешающий элемент

            # Проверка на вырожденность (если bi = 0, это может привести к зацикливанию)
            if abs(self.tableau[pivot_row, 0]) < 1e-10:
                logger.warning(f"Обнаружена вырожденность на итерации {iteration} (b_{pivot_row + 1} = 0)")

            # Сохраняем информацию о текущей итерации до преобразования таблицы
            entering_var = self.all_var_names[pivot_col_idx]
            leaving_var = self.all_var_names[self.basis[pivot_row]]
            logger.info(f"Итерация {iteration + 1}: вводим {entering_var}, выводим {leaving_var}, "
                       f"разрешающий элемент = {pivot_element:.4f} (строка {pivot_row + 1}, столбец {pivot_col_idx + 1})")

            # --- Шаг 4: Преобразование таблицы (метод Жордана-Гаусса) ---
            # Нормируем ведущую строку, разделив на разрешающий элемент
            self.tableau[pivot_row, :] = self.tableau[pivot_row, :] / pivot_element

            # Обнуляем остальные элементы ведущего столбца
            for i in range(self.m + 1): # Перебираем все строки (включая F(X))
                if i != pivot_row: # Пропускаем ведущую строку
                    factor = self.tableau[i, pivot_col] # Текущий элемент ведущего столбца
                    # Вычитаем ведущую строку, умноженную на фактор, из текущей строки
                    self.tableau[i, :] -= factor * self.tableau[pivot_row, :]

            # Обновляем базис: заменяем переменную, вышедшую из базиса, на введенную
            self.basis[pivot_row] = pivot_col_idx

            # Сохраняем результаты текущей итерации (после преобразования)
            self._save_iteration(iteration, pivot_row, pivot_col, pivot_element, entering_var, leaving_var)

            # Увеличиваем счётчик итерации
            iteration += 1

        # Проверка на превышение лимита итераций (возможное зацикливание)
        if iteration >= max_iterations:
            logger.warning(f"Достигнут лимит итераций ({max_iterations}). Возможна вырожденность или зацикливание задачи.")
            # Даже если лимит превышен, последняя сохранённая таблица может быть полезной
            # Но логично считать, что решение не найдено успешно.
            return False, np.zeros(self.n), 0.0

        # --- Формирование результата ---
        # После завершения цикла (если не было превышения лимита), таблица оптимальна
        # Формируем вектор решения только для переменных x (размерности n)
        solution = np.zeros(self.n)
        for i, basis_idx in enumerate(self.basis):
            # Если базисная переменная - это одна из исходных переменных x (а не s)
            if basis_idx < self.n:
                # Её значение равно элементу в столбце B соответствующей строки
                solution[basis_idx] = self.tableau[i, 0]

        # Значение целевой функции находится в правом нижнем углу таблицы (F(X), B)
        optimal_value = self.tableau[self.m, 0]

        # Логируем результат
        logger.info(f"Решение завершено. Оптимальное значение F = {optimal_value:.2f}")
        logger.info(f"Оптимальный план: {dict(zip(self.var_names, solution.round(2)))}")

        # Возвращаем успех, найденное решение и значение функции
        return True, solution, optimal_value


    def _save_iteration(self, iteration_num: int, pivot_row: Optional[int] = None,
                       pivot_col: Optional[int] = None, pivot_element: Optional[float] = None,
                       entering_var: Optional[str] = None, leaving_var: Optional[str] = None):
        """
        Сохраняет текущее состояние симплекс-таблицы в историю итераций.

        Аргументы:
            iteration_num (int): Номер итерации.
            pivot_row (int, optional): Индекс ведущей строки.
            pivot_col (int, optional): Индекс ведущего столбца (с учётом столбца B).
            pivot_element (float, optional): Значение разрешающего элемента.
            entering_var (str, optional): Имя переменной, вводимой в базис.
            leaving_var (str, optional): Имя переменной, выводимой из базиса.
        """
        # Создаём словарь с данными текущей итерации
        iteration_data = {
            'iteration': iteration_num,
            'tableau': self.tableau.copy(), # ВАЖНО: делаем копию таблицы
            'basis': self.basis.copy(),     # ВАЖНО: делаем копию базиса
            'pivot_row': pivot_row,
            'pivot_col': pivot_col,
            'pivot_element': pivot_element,
            'entering_var': entering_var,
            'leaving_var': leaving_var
        }
        # Добавляем словарь в список историй
        self.iterations.append(iteration_data)


    def get_iteration_table(self, iteration_num: int) -> str:
        """
        Формирует текстовое представление симплекс-таблицы для указанной итерации.

        Аргументы:
            iteration_num (int): Номер итерации (0 — начальная таблица).

        Возвращает:
            str: Форматированная строка с симплекс-таблицей.
        """
        # Проверяем, существует ли запрашиваемая итерация
        if iteration_num >= len(self.iterations):
            return f"Итерация {iteration_num} не найдена. Доступно итераций: {len(self.iterations)}"

        # Получаем данные конкретной итерации
        iter_data = self.iterations[iteration_num]
        tableau = iter_data['tableau']
        basis = iter_data['basis']

        # Формируем заголовок таблицы
        header = f"{'Базис':<12} | {'B':>10} | "
        for name in self.all_var_names:
            header += f"{name:>10} | "
        header += "\n" + "-" * (12 + 13 + (13 * len(self.all_var_names)))

        # Формируем строки таблицы (ограничения)
        rows = []
        for i in range(self.m):
            # Имя базисной переменной для строки i
            basis_var = self.all_var_names[basis[i]]
            # Начинаем формировать строку с базисной переменной и значением B
            row_str = f"{basis_var:<12} | {tableau[i, 0]:>10.4f} | "
            # Добавляем коэффициенты для всех переменных (x и s)
            for j in range(1, self.n + self.m + 1):
                row_str += f"{tableau[i, j]:>10.4f} | "
            rows.append(row_str)

        # Формируем строку целевой функции (F(X))
        f_row = f"{'F(X)':<12} | {tableau[self.m, 0]:>10.4f} | "
        for j in range(1, self.n + self.m + 1):
            f_row += f"{tableau[self.m, j]:>10.4f} | "

        # Формируем информацию о ведущем элементе (если она есть для данной итерации)
        pivot_info = ""
        if iter_data['pivot_row'] is not None:
            pivot_info = (f"\nВедущий элемент: {iter_data['pivot_element']:.4f} "
                         f"(строка {iter_data['pivot_row'] + 1}, столбец {iter_data['pivot_col']})\n"
                         f"Вводим: {iter_data['entering_var']}, Выводим: {iter_data['leaving_var']}")

        # Собираем финальную строку
        result = f"\nИтерация №{iteration_num}{pivot_info}\n{header}\n"
        result += "\n".join(rows) + "\n" + f_row

        return result


    def get_optimal_plan(self) -> Dict[str, float]:
        """
        Возвращает оптимальный план в виде словаря {имя_переменной: значение}.

        Возвращает:
            Dict[str, float]: Словарь с оптимальными значениями переменных x.
                              Если решение ещё не найдено, возвращает пустой словарь.
        """
        # Проверяем, есть ли история итераций
        if not self.iterations:
            logger.warning("get_optimal_plan: Решение ещё не найдено, возвращается пустой словарь.")
            return {}

        # Берём данные последней сохранённой итерации (она должна быть оптимальной)
        last_iter = self.iterations[-1]
        basis = last_iter['basis']
        tableau = last_iter['tableau']

        # Создаём словарь, инициализированный нулями для всех переменных x
        plan = {name: 0.0 for name in self.var_names}

        # Перебираем базисные переменные и их значения
        for i, basis_idx in enumerate(basis):
            # Если базисная переменная - это одна из исходных переменных x
            if basis_idx < self.n:
                # Присваиваем значение из столбца B
                plan[self.var_names[basis_idx]] = tableau[i, 0]

        return plan


    def get_objective_value(self) -> float:
        """
        Возвращает значение целевой функции в оптимальном плане.

        Возвращает:
            float: Значение целевой функции F(X).
                   Если решение ещё не найдено, возвращает 0.0.
        """
        # Проверяем, есть ли история итераций
        if not self.iterations:
            logger.warning("get_objective_value: Решение ещё не найдено, возвращается 0.0.")
            return 0.0
        # Берём значение F(X) из последней сохранённой таблицы
        return self.iterations[-1]['tableau'][self.m, 0]


    def solve_with_modified_b(self, b_new: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Решает задачу с изменённым вектором b (правыми частями ограничений).
        Используется для сценарного анализа.

        Аргументы:
            b_new (np.ndarray): Новый вектор правых частей ограничений (размерность m).

        Возвращает:
            Tuple[bool, np.ndarray, float]:
                - success: True если найден оптимальный план, False если задача неразрешима.
                - solution: Вектор оптимального решения (размерность n).
                - optimal_value: Значение целевой функции в оптимальной точке.
        """
        # Проверяем размерность нового вектора
        if len(b_new) != self.m:
            raise ValueError(f"Размерность b_new ({len(b_new)}) не соответствует количеству ограничений ({self.m})")

        # Создаём *новый* экземпляр решателя с теми же A, c, именами, но новым b
        # Это необходимо, так как алгоритм симплекс-метода изменяет внутреннее состояние объекта
        solver_copy = SimplexSolver(self.c, self.A, b_new, self.var_names, self.constraint_names)
        # Вызываем метод solve для нового экземпляра
        return solver_copy.solve()
