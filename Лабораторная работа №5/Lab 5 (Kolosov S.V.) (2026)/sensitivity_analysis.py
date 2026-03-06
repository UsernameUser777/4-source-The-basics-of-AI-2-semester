# sensitivity_analysis.py
"""
Модуль для анализа устойчивости (чувствительности) оптимального решения задачи ЛП.
Реализует расчёт теневых цен, допустимых изменений ресурсов и минимальных цен для грузов.
Автор: Колосов Станислав
Дата: 2026
"""

import numpy as np
from typing import Dict, List, Tuple
# --- ИМПОРТ ДЛЯ ЛОГИРОВАНИЯ ---
import logging
# --------------------------

from simplex_solver import SimplexSolver

# --- СОЗДАНИЕ ЛОГГЕРА ДЛЯ ЭТОГО МОДУЛЯ ---
logger = logging.getLogger(__name__)
# ---------------------------------------

class SensitivityAnalyzer:
    """
    Класс для проведения анализа устойчивости оптимального решения задачи ЛП.

    Анализ включает:
    1. Расчёт теневых цен (двойственных оценок) для каждого ограничения
    2. Определение допустимых интервалов изменения ресурсов без изменения базиса
    3. Расчёт минимально допустимой цены для невыгодных грузов
    4. Анализ влияния изменения наличия грузов на общую прибыль
    """

    def __init__(self, solver: SimplexSolver):
        """
        Инициализация анализатора устойчивости.

        Аргументы:
            solver (SimplexSolver): Экземпляр решателя с найденным оптимальным решением.
                                    Требуется, чтобы solver.solve() был вызван успешно.
        """
        self.solver = solver
        # Словари для хранения результатов анализа
        self.shadow_prices: Dict[str, float] = {} # {имя_ограничения: теневая_цена}
        self.allowable_increase: Dict[str, float] = {} # {имя_ограничения: max_увеличение}
        self.allowable_decrease: Dict[str, float] = {} # {имя_ограничения: max_уменьшение}
        # Список для хранения информации о невыгодных грузах
        self.unprofitable_cargos: List[Tuple[str, float, float]] = []  # (груз, текущая_цена, мин_цена)

        # Проверка, что решение найдено (есть история итераций)
        if not self.solver.iterations:
            raise ValueError("Решение задачи не найдено. Сначала выполните метод solve().")


    def calculate_shadow_prices(self) -> Dict[str, float]:
        """
        Расчёт теневых цен (двойственных оценок) для каждого ограничения.

        Теневая цена показывает, насколько изменится целевая функция при увеличении
        правой части ограничения на единицу (при условии сохранения оптимального базиса).

        Для задачи максимизации теневые цены находятся в индексной строке F(X)
        в позициях, соответствующих дополнительным переменным (столбцы после n).
        Поскольку в строке F(X) коэффициенты при дополнительных переменных уже
        учитывают знак (они отрицательны для максимизации), то истинная теневая цена
        равна -(коэффициент из строки F(X)).

        Возвращает:
            Dict[str, float]: Словарь {имя_ограничения: теневая_цена}.
        """
        # Получаем финальную таблицу из последней итерации
        last_tableau = self.solver.iterations[-1]['tableau']
        m = self.solver.m # Количество ограничений

        # Извлекаем коэффициенты из строки F(X) для дополнительных переменных (столбцы n+1 до n+m)
        # Эти коэффициенты обозначим как r_s (reduced costs для s)
        # Теневая цена для ограничения i равна -r_s[i]
        # Индексы столбцов дополнительных переменных в tableau: от n+1 до n+m
        # Извлекаем срез [n+1 : n+m+1]
        reduced_costs_slack = last_tableau[m, self.solver.n + 1:self.solver.n + m + 1]
        # Вычисляем теневые цены
        shadow_prices_raw = -reduced_costs_slack

        # Формируем словарь, сопоставляя имена ограничений и вычисленные теневые цены
        self.shadow_prices = {
            self.solver.constraint_names[i]: shadow_prices_raw[i]
            for i in range(m)
        }

        return self.shadow_prices


    def calculate_allowable_changes(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Расчёт допустимых изменений правых частей ограничений без изменения базиса.

        Этот анализ показывает, насколько можно увеличить или уменьшить ресурс (b_i),
        чтобы текущий оптимальный базис оставался допустимым и оптимальным.
        Используется двойственный симплекс-метод или анализ чувствительности.

        Алгоритм:
        1. Рассматриваем финальную симплекс-таблицу.
        2. Для каждого ограничения (строки i) и соответствующего столбца дополнительной переменной s_i,
           анализируем, как изменение b_i влияет на значения переменных в базисе.
        3. Используем формулу: Delta_b_i_max = min(B^{-1}e_i)_j / |a_jk| для a_jk < 0 (увеличение)
                               Delta_b_i_min = min(B^{-1}e_i)_j / |a_jk| для a_jk > 0 (уменьшение)
           где B - базисная матрица, e_i - единичный вектор.
           На практике это сводится к анализу коэффициентов в столбце s_i финальной таблицы.

        Возвращает:
            Tuple[Dict, Dict]:
                - допустимое увеличение для каждого ограничения
                - допустимое уменьшение для каждого ограничения
        """
        # Получаем финальную таблицу и базис
        last_tableau = self.solver.iterations[-1]['tableau']
        m = self.solver.m
        n = self.solver.n

        # Очищаем старые данные
        self.allowable_increase = {}
        self.allowable_decrease = {}

        # Перебираем все ограничения (каждое связано с одной дополнительной переменной s_i)
        for i in range(m):
            constraint_name = self.solver.constraint_names[i]
            # b_i - текущее значение правой части
            b_i = last_tableau[i, 0]

            # Индекс столбца дополнительной переменной s_i в таблице: n + 1 + i
            # (n+1 - начало столбцов s, i - смещение)
            slack_col = n + 1 + i

            # Извлекаем коэффициенты из *всех* строк (включая F(X)) для этого столбца
            col_coeffs = last_tableau[:, slack_col]

            # --- Расчёт допустимого увеличения (Delta b_i+) ---
            # Определяет, насколько можно *увеличить* b_i, чтобы b_i >= 0
            # Это зависит от того, как изменится b_i в результате изменения b_j в системе B*x = b_new
            # В терминах симплекс-таблицы: b_new_i = b_i + sum(T_jk * delta_b_k) для k
            # где T_jk - элемент таблицы в строке j, столбце k (для дополнительных переменных k).
            # Мы рассматриваем изменение только одного b_i.
            # b_new_i = b_i + T_ii * delta_b_i
            # Чтобы b_new_i >= 0, нужно delta_b_i >= -b_i / T_ii (если T_ii > 0)
            # delta_b_i <= -b_i / T_ii (если T_ii < 0)
            # Однако, для анализа изменения *одного* b_i на величину delta_b_i,
            # нужно посмотреть, как это изменение влияет на *все* b_j в базисе.
            # b_new = b_old + delta_b_i * B^{-1} * e_i
            # где e_i - единичный вектор. В симплекс-таблице столбец B^{-1} * e_i - это столбец s_i.
            # Новое значение b_j = b_old_j + delta_b_i * T_ji
            # Условие: b_new_j >= 0 для всех j
            # b_old_j + delta_b_i * T_ji >= 0
            # delta_b_i * T_ji >= -b_old_j
            # Если T_ji > 0: delta_b_i >= -b_old_j / T_ji
            # Если T_ji < 0: delta_b_i <= -b_old_j / T_ji
            # Нас интересует max delta_b_i (увеличение) и min delta_b_i (уменьшение).
            # max delta_b_i = min_j (-b_old_j / T_ji) for j where T_ji < 0
            # min delta_b_i = max_j (-b_old_j / T_ji) for j where T_ji > 0
            # Т.е. max_delta = min( (-b_j / T_ji) for j where T_ji < 0 )
            #      min_delta = max( (-b_j / T_ji) for j where T_ji > 0 )
            # allowable_increase = max_delta - current_b_i
            # allowable_decrease = current_b_i - min_delta

            # Упрощённый подход, часто используемый в анализе чувствительности:
            # allowable_increase_i = min_j ( b_j / |T_ji| ) for j where T_ji < 0
            # allowable_decrease_i = min_j ( b_j / |T_ji| ) for j where T_ji > 0
            # Это приближение, но часто даёт разумные оценки.
            # Правильный способ: решить систему неравенств b_old + delta_b_i * col_coeffs >= 0
            # относительно delta_b_i.

            # Реализуем упрощённый подход, как в многих учебных примерах
            increases = []
            decreases = []

            # Перебираем строки базисных переменных (не включая F(X))
            for j in range(m):
                coeff = col_coeffs[j] # T_ji - коэффициент в строке j, столбце s_i
                if coeff < -1e-10: # a_ji < 0 (с учётом погрешности)
                    # delta_b_i <= b_j / |coeff| = b_j / (-coeff)
                    # Это потенциальное max увеличение
                    potential_inc = b_j / (-coeff) if (b_j := last_tableau[j, 0]) >= 0 else float('inf')
                    if potential_inc >= 0: # Только если результат имеет смысл
                        increases.append(potential_inc)
                elif coeff > 1e-10: # a_ji > 0
                    # delta_b_i >= - b_j / coeff
                    # Это потенциальное max уменьшение (в отрицательную сторону)
                    # allowable_decrease = min ( b_j / coeff )
                    potential_dec = b_j / coeff if (b_j := last_tableau[j, 0]) >= 0 else float('inf')
                    if potential_dec >= 0:
                        decreases.append(potential_dec)

            # Определяем max увеличение и max уменьшение
            # Если нет ограничений, то изменение неограничено
            self.allowable_increase[constraint_name] = min(increases) if increases else float('inf')
            self.allowable_decrease[constraint_name] = min(decreases) if decreases else float('inf')

        return self.allowable_increase, self.allowable_decrease


    def calculate_min_price_for_unprofitable_cargos(self, cargo_data: List[Dict]) -> List[Tuple[str, float, float]]:
        """
        Расчёт минимально допустимой цены для грузов, которые не включены в оптимальный план.

        Алгоритм:
        1. Определяем грузы с нулевым количеством в оптимальном плане.
        2. Для каждого такого груза вычисляем "снижение" (reduced cost):
           rc = c_j - sum(y_i * a_ij), где:
           - c_j - текущая цена (коэффициент в целевой функции) груза j
           - y_i - теневая цена i-го ограничения
           - a_ij - коэффициент при переменной x_j в i-ом ограничании
        3. Если rc < 0, груз потенциально выгоден. Его минимальная цена для включения
           в план равна c_j_new = c_j - rc (чтобы rc стало >= 0).
           Если rc >= 0, груз не выгоден даже при текущей цене.

        Аргументы:
            cargo_data (List[Dict]): Список данных о грузах с ключами:
                - 'name': название груза
                - 'price': текущая цена (коэффициент c_j)
                - 'weight': вес единицы груза (коэффициент в ограничении по весу)
                - 'volume': объём единицы груза (коэффициент в ограничении по объёму)
                - 'index': индекс груза (1-5), чтобы сопоставить с именами переменных x_ij

        Возвращает:
            List[Tuple[str, float, float]]: Список кортежей (груз, текущая_цена, мин_цена).
                                           Включает только грузы с rc < 0 (потенциально невыгодные).
        """
        # Получаем оптимальный план и теневые цены
        optimal_plan = self.solver.get_optimal_plan()
        if not self.shadow_prices:
            self.calculate_shadow_prices()

        # Определяем, какие грузы не используются (суммарное количество = 0)
        unused_cargo_indices = set()
        for cargo_info in cargo_data:
            cargo_idx = cargo_info['index']
            # Суммируем количество этого груза по всем отсекам
            total_amount = sum(optimal_plan.get(f'x{cargo_idx}{j}', 0.0) for j in range(1, 4))
            if total_amount < 1e-5: # Проверяем на "нуль" с погрешностью
                unused_cargo_indices.add(cargo_idx)

        # Рассчитываем минимальную цену для невыгодных грузов
        self.unprofitable_cargos = []
        for cargo_info in cargo_data:
            cargo_idx = cargo_info['index']
            if cargo_idx not in unused_cargo_indices:
                continue # Пропускаем, если груз используется

            # Получаем текущую цену и характеристики груза
            current_price = cargo_info['price']
            weight = cargo_info['weight']
            volume = cargo_info['volume']

            # Рассчитываем "снижение" (reduced cost) rc = c_j - sum(y_i * a_ij)
            # a_ij - коэффициент при x_ij в ограничениях
            # Для груза i и отсека j:
            # - в ограничении по весу отсека j: a_ij = weight
            # - в ограничении по объёму отсека j: a_ij = volume
            # - в ограничении по наличию груза i: a_ij = 1 (для любого j)
            # rc_j = current_price - [ (sum по j (shadow_price_weight_j * weight)) +
            #                          (sum по j (shadow_price_volume_j * volume)) +
            #                          (shadow_price_availability_i * 1) ]
            # Упрощение: Посчитаем rc для x_i1 (груз i в отсек 1), это будет одинаково для всех x_ij
            # т.к. коэффициенты a_ij для разных j (отсеков) одинаковы для фиксированного i (груза).
            # rc_i = c_i - [ sum_j(shadow_price_weight_j * weight_i) +
            #                sum_j(shadow_price_volume_j * volume_i) +
            #                shadow_price_availability_i * 1 ]
            # rc_i = c_i - [ weight_i * sum_j(shadow_price_weight_j) +
            #                volume_i * sum_j(shadow_price_volume_j) +
            #                shadow_price_availability_i ]

            # Более точный расчёт: суммируем по каждому ограничению
            rc_sum = 0.0
            for j in range(1, 4): # Для каждого отсека (1, 2, 3)
                # Коэффициент в ограничении по весу отсека j
                weight_constraint_name = f'Вес_отсек{j}'
                y_weight = self.shadow_prices.get(weight_constraint_name, 0.0)
                rc_sum += y_weight * weight

                # Коэффициент в ограничении по объёму отсека j
                volume_constraint_name = f'Объем_отсек{j}'
                y_volume = self.shadow_prices.get(volume_constraint_name, 0.0)
                rc_sum += y_volume * volume

            # Коэффициент в ограничении по наличию груза i
            # Имя ограничения: предполагается формат 'Наличие_груз{idx}_{name}'
            # Найдём точное имя по индексу
            availability_constraint_name = next((name for name in self.solver.constraint_names if f'Наличие_груз{cargo_idx}_' in name), None)
            if availability_constraint_name:
                y_availability = self.shadow_prices.get(availability_constraint_name, 0.0)
                rc_sum += y_availability * 1.0 # Коэффициент при x_ij в ограничении по наличию = 1

            # Вычисляем снижение
            reduced_cost = current_price - rc_sum

            # Минимальная цена для рентабельности: c_new = c_old - rc
            # Если rc < 0, то c_new > c_old, что означает, что цену нужно повысить,
            # чтобы компенсировать недостаток прибыли из-за ограничений.
            # Если rc >= 0, груз и так не выгоден.
            min_price = current_price - reduced_cost if reduced_cost < 0 else current_price

            # Добавляем в результат, если груз действительно невыгоден (rc < 0)
            if reduced_cost < 0:
                 self.unprofitable_cargos.append((
                    cargo_info['name'],
                    current_price,
                    max(min_price, 0.0)  # Цена не может быть отрицательной
                ))

        return self.unprofitable_cargos


    def analyze_scenario(self, scenario_changes: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Анализ изменения прибыли при изменении параметров (наличия грузов).
        Это упрощённый сценарный анализ, предполагающий, что структура оптимального
        плана (какие грузы в каких отсеках) не меняется радикально.

        Аргументы:
            scenario_changes (Dict[str, float]): Словарь изменений значений переменных
                в *уже найденном* оптимальном плане. Ключ — имя переменной (например, 'x51'),
                значение — новое количество (ожидается, что оно >= 0 и не нарушает ограничений).

        Возвращает:
            Tuple[float, Dict[str, float]]:
                - прирост (или убыль) прибыли (дельта) при применении сценария.
                - новый "план" с изменёнными значениями (остальные значения как в оптимальном).
        """
        # Получаем текущий оптимальный план
        current_plan = self.solver.get_optimal_plan()
        current_profit = self.solver.get_objective_value()

        # Применяем изменения к плану
        new_plan = current_plan.copy()
        delta_profit = 0.0

        # Перебираем изменения из сценария
        for var_name, new_value in scenario_changes.items():
            if var_name in new_plan:
                old_value = new_plan[var_name]
                delta = new_value - old_value

                # Определяем индекс груза из имени переменной (например, 'x51' -> 5)
                try:
                    cargo_idx_str = var_name[1] # Второй символ (после 'x')
                    # Если имя переменной xIJ, где IJ - двузначный индекс, используем срез
                    # cargo_idx_str = var_name[1] # Пока предполагаем однозначный индекс
                    cargo_idx = int(cargo_idx_str)
                except (ValueError, IndexError):
                    # --- ИСПРАВЛЕНО: используем локальный logger ---
                    logger.warning(f"Не удалось определить индекс груза из переменной '{var_name}'. Пропуск.")
                    # ----------------------------------------
                    continue

                # Находим цену этого груза (коэффициент в целевой функции)
                # Это можно сделать, зная, как строился вектор c в _build_problem_matrices
                # В данном случае, цена зависит только от индекса груза
                # Используем вспомогательный метод или предполагаем структуру
                price = self._get_cargo_price(cargo_idx)

                # Изменение прибыли = (новое_кол-во - старое_кол-во) * цена
                delta_profit += delta * price
                # Обновляем значение в новом плане
                new_plan[var_name] = new_value

        return delta_profit, new_plan


    def _get_cargo_price(self, cargo_index: int) -> float:
        """
        Вспомогательный метод для получения цены груза по его индексу.
        Этот метод нужен, потому что SimplexSolver не хранит исходные цены грузов напрямую.

        Аргументы:
            cargo_index (int): Индекс груза (1-5).

        Возвращает:
            float: Цена единицы груза.
        """
        # Сопоставление индекса груза и его цены (берётся из _build_problem_matrices)
        cargo_prices = {
            1: 8.0,      # Мини-тракторы
            2: 21.5,     # Бумага
            3: 51.0,     # Контейнеры
            4: 275.0,    # Металлопрокат
            5: 110.0     # Пиломатериалы
        }
        return cargo_prices.get(cargo_index, 0.0)

    def generate_stability_report(self) -> str:
        """
        Формирует текстовый отчёт об устойчивости решения.
        Исправлено: защита от ошибки форматирования строк вместо чисел.
        """
        # Рассчитываем все показатели анализа, если ещё не рассчитаны
        if not self.shadow_prices:
            self.calculate_shadow_prices()
        if not self.allowable_increase or not self.allowable_decrease:
            self.calculate_allowable_changes()
        # unprofitable_cargos рассчитываются отдельно, когда нужно

        report = "=" * 80 + "\n"
        report += "ОТЧЁТ ОБ УСТОЙЧИВОСТИ ОПТИМАЛЬНОГО РЕШЕНИЯ (АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ)\n"
        report += "=" * 80 + "\n\n"

        # --- Помощь: функция для безопасного форматирования числа ---
        def safe_float_format(value, fmt=".4f"):
            """Преобразует значение в float и форматирует его, или возвращает 'N/A'."""
            try:
                # Если value уже число, просто форматируем
                if isinstance(value, (int, float)):
                    return f"{value:{fmt}}"
                # Если это строка, пробуем преобразовать в float
                elif isinstance(value, str):
                    return f"{float(value):{fmt}}"
                else:
                    return "N/A"
            except (ValueError, TypeError):
                return "N/A"

        # 1. Теневые цены
        report += "1. ТЕНЕВЫЕ ЦЕНЫ (ДВОЙСТВЕННЫЕ ОЦЕНКИ)\n"
        report += "-" * 80 + "\n"
        report += f"{'Ограничение':<25} | {'Теневая цена':>15} | {'Экономический смысл':<35}\n"
        report += "-" * 80 + "\n"

        for constraint, price in self.shadow_prices.items():
            if 'Вес' in constraint:
                meaning = "Доход от +1 т грузоподъёмности"
            elif 'Объем' in constraint:
                meaning = "Доход от +1 м³ объёма"
            elif 'Наличие' in constraint:
                meaning = "Доход от +1 ед. груза"
            else:
                meaning = "—"

            # Используем безопасный форматтер
            price_str = safe_float_format(price)
            report += f"{constraint:<25} | {price_str:>15} | {meaning:<35}\n"

        report += "\n"

        # 2. Допустимые изменения
        report += "2. ДОПУСТИМЫЕ ИЗМЕНЕНИЯ РЕСУРСОВ БЕЗ ИЗМЕНЕНИЯ БАЗИСА\n"
        report += "-" * 80 + "\n"
        report += f"{'Ограничение':<25} | {'Текущее':>10} | {'+Δ':>10} | {'-Δ':>10} | {'Новый диапазон':<20}\n"
        report += "-" * 80 + "\n"

        # Исходные значения правых частей ограничений (из начальной задачи)
        original_b = self.solver.b

        for i, constraint in enumerate(self.solver.constraint_names):
            current = original_b[i]
            inc = self.allowable_increase.get(constraint, float('inf'))
            dec = self.allowable_decrease.get(constraint, float('inf'))

            # Форматируем все значения через безопасный форматтер
            current_str = safe_float_format(current)
            inc_str = safe_float_format(inc, ".1f") if inc != float('inf') else "∞"
            dec_str = safe_float_format(dec, ".1f") if dec != float('inf') else "∞"
            range_start = safe_float_format(current - dec, ".1f") if dec != float('inf') else "-∞"
            range_end = safe_float_format(current + inc, ".1f") if inc != float('inf') else "+∞"
            range_str = f"[{range_start}; {range_end}]"

            report += f"{constraint:<25} | {current_str:>10} | {inc_str:>10} | {dec_str:>10} | {range_str:<20}\n"

        report += "\n"

        # 3. Невыгодные грузы
        if self.unprofitable_cargos:
            report += "3. НЕВЫГОДНЫЕ ГРУЗЫ И МИНИМАЛЬНО ДОПУСТИМАЯ ЦЕНА\n"
            report += "-" * 80 + "\n"
            report += f"{'Груз':<20} | {'Текущая цена':>15} | {'Мин. цена':>15} | {'Разница':>15}\n"
            report += "-" * 80 + "\n"

            for cargo, current_price, min_price, diff in [(c[0], c[1], c[2], c[2] - c[1]) for c in
                                                          self.unprofitable_cargos]:
                # Используем безопасный форматтер для всех чисел
                curr_str = safe_float_format(current_price)
                min_str = safe_float_format(min_price)
                diff_str = safe_float_format(diff)
                report += f"{cargo:<20} | {curr_str:>15} | {min_str:>15} | {diff_str:>15}\n"

            report += "\nПримечание: Груз станет рентабельным при увеличении цены на величину разницы.\n"
        else:
            report += "3. НЕВЫГОДНЫЕ ГРУЗЫ И МИНИМАЛЬНО ДОПУСТИМАЯ ЦЕНА\n"
            report += "-" * 80 + "\n"
            report += "Все грузы, доступные в заданном количестве, включены в оптимальный план.\n"

        report += "\n" + "=" * 80 + "\n"
        return report
