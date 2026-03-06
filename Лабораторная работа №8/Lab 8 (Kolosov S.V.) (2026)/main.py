# -*- coding: utf-8 -*-
"""Основной исполняемый файл для запуска приложения анализа иерархий."""

import sys
import tkinter as tk
from gui import MCDAApp  # Импортируем класс приложения из gui.py

def main():
    """Основная функция для запуска приложения."""
    root = tk.Tk()
    app = MCDAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
