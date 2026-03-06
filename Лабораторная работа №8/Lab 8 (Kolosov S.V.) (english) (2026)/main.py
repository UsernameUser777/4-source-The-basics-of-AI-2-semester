# -*- coding: utf-8 -*-
"""Main executable file to run the Hierarchy Analysis application"""

import sys
import tkinter as tk
from gui import MCDAApp  # Import the application class from gui.py

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = MCDAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
