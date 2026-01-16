"""Strategy grid widget for displaying Power Blackjack strategy."""

import tkinter as tk
from typing import Optional, Tuple

from blackjack_logic.strategy import PowerBlackjackStrategy, StrategyResult, Action


class StrategyGrid(tk.Frame):
    """Compact strategy grid with fixed width."""
    
    PANEL_WIDTH = 240
    
    BG = '#1e1e2e'
    CELL_BG = '#313244'
    HEADER_BG = '#45475a'
    HIGHLIGHT_ROW = '#3d5a80'
    HIGHLIGHT_COL = '#3d5a80'
    HIGHLIGHT_CELL = '#f9e2af'
    TEXT_NORMAL = '#a6adc8'
    TEXT_HEADER = '#cdd6f4'
    
    ACTION_COLORS = {
        'H': '#a6e3a1',
        'S': '#89b4fa',
        'P': '#cba6f7',
        '4': '#f9e2af',
    }

    def __init__(self, parent, width: int = 240):
        super().__init__(parent, bg=self.BG, width=width)
        self.pack_propagate(False)  # Prevent resizing
        self.configure(width=self.PANEL_WIDTH)
        
        self._current_grid = "HARD"
        self._highlight_row = -1
        self._highlight_col = -1
        self._cells = {}
        self._row_headers = []
        self._col_headers = []
        
        self._last_result_key = None
        self._last_event_key = None
        self._showing_event = False
        
        self._create_widgets()

    def _create_widgets(self):
        # Header
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill=tk.X, padx=4, pady=4)
        
        tk.Label(header, text="Strategy", font=('Consolas', 10, 'bold'),
                fg='#cdd6f4', bg=self.BG).pack(side=tk.LEFT)
        
        self._grid_label = tk.Label(header, text="HARD", font=('Consolas', 9),
                                    fg='#f9e2af', bg=self.BG, width=6, anchor='e')
        self._grid_label.pack(side=tk.RIGHT)
        
        # Grid container
        self._grid_frame = tk.Frame(self, bg=self.BG)
        self._grid_frame.pack(fill=tk.BOTH, expand=True, padx=2)
        
        # Event overlay (hidden)
        self._event_frame = tk.Frame(self, bg='#313244')
        self._event_title = tk.Label(self._event_frame, text="", 
                                     font=('Consolas', 12, 'bold'),
                                     fg='#f9e2af', bg='#313244', pady=20,
                                     width=20)
        self._event_title.pack()
        self._event_message = tk.Label(self._event_frame, text="",
                                       font=('Consolas', 9),
                                       fg='#cdd6f4', bg='#313244', pady=8,
                                       wraplength=220, width=25)
        self._event_message.pack()
        
        # Action display
        self._action_frame = tk.Frame(self, bg='#313244')
        self._action_frame.pack(fill=tk.X, padx=4, pady=4)
        
        self._action_label = tk.Label(self._action_frame, text="--", 
                                      font=('Consolas', 16, 'bold'),
                                      fg='#6c7086', bg='#313244', pady=6,
                                      width=10)
        self._action_label.pack()
        
        self._info_label = tk.Label(self._action_frame, text="",
                                    font=('Consolas', 8), fg='#6c7086', bg='#313244',
                                    width=20)
        self._info_label.pack(pady=(0, 4))
        
        self._build_grid("HARD")

    def _build_grid(self, grid_type: str):
        for widget in self._grid_frame.winfo_children():
            widget.destroy()
        self._cells.clear()
        self._row_headers.clear()
        self._col_headers.clear()
        
        if grid_type == "HARD":
            rows = PowerBlackjackStrategy.HARD_ROWS
            grid = PowerBlackjackStrategy.HARD_GRID
        elif grid_type == "SOFT":
            rows = PowerBlackjackStrategy.SOFT_ROWS
            grid = PowerBlackjackStrategy.SOFT_GRID
        else:
            rows = PowerBlackjackStrategy.PAIRS_ROWS
            grid = PowerBlackjackStrategy.PAIRS_GRID
        
        cols = PowerBlackjackStrategy.DEALER_CARDS
        
        container = tk.Frame(self._grid_frame, bg=self.BG)
        container.pack(anchor='center')
        
        # Corner
        tk.Label(container, text="", width=4, bg=self.BG, 
                font=('Consolas', 7)).grid(row=0, column=0)
        
        # Column headers
        for c, col in enumerate(cols):
            lbl = tk.Label(container, text=col, font=('Consolas', 7, 'bold'),
                          fg=self.TEXT_HEADER, bg=self.HEADER_BG, width=2)
            lbl.grid(row=0, column=c+1, padx=1, pady=1)
            self._col_headers.append(lbl)
        
        # Rows
        for r, row_key in enumerate(rows):
            row_lbl = tk.Label(container, text=row_key, font=('Consolas', 7, 'bold'),
                              fg=self.TEXT_HEADER, bg=self.HEADER_BG, width=4, anchor='e')
            row_lbl.grid(row=r+1, column=0, padx=1, pady=1)
            self._row_headers.append(row_lbl)
            
            for c, col in enumerate(cols):
                action = grid[r][c]
                color = self.ACTION_COLORS.get(action, self.TEXT_NORMAL)
                cell = tk.Label(container, text=action, font=('Consolas', 7),
                               fg=color, bg=self.CELL_BG, width=2)
                cell.grid(row=r+1, column=c+1, padx=1, pady=1)
                self._cells[(r, c)] = cell
        
        self._current_grid = grid_type
        self._grid_label.config(text=grid_type)

    def show_special_event(self, title: str, message: str, color: str):
        key = (title, message)
        if self._last_event_key == key and self._showing_event:
            return
        
        self._last_event_key = key
        self._showing_event = True
        
        self._grid_frame.pack_forget()
        self._event_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self._event_title.config(text=title, fg=color)
        self._event_message.config(text=message)
        self._action_label.config(text="⏸️", fg=color)
        self._info_label.config(text="Paused")

    def hide_special_event(self):
        if not self._showing_event:
            return
        
        self._showing_event = False
        self._last_event_key = None
        
        self._event_frame.pack_forget()
        self._grid_frame.pack(fill=tk.BOTH, expand=True, padx=2)

    def update(self, result: Optional[StrategyResult]):
        if self._showing_event:
            return
        
        if result is None:
            new_key = None
        else:
            new_key = (result.grid_type, result.row_index, result.col_index)
        
        if new_key == self._last_result_key:
            return
        
        self._last_result_key = new_key
        
        if result is None:
            self._clear_highlight()
            self._action_label.config(text="--", fg='#6c7086')
            self._info_label.config(text="")
            return
        
        if result.grid_type != self._current_grid:
            self._build_grid(result.grid_type)
        
        self._clear_highlight()
        
        if 0 <= result.row_index < len(self._row_headers):
            self._row_headers[result.row_index].config(bg=self.HIGHLIGHT_ROW)
        
        if 0 <= result.col_index < len(self._col_headers):
            self._col_headers[result.col_index].config(bg=self.HIGHLIGHT_COL)
        
        for (r, c), cell in self._cells.items():
            if r == result.row_index:
                cell.config(bg=self.HIGHLIGHT_ROW)
            elif c == result.col_index:
                cell.config(bg=self.HIGHLIGHT_COL)
        
        key = (result.row_index, result.col_index)
        if key in self._cells:
            self._cells[key].config(bg=self.HIGHLIGHT_CELL, fg='#1e1e2e')
        
        self._highlight_row = result.row_index
        self._highlight_col = result.col_index
        
        text, color = PowerBlackjackStrategy.get_action_display(result.action)
        self._action_label.config(text=text, fg=color)
        self._info_label.config(text=f"{result.player_key} vs {result.dealer_key}")

    def _clear_highlight(self):
        for lbl in self._row_headers:
            lbl.config(bg=self.HEADER_BG)
        for lbl in self._col_headers:
            lbl.config(bg=self.HEADER_BG)
        for (r, c), cell in self._cells.items():
            action = cell.cget('text')
            color = self.ACTION_COLORS.get(action, self.TEXT_NORMAL)
            cell.config(bg=self.CELL_BG, fg=color)
        self._highlight_row = -1
        self._highlight_col = -1
