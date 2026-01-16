"""Information panel widget for displaying blackjack state."""

import tkinter as tk
from typing import List


class InfoPanel(tk.Frame):
    """Compact side panel with fixed width elements."""

    # Fixed width to prevent resizing
    PANEL_WIDTH = 220

    def __init__(self, parent, width: int = 220):
        super().__init__(parent, bg='#1e1e2e', width=width)
        self.pack_propagate(False)  # Prevent resizing
        self.configure(width=self.PANEL_WIDTH)
        self._create_widgets()

    def _create_widgets(self):
        # Title row with detections
        header = tk.Frame(self, bg='#1e1e2e', width=self.PANEL_WIDTH)
        header.pack(fill=tk.X, padx=6, pady=(6, 4))
        
        tk.Label(header, text="♠ Analyzer", font=('Consolas', 10, 'bold'),
                fg='#cdd6f4', bg='#1e1e2e').pack(side=tk.LEFT)
        
        # Compact detection display (fixed width)
        det_frame = tk.Frame(header, bg='#1e1e2e')
        det_frame.pack(side=tk.RIGHT)
        
        self._det_cards = tk.Label(det_frame, text="C:00", font=('Consolas', 8),
                                   fg='#a6e3a1', bg='#1e1e2e', width=4, anchor='e')
        self._det_cards.pack(side=tk.LEFT)
        self._det_holders = tk.Label(det_frame, text="H:0", font=('Consolas', 8),
                                     fg='#cba6f7', bg='#1e1e2e', width=3, anchor='e')
        self._det_holders.pack(side=tk.LEFT)
        self._det_traps = tk.Label(det_frame, text="T:0", font=('Consolas', 8),
                                   fg='#f38ba8', bg='#1e1e2e', width=3, anchor='e')
        self._det_traps.pack(side=tk.LEFT)
        
        # Zone status row
        zone_frame = tk.Frame(self, bg='#313244')
        zone_frame.pack(fill=tk.X, padx=6, pady=4)
        
        self._zone_status = tk.Label(zone_frame, text="⏳ Waiting...", font=('Consolas', 9),
                                     fg='#f9e2af', bg='#313244', padx=6, pady=4,
                                     width=18, anchor='w')
        self._zone_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self._lock_btn = tk.Button(zone_frame, text="🔒", font=('Consolas', 10),
                                   bg='#45475a', fg='#cdd6f4', bd=0, padx=6, pady=2,
                                   cursor='hand2', activebackground='#585b70')
        self._lock_btn.pack(side=tk.RIGHT, padx=2)
        
        # Alignment indicators (compact row)
        align_frame = tk.Frame(self, bg='#1e1e2e')
        align_frame.pack(fill=tk.X, padx=6, pady=4)
        
        self._align_labels = {}
        for name in ['D', 'P', 'H', 'T']:
            lbl = tk.Label(align_frame, text=f"{name}○", font=('Consolas', 9),
                          fg='#6c7086', bg='#1e1e2e', width=3)
            lbl.pack(side=tk.LEFT, padx=2)
            self._align_labels[name] = lbl
        
        # Separator
        tk.Frame(self, bg='#45475a', height=1).pack(fill=tk.X, padx=6, pady=6)
        
        # DEALER section
        dealer_header = tk.Frame(self, bg='#313244')
        dealer_header.pack(fill=tk.X, padx=6, pady=2)
        
        tk.Label(dealer_header, text="DEALER", font=('Consolas', 8, 'bold'),
                fg='#f9e2af', bg='#313244', padx=6, pady=2).pack(side=tk.LEFT)
        
        self._dealer_value = tk.Label(dealer_header, text="--", font=('Consolas', 12, 'bold'),
                                      fg='#f9e2af', bg='#313244', padx=6, width=4, anchor='e')
        self._dealer_value.pack(side=tk.RIGHT)
        
        self._dealer_cards = tk.Label(self, text="--", font=('Consolas', 14, 'bold'),
                                      fg='#cdd6f4', bg='#1e1e2e', width=12, anchor='center')
        self._dealer_cards.pack(fill=tk.X, padx=6, pady=4)
        
        # Pairs row (fixed width labels)
        self._pairs_frame = tk.Frame(self, bg='#1e1e2e')
        self._pairs_frame.pack(fill=tk.X, padx=6, pady=2)
        self._pair_labels = []
        for i in range(4):
            lbl = tk.Label(self._pairs_frame, text="", font=('Consolas', 9),
                          fg='#6c7086', bg='#1e1e2e', width=6, anchor='center')
            lbl.pack(side=tk.LEFT, padx=1)
            self._pair_labels.append(lbl)
        
        # Separator
        tk.Frame(self, bg='#45475a', height=1).pack(fill=tk.X, padx=6, pady=6)
        
        # PLAYER section
        player_header = tk.Frame(self, bg='#313244')
        player_header.pack(fill=tk.X, padx=6, pady=2)
        
        tk.Label(player_header, text="PLAYER", font=('Consolas', 8, 'bold'),
                fg='#89b4fa', bg='#313244', padx=6, pady=2).pack(side=tk.LEFT)
        
        self._player_value = tk.Label(player_header, text="--", font=('Consolas', 12, 'bold'),
                                      fg='#89b4fa', bg='#313244', padx=6, width=4, anchor='e')
        self._player_value.pack(side=tk.RIGHT)
        
        self._player_cards = tk.Label(self, text="--", font=('Consolas', 14, 'bold'),
                                      fg='#cdd6f4', bg='#1e1e2e', width=12, anchor='center')
        self._player_cards.pack(fill=tk.X, padx=6, pady=4)
        
        # Separator
        tk.Frame(self, bg='#45475a', height=1).pack(fill=tk.X, padx=6, pady=6)
        
        # Recommendation
        rec_frame = tk.Frame(self, bg='#313244')
        rec_frame.pack(fill=tk.X, padx=6, pady=4)
        
        self._recommendation = tk.Label(rec_frame, text="--", font=('Consolas', 16, 'bold'),
                                        fg='#f9e2af', bg='#313244', pady=8, width=12)
        self._recommendation.pack()
        
        self._game_phase = tk.Label(rec_frame, text="", font=('Consolas', 8),
                                    fg='#6c7086', bg='#313244', width=20)
        self._game_phase.pack(pady=(0, 4))

    # === UPDATE METHODS (with caching) ===
    
    def update_detections(self, cards: int, holders: int, traps: int):
        key = (cards, holders, traps)
        if getattr(self, '_cache_det', None) == key:
            return
        self._cache_det = key
        self._det_cards.config(text=f"C:{cards}")
        self._det_holders.config(text=f"H:{holders}")
        self._det_traps.config(text=f"T:{traps}")

    def update_zone_status(self, initialized: bool, locked: bool, message: str = ""):
        key = (initialized, locked, message)
        if getattr(self, '_cache_zone', None) == key:
            return
        self._cache_zone = key
        if locked:
            self._zone_status.config(text="🔒 Locked", fg='#89b4fa')
            self._lock_btn.config(text="🔄")
        elif initialized:
            self._zone_status.config(text="✓ Ready", fg='#a6e3a1')
            self._lock_btn.config(text="🔒")
        else:
            txt = message[:16] if len(message) > 16 else message
            self._zone_status.config(text=f"⏳ {txt}", fg='#f9e2af')
            self._lock_btn.config(text="🔒")

    def set_lock_callback(self, callback):
        self._lock_btn.config(command=callback)

    def update_alignment(self, dealer: bool, player: bool, holder: bool, trap: bool):
        key = (dealer, player, holder, trap)
        if getattr(self, '_cache_align', None) == key:
            return
        self._cache_align = key
        mapping = {'D': dealer, 'P': player, 'H': holder, 'T': trap}
        for name, status in mapping.items():
            self._align_labels[name].config(
                text=f"{name}●" if status else f"{name}○",
                fg='#a6e3a1' if status else '#6c7086'
            )

    def update_dealer(self, cards_str: str, value: int, pairs: List):
        pairs_key = tuple((p.scanned.rank if p.scanned else None, 
                          p.real.rank if p.real else None) for p in pairs) if pairs else ()
        key = (cards_str, value, pairs_key)
        if getattr(self, '_cache_dealer', None) == key:
            return
        self._cache_dealer = key
        
        self._dealer_cards.config(text=cards_str if cards_str != "--" else "--")
        self._dealer_value.config(text=str(value) if value > 0 else "--")
        
        for i, lbl in enumerate(self._pair_labels):
            if i < len(pairs):
                p = pairs[i]
                scan = p.scanned.rank.value if p.scanned and p.scanned.rank else "?"
                real = p.real.rank.value if p.real and p.real.rank else "?"
                
                if p.has_both:
                    sym = "=" if p.is_matched else "≠"
                    color = '#a6e3a1' if p.is_matched else '#f38ba8'
                    lbl.config(text=f"{scan}{sym}{real}", fg=color)
                else:
                    lbl.config(text=f"[{scan if p.scanned else real}]", fg='#6c7086')
            else:
                lbl.config(text="")

    def update_player(self, cards_str: str, value: int):
        key = (cards_str, value)
        if getattr(self, '_cache_player', None) == key:
            return
        self._cache_player = key
        self._player_cards.config(text=cards_str if cards_str != "--" else "--")
        self._player_value.config(text=str(value) if value > 0 else "--")

    def update_recommendation(self, text: str, color: str = '#f9e2af'):
        key = (text, color)
        if getattr(self, '_cache_rec', None) == key:
            return
        self._cache_rec = key
        self._recommendation.config(text=text, fg=color)

    def update_phase(self, phase: str):
        if getattr(self, '_cache_phase', None) == phase:
            return
        self._cache_phase = phase
        self._game_phase.config(text=phase)
