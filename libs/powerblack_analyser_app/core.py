"""
Blackjack Analyzer
==================

Real-time blackjack game analysis with YOLO detection.
Responsive UI with persistent configuration.
"""

# ====== Standard Library Imports ======
from __future__ import annotations

from pathlib import Path
import tkinter as tk
import uuid
from typing import Optional

# ====== Third-Party Imports ======
import cv2
import numpy as np
from PIL import Image, ImageTk

# ====== Internal Project Imports ======
from gui import (
    CaptureRegion,
    ScreenInfo,
    ScreenCapturer,
    DetectionRenderer,
    OverlayColors,
    EdgeHandle,
    InfoPanel,
    StrategyGrid,
    GuiConfigManager,
)
from blackjack_logic import PowerBlackjackState, PowerBlackjackStrategy, SpecialEvent
from yolo_model import MetaModel
from loggerplusplus import LoggerClass


class PowerBlackjackAnalyzerApp(LoggerClass):
    """Main application with responsive UI and persistent config.

    This version adds full method-level logging via LoggerClass / LoggerPlusPlus.
    """

    _LOG_SAMPLE_EVERY_N_TICKS: int = 30

    def __init__(
            self,
            model: MetaModel,
            fps: int,
            min_confidence: float,
            gui_config_path: Path | None,
    ) -> None:
        LoggerClass.__init__(self)

        self._run_id: str = self._new_run_id()
        self._seq: int = 0
        self.logger.info(
            f"START initializing BlackjackAnalyzer (run_id={self._run_id}, fps={fps}, "
            f"min_confidence={min_confidence}, gui_config_path={gui_config_path})"
        )
        try:
            self._model = model
            self._fps = fps
            self._min_confidence = min_confidence

            # Load config
            self._config_mgr = GuiConfigManager(config_path=gui_config_path)
            self._cfg = self._config_mgr.load()
            self.logger.info(
                f"END loading GUI config (run_id={self._run_id}, "
                f"window={self._cfg.window_width}x{self._cfg.window_height}+{self._cfg.window_x}+{self._cfg.window_y}, "
                f"capture={self._cfg.capture_width}x{self._cfg.capture_height}+{self._cfg.capture_x}+{self._cfg.capture_y})"
            )

            # Initialize region from config
            region = CaptureRegion(
                self._cfg.capture_x,
                self._cfg.capture_y,
                self._cfg.capture_width,
                self._cfg.capture_height,
            )

            self._capturer = ScreenCapturer(region=region, target_fps=fps)
            self._renderer = DetectionRenderer(min_confidence=min_confidence)
            self._game_state = PowerBlackjackState()

            self._root: Optional[tk.Tk] = None
            self._canvas: Optional[tk.Canvas] = None
            self._photo: Optional[ImageTk.PhotoImage] = None
            self._info_panel: Optional[InfoPanel] = None
            self._strategy_grid: Optional[StrategyGrid] = None
            self._handles: list[EdgeHandle] = []
            self._handles_visible = True
            self._running = False
            self._detection_count = 0

            self.logger.info(f"END initializing BlackjackAnalyzer (run_id={self._run_id})")
        except Exception as exc:
            # Log exactly once, then re-raise because init failure should be fatal.
            self.logger.exception(
                f"CRITICAL failed initializing BlackjackAnalyzer (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )
            raise

    @staticmethod
    def _new_run_id() -> str:
        return uuid.uuid4().hex[:10]

    def _create_window(self) -> None:
        """Create the main window."""
        self.logger.info(f"START creating window (run_id={self._run_id})")
        try:
            self._root = tk.Tk()
            self._root.title("Blackjack Analyzer")
            self._root.configure(bg="#1e1e2e")

            # Restore window position/size from config
            self._root.geometry(
                f"{self._cfg.window_width}x{self._cfg.window_height}"
                f"+{self._cfg.window_x}+{self._cfg.window_y}"
            )
            self._root.minsize(800, 400)
            self._root.resizable(True, True)

            # Main horizontal container
            main = tk.Frame(self._root, bg="#1e1e2e")
            main.pack(fill=tk.BOTH, expand=True)

            main.columnconfigure(0, weight=1)  # Preview expands
            main.columnconfigure(1, weight=0, minsize=self._cfg.info_panel_width)
            main.columnconfigure(2, weight=0, minsize=self._cfg.strategy_panel_width)
            main.rowconfigure(0, weight=1)

            # Left: Preview area (expandable)
            left = tk.Frame(main, bg="#1e1e2e")
            left.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
            left.rowconfigure(1, weight=1)
            left.columnconfigure(0, weight=1)

            self._create_toolbar(left)
            self._create_preview(left)

            # Middle: Info panel
            self._info_panel = InfoPanel(main, width=self._cfg.info_panel_width)
            self._info_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 5), pady=5)
            self._info_panel.set_lock_callback(self._toggle_zone_lock)

            # Right: Strategy grid
            self._strategy_grid = StrategyGrid(main, width=self._cfg.strategy_panel_width)
            self._strategy_grid.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=5)

            # Bindings
            self._root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._root.bind("<Escape>", lambda _e: self._on_close())
            self._root.bind("<h>", lambda _e: self._toggle_handles())
            self._root.bind("<H>", lambda _e: self._toggle_handles())
            self._root.bind("<Configure>", self._on_resize)

            self.logger.info(
                f"END creating window (run_id={self._run_id}, "
                f"info_panel_width={self._cfg.info_panel_width}, strategy_panel_width={self._cfg.strategy_panel_width})"
            )
        except Exception as exc:
            self.logger.exception(
                f"ERROR creating window (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )
            raise

    def _create_toolbar(self, parent: tk.Misc) -> None:
        """Create top toolbar."""
        self.logger.debug(f"START creating toolbar (run_id={self._run_id})")
        toolbar = tk.Frame(parent, bg="#313244", height=24)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        toolbar.grid_propagate(False)

        self._toggle_btn = tk.Button(
            toolbar,
            text="◇",
            font=("Consolas", 10),
            bg="#45475a",
            fg="#cdd6f4",
            bd=0,
            padx=8,
            command=self._toggle_handles,
            cursor="hand2",
        )
        self._toggle_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self._fps_label = tk.Label(
            toolbar,
            text="-- fps",
            font=("Consolas", 9),
            fg="#a6e3a1",
            bg="#313244",
        )
        self._fps_label.pack(side=tk.LEFT, padx=8)

        self._det_label = tk.Label(
            toolbar,
            text="0 det",
            font=("Consolas", 9),
            fg="#89b4fa",
            bg="#313244",
        )
        self._det_label.pack(side=tk.LEFT, padx=4)

        # Region info
        self._region_label = tk.Label(
            toolbar,
            text="",
            font=("Consolas", 8),
            fg="#6c7086",
            bg="#313244",
        )
        self._region_label.pack(side=tk.RIGHT, padx=8)
        self.logger.debug(f"END creating toolbar (run_id={self._run_id})")

    def _create_preview(self, parent: tk.Misc) -> None:
        """Create preview canvas with edge handles."""
        self.logger.debug(f"START creating preview (run_id={self._run_id})")
        frame = tk.Frame(parent, bg="#11111b")
        frame.grid(row=1, column=0, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        for edge, row, col, padx in [
            ("top", 0, 1, 0),
            ("left", 1, 0, (0, 2)),
            ("right", 1, 2, (2, 0)),
            ("bottom", 2, 1, 0),
        ]:
            handle = EdgeHandle(frame, edge, self._adjust_edge)
            handle.grid(row=row, column=col, padx=padx)
            self._handles.append(handle)

        self._canvas = tk.Canvas(
            frame,
            bg="#11111b",
            highlightthickness=1,
            highlightbackground="#45475a",
        )
        self._canvas.grid(row=1, column=1, sticky="nsew")
        self.logger.debug(f"END creating preview (run_id={self._run_id}, handles={len(self._handles)})")

    def _on_resize(self, event: tk.Event) -> None:
        """Handle window resize."""
        if event.widget == self._root:
            self._seq += 1
            if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
                self.logger.debug(
                    f"PROGRESS window resize tick (run_id={self._run_id}, seq={self._seq}, geometry={self._root.geometry()})"
                )
            self._root.update_idletasks()

    def _toggle_handles(self) -> None:
        self._handles_visible = not self._handles_visible
        for handle in self._handles:
            handle.grid() if self._handles_visible else handle.grid_remove()
        self._toggle_btn.configure(text="◇" if self._handles_visible else "◆")
        self.logger.info(
            f"END toggling handles (run_id={self._run_id}, visible={self._handles_visible}, handles={len(self._handles)})"
        )

    def _toggle_zone_lock(self) -> None:
        """Toggle zone lock state."""
        self.logger.info(
            f"START toggling zone lock (run_id={self._run_id}, "
            f"is_initialized={self._game_state.is_initialized}, is_locked={self._game_state.is_locked})"
        )
        try:
            if self._game_state.is_locked:
                self._game_state.unlock_zones()
                self.logger.info(f"END toggling zone lock (run_id={self._run_id}, action=unlock)")
                return

            if self._game_state.is_initialized:
                self._game_state.lock_zones()
                self.logger.info(f"END toggling zone lock (run_id={self._run_id}, action=lock)")
                return

            self.logger.warning(
                f"END toggling zone lock (run_id={self._run_id}, action=ignored, reason=not_initialized)"
            )
        except Exception as exc:
            self.logger.exception(
                f"ERROR toggling zone lock (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )

    def _adjust_edge(self, edge: str, delta: int) -> None:
        region = self._capturer.region
        sw, sh = ScreenInfo.get_primary_monitor_size()
        x, y, w, h = region.x, region.y, region.width, region.height

        self.logger.debug(
            f"START adjusting edge (run_id={self._run_id}, edge={edge}, delta={delta}, "
            f"region={w}x{h}+{x}+{y}, screen={sw}x{sh})"
        )

        if edge == "left":
            nx = max(0, x - delta)
            w += x - nx
            x = nx
        elif edge == "right":
            w = min(sw - x, max(100, w + delta))
        elif edge == "top":
            ny = max(0, y - delta)
            h += y - ny
            y = ny
        elif edge == "bottom":
            h = min(sh - y, max(100, h + delta))
        else:
            self.logger.warning(f"END adjusting edge (run_id={self._run_id}, action=ignored, edge={edge})")
            return

        w, h = max(100, w), max(100, h)
        self._capturer.region = CaptureRegion(x, y, w, h)

        self.logger.info(
            f"END adjusting edge (run_id={self._run_id}, edge={edge}, "
            f"new_region={w}x{h}+{x}+{y})"
        )

    def _calculate_scale(self, frame_w: int, frame_h: int) -> float:
        """Calculate scale to fit frame in available canvas space."""
        if self._canvas is None:
            self.logger.warning(f"END calculating scale (run_id={self._run_id}, reason=no_canvas, scale=0.5)")
            return 0.5

        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            self.logger.debug(
                f"END calculating scale (run_id={self._run_id}, canvas={canvas_w}x{canvas_h}, scale=0.5)"
            )
            return 0.5

        scale_w = canvas_w / frame_w
        scale_h = canvas_h / frame_h
        scale = min(scale_w, scale_h, 1.0)

        if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
            self.logger.debug(
                f"END calculating scale (run_id={self._run_id}, frame={frame_w}x{frame_h}, "
                f"canvas={canvas_w}x{canvas_h}, scale={scale:.3f})"
            )
        return scale

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with inference and overlay."""
        self.logger.debug(
            f"START processing frame (run_id={self._run_id}, shape={frame.shape}, dtype={frame.dtype})"
        )
        try:
            inferences = self._model.infer(frame)
            self._game_state.update(inferences, frame.shape[:2])
            frame, self._detection_count = self._renderer.render(frame, inferences)

            if self._game_state.is_initialized:
                colors = OverlayColors()
                if self._game_state.dealer_zone:
                    frame = self._renderer.draw_zone(
                        frame,
                        self._game_state.dealer_zone.rect,
                        "DEALER",
                        colors.DEALER_ZONE,
                        0.15,
                    )
                if self._game_state.player_zone:
                    frame = self._renderer.draw_zone(
                        frame,
                        self._game_state.player_zone.rect,
                        "PLAYER",
                        colors.PLAYER_ZONE,
                        0.15,
                    )

                if self._game_state.dealer_pairs:
                    frame = self._renderer.draw_card_pairs(frame, self._game_state.dealer_pairs)

            self.logger.debug(
                f"END processing frame (run_id={self._run_id}, det={self._detection_count}, "
                f"is_initialized={self._game_state.is_initialized}, is_locked={self._game_state.is_locked})"
            )
            return frame
        except Exception as exc:
            # Preserve current behavior: swallow and continue with original frame.
            self.logger.exception(
                f"ERROR processing frame (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )
            return frame

    def _update_info_panel(self) -> None:
        """Update info panel and strategy grid."""
        if self._info_panel is None or self._strategy_grid is None:
            self.logger.warning(
                f"END updating panels (run_id={self._run_id}, reason=panels_not_ready, "
                f"info_panel={self._info_panel is not None}, strategy_grid={self._strategy_grid is not None})"
            )
            return

        gs = self._game_state
        zd = gs.zone_detector

        try:
            self._info_panel.update_detections(gs.cards_count, gs.holders_count, gs.traps_count)
            self._info_panel.update_zone_status(
                gs.is_initialized,
                gs.is_locked,
                zd.get_status() if not gs.is_initialized else "",
            )
            self._info_panel.update_alignment(
                zd.dealer_aligned,
                zd.player_aligned,
                zd.holder_found,
                zd.trap_found,
            )

            dealer_str = gs.get_dealer_display()
            dealer_val = gs.get_dealer_value()
            self._info_panel.update_dealer(dealer_str, dealer_val, gs.dealer_pairs)

            player_str = gs.get_player_display()
            player_val = gs.get_player_value()
            self._info_panel.update_player(player_str, player_val)

            rec_text, rec_color = gs.get_recommendation()
            self._info_panel.update_recommendation(rec_text, rec_color)
            self._info_panel.update_phase(gs.get_status())

            # Special events
            if gs.special_event != SpecialEvent.NONE:
                title, message, color = gs.get_special_event_message()
                self._strategy_grid.show_special_event(title, message, color)
                self.logger.info(
                    f"PROGRESS special event (run_id={self._run_id}, event={gs.special_event})"
                )
                return

            self._strategy_grid.hide_special_event()

            if gs.is_initialized and gs.dealer_pairs and gs.player_cards:
                dealer_rank = gs.dealer_pairs[0].consolidated_rank if gs.dealer_pairs else None
                player_ranks = [c.rank for c in gs.player_cards]

                if dealer_rank and player_ranks:
                    result = PowerBlackjackStrategy.lookup(player_ranks, dealer_rank)
                    self._strategy_grid.update(result)
                    if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
                        self.logger.debug(
                            f"PROGRESS strategy lookup (run_id={self._run_id}, dealer_rank={dealer_rank}, "
                            f"player_ranks={player_ranks}, result={result})"
                        )
                else:
                    self._strategy_grid.update(None)
            else:
                self._strategy_grid.update(None)

            if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
                self.logger.debug(
                    f"END updating panels (run_id={self._run_id}, "
                    f"dealer_cards={len(gs.dealer_pairs) if gs.dealer_pairs else 0}, "
                    f"player_cards={len(gs.player_cards) if gs.player_cards else 0}, "
                    f"is_initialized={gs.is_initialized}, is_locked={gs.is_locked})"
                )
        except Exception as exc:
            # Preserve behavior: UI failures should not kill the loop.
            self.logger.exception(
                f"ERROR updating panels (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )

    def _update(self) -> None:
        """Main update loop."""
        if not self._running:
            self.logger.debug(f"END update tick (run_id={self._run_id}, running=False)")
            return

        if self._root is None or self._canvas is None:
            self.logger.warning(
                f"END update tick (run_id={self._run_id}, reason=ui_not_ready, "
                f"root={self._root is not None}, canvas={self._canvas is not None})"
            )
            return

        self._seq += 1
        frame = self._capturer.get_current_frame()

        if frame is not None:
            frame = self._process_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            scale = self._calculate_scale(frame_rgb.shape[1], frame_rgb.shape[0])
            pw = int(frame_rgb.shape[1] * scale)
            ph = int(frame_rgb.shape[0] * scale)

            if pw <= 0 or ph <= 0:
                self.logger.warning(
                    f"PROGRESS skipped render due to invalid size (run_id={self._run_id}, "
                    f"pw={pw}, ph={ph}, scale={scale})"
                )
            else:
                resized = cv2.resize(frame_rgb, (pw, ph), interpolation=cv2.INTER_LINEAR)

                self._photo = ImageTk.PhotoImage(Image.fromarray(resized))
                self._canvas.delete("all")
                self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

                fps = self._capturer.actual_fps
                self._fps_label.config(text=f"{fps:.0f} fps")
                self._det_label.config(text=f"{self._detection_count} det")

                region = self._capturer.region
                self._region_label.config(text=f"{region.width}×{region.height}")

                self._update_info_panel()

                if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
                    self.logger.debug(
                        f"PROGRESS render tick (run_id={self._run_id}, seq={self._seq}, "
                        f"fps={fps:.1f}, det={self._detection_count}, region={region.width}x{region.height}+{region.x}+{region.y}, "
                        f"preview={pw}x{ph})"
                    )
        else:
            if self._seq % self._LOG_SAMPLE_EVERY_N_TICKS == 0:
                self.logger.warning(
                    f"PROGRESS no frame available (run_id={self._run_id}, seq={self._seq})"
                )

        self._root.after(int(1000 / self._fps), self._update)

    def _save_config(self) -> None:
        """Save current state to config."""
        self.logger.info(f"START saving config (run_id={self._run_id})")
        try:
            if self._root is None:
                self.logger.warning(f"END saving config (run_id={self._run_id}, reason=no_root)")
                return

            geo = self._root.geometry()
            size, pos = geo.split("+", 1)
            w, h = map(int, size.split("x"))
            x, y = map(int, pos.split("+"))

            region = self._capturer.region
            self._root.update_idletasks()

            info_w = self._info_panel.winfo_width() if self._info_panel else self._cfg.info_panel_width
            strat_w = (
                self._strategy_grid.winfo_width() if self._strategy_grid else self._cfg.strategy_panel_width
            )

            self._config_mgr.update(
                window_x=x,
                window_y=y,
                window_width=w,
                window_height=h,
                capture_x=region.x,
                capture_y=region.y,
                capture_width=region.width,
                capture_height=region.height,
                info_panel_width=int(info_w),
                strategy_panel_width=int(strat_w),
            )
            self._config_mgr.save()

            self.logger.info(
                f"END saving config (run_id={self._run_id}, window={w}x{h}+{x}+{y}, "
                f"capture={region.width}x{region.height}+{region.x}+{region.y}, "
                f"info_panel_width={int(info_w)}, strategy_panel_width={int(strat_w)})"
            )
        except Exception as exc:
            # Preserve behavior: warn and keep closing.
            self.logger.warning(
                f"WARNING could not save config (run_id={self._run_id}, exc_type={type(exc).__name__}, "
                f"message={exc})"
            )

    def _on_close(self) -> None:
        """Clean shutdown."""
        self.logger.info(f"START shutdown (run_id={self._run_id})")
        try:
            self._running = False
            self._save_config()
            self._capturer.stop()
            if self._root:
                self._root.destroy()
            self.logger.info(f"END shutdown (run_id={self._run_id})")
        except Exception as exc:
            # If shutdown fails, log once; do not re-raise from GUI close path.
            self.logger.exception(
                f"ERROR during shutdown (run_id={self._run_id}, exc_type={type(exc).__name__})"
            )

    def run(self) -> None:
        """Start the application."""
        self.logger.info(f"START run (run_id={self._run_id})")
        try:
            self._create_window()
            self._capturer.start()
            self._running = True
            self.logger.info(
                f"PROGRESS capture started (run_id={self._run_id}, target_fps={self._fps}, "
                f"region={self._capturer.region.width}x{self._capturer.region.height}+{self._capturer.region.x}+{self._capturer.region.y})"
            )
            self._update()
            self._root.mainloop()
            self.logger.info(f"END run (run_id={self._run_id})")
        except Exception as exc:
            self.logger.exception(f"CRITICAL run failed (run_id={self._run_id}, exc_type={type(exc).__name__})")
            raise
        finally:
            # Ensure END story at INFO level if START happened, even if an exception bubbles.
            if self._running:
                self.logger.info(f"END run cleanup (run_id={self._run_id}, running=True -> False)")
            self._running = False
