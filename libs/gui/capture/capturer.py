# ====== Code Summary ======
# Provides a threaded, real-time screen capture engine using `mss`.
# Captures a defined region at a target FPS and makes the latest frame available,
# while supporting asynchronous callbacks and safe region updates.

"""Real-time screen capture engine."""

# ====== Standard Library Imports ======
import threading
import time
from typing import Callable, Optional

# ====== Third-Party Library Imports ======
import numpy as np
import mss

# ====== Local Project Imports ======
from .region import CaptureRegion, ScreenInfo


class ScreenCapturer:
    """
    High-performance threaded screen capturer for real-time frame grabbing.

    Captures a configurable screen region at a target frame rate.
    Maintains the latest frame, allows region updates, and supports custom callbacks.
    """

    def __init__(self, region: Optional[CaptureRegion] = None, target_fps: int = 30) -> None:
        """
        Initialize the screen capturer.

        Args:
            region (Optional[CaptureRegion]): Region to capture. Defaults to bottom screen region.
            target_fps (int): Target capture rate in frames per second.
        """
        # 1. Capture settings
        self._region = region or ScreenInfo.get_default_region()
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps

        # 2. Threading and runtime state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 3. Frame storage and metrics
        self._current_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._actual_fps = 0.0

        # 4. Optional user callbacks
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    @property
    def region(self) -> CaptureRegion:
        """
        Get the currently configured capture region.

        Returns:
            CaptureRegion: Current capture region.
        """
        with self._lock:
            return self._region

    @region.setter
    def region(self, value: CaptureRegion) -> None:
        """
        Update the capture region.

        Args:
            value (CaptureRegion): New region to capture.
        """
        with self._lock:
            self._region = value

    @property
    def actual_fps(self) -> float:
        """
        Get the measured frames per second (averaged over 1 second).

        Returns:
            float: Actual capture rate in FPS.
        """
        return self._actual_fps

    @property
    def is_running(self) -> bool:
        """
        Check if the capture loop is currently running.

        Returns:
            bool: True if running.
        """
        return self._running

    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Add a callback function to receive each captured frame.

        Args:
            callback (Callable[[np.ndarray], None]): Function to call with each frame.
        """
        self._callbacks.append(callback)

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get a copy of the latest captured frame.

        Returns:
            Optional[np.ndarray]: RGB frame array or None if no frame available yet.
        """
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def start(self) -> None:
        """
        Start the capture loop in a background thread.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the capture loop and wait for the thread to exit.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """
        Internal capture loop that runs in the background thread.
        Captures frames from the screen at the target FPS.
        """
        with mss.mss() as sct:
            fps_counter = 0
            fps_start = time.time()

            while self._running:
                # 1. Capture start time
                loop_start = time.time()

                # 2. Read current region safely
                with self._lock:
                    monitor = self._region.to_mss_monitor()

                # 3. Capture frame from screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]  # Strip alpha channel

                # 4. Store frame safely
                with self._lock:
                    self._current_frame = frame
                    self._frame_count += 1

                # 5. Trigger callbacks
                for cb in self._callbacks:
                    try:
                        cb(frame)
                    except Exception as e:
                        print(f"Callback error: {e}")

                # 6. Update FPS tracking
                fps_counter += 1
                now = time.time()
                if now - fps_start >= 1.0:
                    self._actual_fps = fps_counter / (now - fps_start)
                    fps_counter = 0
                    fps_start = now

                # 7. Sleep to maintain target frame interval
                elapsed = time.time() - loop_start
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
