"""Detection overlay renderer."""

from typing import List, Tuple
import numpy as np
import cv2

from public_models.obb_type import ObbType
from public_models.rank import Rank
from .colors import OverlayColors


class DetectionRenderer:
    """Renders detection overlays on frames."""

    def __init__(self, colors: OverlayColors = None, min_confidence: float = 0.3):
        self._colors = colors or OverlayColors()
        self._min_confidence = min_confidence
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._thickness = 2

    def render(self, frame: np.ndarray, inferences: List) -> Tuple[np.ndarray, int]:
        """
        Render all detections on frame.

        Returns:
            (rendered_frame, detection_count)
        """
        frame = frame.copy()
        count = 0

        for inf in inferences:
            obb = inf.obb_inference
            if obb.confidence < self._min_confidence:
                continue
            count += 1
            self._draw_detection(frame, inf)

        return frame, count

    def _draw_detection(self, frame: np.ndarray, inference) -> None:
        """Draw a single detection."""
        obb = inference.obb_inference
        rank_inf = inference.rank_inference

        obb_type = obb.obb_type
        pts = np.array(obb.box, dtype=np.int32)

        # Determine color
        color = self._colors.get_obb_color(obb_type)
        label_parts = []

        if obb_type == ObbType.CARD and rank_inf is not None:
            rank = rank_inf.rank
            special_color = self._colors.get_rank_color(rank)

            if special_color:
                color = special_color
                label_parts.append(rank.value)
            elif rank != Rank.UNKNOWN:
                label_parts.append(rank.value)
            else:
                label_parts.append("Card")

            label_parts.append(f"{rank_inf.confidence * 100:.0f}%")
        else:
            label_parts.append(obb_type.value)
            label_parts.append(f"{obb.confidence * 100:.0f}%")

        # Draw OBB polygon
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=self._thickness)

        # Draw label
        label = " ".join(label_parts)
        self._draw_label(frame, label, pts, color)

    def _draw_label(self, frame: np.ndarray, label: str, pts: np.ndarray,
                    color: Tuple[int, int, int]) -> None:
        """Draw label with background."""
        top_idx = np.argmin(pts[:, 1])
        x, y = int(pts[top_idx][0]), int(pts[top_idx][1])

        (tw, th), _ = cv2.getTextSize(label, self._font, self._font_scale, 1)
        label_y = y - 5 if y - th - 10 > 0 else y + th + 15

        cv2.rectangle(frame, (x, label_y - th - 4), (x + tw + 6, label_y + 2), color, -1)
        cv2.putText(frame, label, (x + 3, label_y - 2), self._font,
                   self._font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_zone(self, frame: np.ndarray, zone_rect: Tuple[int, int, int, int],
                  label: str, color: Tuple[int, int, int], alpha: float = 0.2) -> np.ndarray:
        """Draw a semi-transparent zone rectangle."""
        x, y, w, h = zone_rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x + 5, y + 20), self._font, 0.6, color, 2, cv2.LINE_AA)
        return frame

    def draw_card_pairs(self, frame: np.ndarray, pairs: List) -> np.ndarray:
        """
        Draw connections between paired cards (scanned <-> real).

        Green line = matched ranks
        Red line = mismatched ranks
        Label in middle shows consolidated prediction with black text for readability
        """
        for pair in pairs:
            if not pair.scanned or not pair.real:
                continue

            # Get center points
            scan_center = (int(pair.scanned.x), int(pair.scanned.y))
            real_center = (int(pair.real.x), int(pair.real.y))

            # Line color based on match
            if pair.is_matched:
                line_color = (100, 255, 100)  # Light green
                bg_color = (150, 255, 150)    # Lighter green for label bg
            else:
                line_color = (100, 100, 255)  # Light red
                bg_color = (150, 150, 255)    # Lighter red for label bg

            # Draw line with thicker stroke
            cv2.line(frame, scan_center, real_center, line_color, 3, cv2.LINE_AA)

            # Draw label at midpoint
            mid_x = (scan_center[0] + real_center[0]) // 2
            mid_y = (scan_center[1] + real_center[1]) // 2

            conf = pair.consolidated_confidence * 100
            label = f"{pair.display_rank} {conf:.0f}%"

            (tw, th), _ = cv2.getTextSize(label, self._font, 0.6, 2)

            # White background with colored border
            pad = 6
            cv2.rectangle(frame, (mid_x - tw//2 - pad, mid_y - th//2 - pad),
                         (mid_x + tw//2 + pad, mid_y + th//2 + pad), (255, 255, 255), -1)
            cv2.rectangle(frame, (mid_x - tw//2 - pad, mid_y - th//2 - pad),
                         (mid_x + tw//2 + pad, mid_y + th//2 + pad), line_color, 2)

            # Black text for readability
            cv2.putText(frame, label, (mid_x - tw//2, mid_y + th//4),
                       self._font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        return frame