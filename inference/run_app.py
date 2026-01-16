"""
Blackjack Analyzer - Application Entry Point

Initializes models and launches the PowerBlackjackAnalyzerApp.
"""

# ====== Imports ======
from config_loader import CONFIG

from yolo_model import MetaModel, ObbModel, RankModel
from powerblack_analyser_app import PowerBlackjackAnalyzerApp


def main() -> None:
    """Initialize and run the Power Blackjack Analyzer."""
    app = PowerBlackjackAnalyzerApp(
        model=MetaModel(
            obb_model=ObbModel(
                path=CONFIG.OBB_MODEL_PATH,
                imgsz=CONFIG.OBB_MODEL_IMGSZ,
                conf=CONFIG.OBB_MODEL_CONF,
                iou=CONFIG.OBB_MODEL_IOU,
                device=CONFIG.YOLO_MODEL_DEVICE,
                verbose=CONFIG.YOLO_MODEL_VERBOSE,
                max_det=CONFIG.OBB_MODEL_MAX_DET,
            ),
            rank_model=RankModel(
                path=CONFIG.CLS_MODEL_PATH,
                imgsz=CONFIG.CLS_MODEL_IMGSZ,
                device=CONFIG.YOLO_MODEL_DEVICE,
                verbose=CONFIG.YOLO_MODEL_VERBOSE,
                conf=CONFIG.CLS_MODEL_CONF,
            ),
        ),
        fps=CONFIG.ANALYSER_APP_FPS,
        min_confidence=CONFIG.ANALYSER_APP_MIN_CONFIDENCE,
        gui_config_path=CONFIG.GUI_CONFIG_PATH,
    )
    app.run()


if __name__ == "__main__":
    main()
