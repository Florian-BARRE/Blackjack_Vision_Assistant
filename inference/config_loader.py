# ====== Standard Library Imports ======
from dotenv import load_dotenv
from typing import Any
import pathlib
import sys
import os

# ====== Third-Party Library Imports ======
from loggerplusplus import loggerplusplus
from loggerplusplus import formats as lpp_formats

# Load .env file
load_dotenv()


# ───────────────────── helper ──────────────────────────────────
def env(key: str, *, default: Any = None, cast: Any = str):
    """Tiny helper to read ENV with optional cast & default."""
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"missing required env var {key}")
    if cast == bool and isinstance(val, str):
        return val.strip().lower() not in {"false", "False", "0", "no", ""}
    return cast(val)


class CONFIG:
    """All configuration values exposed as class attributes."""
    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
    LIBS_DIR = ROOT_DIR / "libs"

    # Add Tools dir to python path
    sys.path.append(str(LIBS_DIR))  # Add libs directory to the path for imports

    # ──── PowerBlackjackAnalyzer - App ────
    ANALYSER_APP_FPS = env("ANALYSER_APP_FPS", cast=int)
    ANALYSER_APP_MIN_CONFIDENCE = env("ANALYSER_APP_MIN_CONFIDENCE", cast=float)

    # ──── GUI ────
    GUI_CONFIG_PATH = ROOT_DIR / env("GUI_CONFIG_PATH")

    # ──── YOLO ────
    # Common model config
    YOLO_MODEL_DEVICE = env("YOLO_MODEL_DEVICE", cast=int)
    YOLO_MODEL_VERBOSE = env("YOLO_MODEL_VERBOSE", cast=bool)
    # OBB model
    OBB_MODEL_IMGSZ = env("OBB_MODEL_IMGSZ", cast=int)
    OBB_MODEL_CONF = env("OBB_MODEL_CONF", cast=float)
    OBB_MODEL_IOU = env("OBB_MODEL_IOU", cast=float)
    OBB_MODEL_MAX_DET = env("OBB_MODEL_MAX_DET", cast=int)
    OBB_MODEL_PATH = env("OBB_MODEL_PATH", cast=pathlib.Path)

    # CLS model
    CLS_MODEL_IMGSZ = env("CLS_MODEL_IMGSZ", cast=int)
    CLS_MODEL_CONF = env("CLS_MODEL_CONF", cast=float)
    CLS_MODEL_PATH = env("CLS_MODEL_PATH", cast=pathlib.Path)

    # ───── logging ─────
    CONSOLE_LEVEL = env("CONSOLE_LEVEL")
    FILE_LEVEL = env("FILE_LEVEL")

    ENABLE_CONSOLE = env("ENABLE_CONSOLE", cast=bool)
    ENABLE_FILE = env("ENABLE_FILE", cast=bool)


# ────── Apply logger config ──────
loggerplusplus.remove()  # avoid double logging
lpp_format = lpp_formats.ShortFormat(identifier_width=15)

if CONFIG.ENABLE_CONSOLE:
    loggerplusplus.add(
        sink=sys.stdout,
        level=CONFIG.CONSOLE_LEVEL,
        format=lpp_format,
    )

if CONFIG.ENABLE_FILE:
    loggerplusplus.add(
        pathlib.Path("logs"),
        level=CONFIG.FILE_LEVEL,
        format=lpp_format,
        rotation="1 week",  # "100 MB" / "00:00"
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
