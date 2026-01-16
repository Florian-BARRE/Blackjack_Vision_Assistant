# ====== Code Summary ======
# Provides the BatchCropper class, a utility for preprocessing datasets by
# cropping all images in a directory using a configurable crop region. Includes
# config loading/saving for pipeline integration and robust folder-level processing.

from __future__ import annotations

# ====== Standard Library Imports ======
from pathlib import Path

# ====== Third-Party Library Imports ======
from PIL import Image

# ====== Internal Project Imports ======
from typing import Any

# ====== Local Project Imports ======
from .config import CropConfig
from .constants import CROPPING_CONSTANTS
from .helpers import CROPPING_HELPERS


class BatchCropper:
    """Batch cropper for dataset preprocessing using a configurable region.

    Typical usage in a pipeline:
        cropper = BatchCropper.load_config(dataset_dir)
        cropped = cropper.crop_image(img)
    """

    def __init__(self, config: CropConfig) -> None:
        """
        Initialize the batch cropper with a given crop configuration.

        Args:
            config (CropConfig): Crop region configuration object.
        """
        self.config: CropConfig = config

    # --------------------------
    # Core operations
    # --------------------------

    def crop_image(self, img: Image.Image) -> Image.Image:
        """
        Crop a PIL image using the cropper's configuration.

        Args:
            img (Image.Image): The image to be cropped.

        Returns:
            Image.Image: The cropped image.
        """
        # 1. Get image dimensions
        w, h = img.size

        # 2. Validate the configured crop box
        CROPPING_HELPERS.validate_crop(
            self.config.left,
            self.config.top,
            self.config.right,
            self.config.bottom,
            w,
            h,
        )

        # 3. Crop and return image
        return img.crop(self.config.to_box())

    def crop_file(self, image_path: Path) -> Image.Image:
        """
        Load an image from disk and crop it using the current configuration.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            Image.Image: Cropped image.
        """
        # 1. Load the image
        img = CROPPING_HELPERS.load_image(image_path)

        # 2. Crop and return the image
        return self.crop_image(img)

    def apply_to_folder(
            self,
            input_dir: Path,
            output_dir: Path,
            overwrite: bool = False,
            keep_format: bool = True,
            suffix: str = "",
    ) -> dict[str, Any]:
        """
        Apply the configured crop to all images in a folder and save the results.

        Args:
            input_dir (Path): Directory containing the input images.
            output_dir (Path): Directory to save cropped images.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            keep_format (bool, optional): Keep the original file extension. Defaults to True.
            suffix (str, optional): Optional suffix to append to output filenames. Defaults to "".

        Returns:
            dict[str, Any]: A report with statistics and errors.
        """
        # 1. Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. List image files from input directory
        paths = CROPPING_HELPERS.list_images(input_dir)
        if not paths:
            raise ValueError(f"No images found in: {input_dir.resolve()}")

        # 3. Initialize processing report
        report: dict[str, Any] = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "processed": 0,
            "skipped": 0,
            "errors": [],
        }

        # 4. Process each image
        for p in paths:
            try:
                # a. Load image
                img = CROPPING_HELPERS.load_image(p)

                # b. Apply crop
                cropped = self.crop_image(img)

                # c. Construct output filename
                out_name = p.stem + (suffix or "") + (p.suffix if keep_format else ".png")
                out_path = output_dir / out_name

                # d. Skip if file exists and not overwriting
                if out_path.exists() and not overwrite:
                    report["skipped"] += 1
                    continue

                # e. Save cropped image
                cropped.save(out_path)
                report["processed"] += 1

            except Exception as e:
                # f. Capture any error in processing
                report["errors"].append({"file": str(p), "error": str(e)})

        # 5. Return final report
        return report

    # --------------------------
    # Config I/O (pipeline-friendly)
    # --------------------------

    @classmethod
    def load_config(
            cls,
            directory: Path = CROPPING_CONSTANTS.DEFAULT_CONFIG_PATH,
            filename: str = CROPPING_CONSTANTS.DEFAULT_CONFIG_FILENAME,
            *,
            required: bool = True,
    ) -> BatchCropper:
        """
        Load crop configuration from a directory.

        If `required=True`, a missing config file raises FileNotFoundError.
        If `required=False`, defaults to a full-image crop based on the first
        image in the directory (if one is found).

        Args:
            directory (Path): Directory to load the config from.
            filename (str, optional): Name of the config file. Defaults to CROPPING_CONSTANTS.DEFAULT_CONFIG_FILENAME.
            required (bool, optional): Whether the config file is required. Defaults to True.

        Raises:
            FileNotFoundError: If config file is required but not found.
            ValueError: If no images exist to infer size when required=False.

        Returns:
            BatchCropper: Cropper initialized with the loaded or inferred config.
        """
        # 1. Build full config path
        config_path = directory / filename

        # 2. Try loading config if it exists
        if config_path.exists():
            cfg = CropConfig.load(config_path)
            return cls(cfg)

        # 3. Raise if required but not found
        if required:
            raise FileNotFoundError(
                f"Crop config not found: {config_path.resolve()} (required=True)"
            )

        # 4. Fallback: infer config from first image in folder
        images = CROPPING_HELPERS.list_images(directory)
        if not images:
            raise ValueError(f"No images found to infer size in: {directory.resolve()}")

        img = CROPPING_HELPERS.load_image(images[0])
        w, h = img.size
        return cls(CropConfig(left=0, top=0, right=w, bottom=h))

    def save_config(
            self,
            directory: Path = CROPPING_CONSTANTS.DEFAULT_CONFIG_PATH,
            filename: str = CROPPING_CONSTANTS.DEFAULT_CONFIG_FILENAME,
    ) -> Path:
        """
        Save the current crop configuration to the specified directory.

        Args:
            directory (Path): Directory where the config should be saved.
            filename (str, optional): Filename for the config. Defaults to CROPPING_CONSTANTS.DEFAULT_CONFIG_FILENAME.

        Returns:
            Path: The full path where the config was saved.
        """
        # 1. Ensure target directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # 2. Save the config
        path = directory / filename
        self.config.save(path)

        # 3. Return the saved path
        return path
