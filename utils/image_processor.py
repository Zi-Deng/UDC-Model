"""Custom image processor replacing HuggingFace AutoImageProcessor.

Provides model-aware normalisation and transform pipelines (training with
augmentation, validation/test deterministic) for ResNet, ConvNeXt, ViT,
EfficientNet, and Swin architectures.  Configuration is auto-detected
from the model name or loaded from a JSON file.
"""

import json
import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image


class RandomQuarterTurns:
    """Randomly rotate a PIL image by 0, 90, 180, or 270 degrees."""

    def __call__(self, image: Image.Image) -> Image.Image:
        turns = random.randint(0, 3)
        if turns == 0:
            return image
        return image.rotate(90 * turns)


class CustomImageProcessor:
    """
    Custom Image Processor that replaces AutoImageProcessor with configurable parameters
    for different models while maintaining the same interface and functionality.
    """

    # Default configurations for common models
    MODEL_CONFIGS = {
        "resnet": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 224, "width": 224},
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        },
        "vit": {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "size": {"height": 224, "width": 224},
            "crop_pct": 0.9,
            "interpolation": "bicubic",
        },
        "convnext": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"shortest_edge": 224},
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
        "efficientnet": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"shortest_edge": 224},
            "crop_pct": 1.0,
            "interpolation": "bicubic",
        },
        "swin": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 224, "width": 224},
            "crop_pct": 0.9,
            "interpolation": "bicubic",
        },
        "timm_dinov3_vit": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 256, "width": 256},
            "crop_pct": 1.0,
            "interpolation": "bicubic",
        },
        "timm_dinov3_convnext": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 224, "width": 224},
            "crop_pct": 1.0,
            "interpolation": "bicubic",
        },
        "facebook_dinov3": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 224, "width": 224},
            "crop_pct": 1.0,
            "interpolation": "bilinear",
        },
    }

    def __init__(
        self,
        model_name: str = "resnet",
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        size: dict[str, int] | None = None,
        crop_pct: float = 0.875,
        interpolation: str = "bilinear",
        image_size: int | None = None,
        custom_config: dict | None = None,
    ):
        """
        Initialize the Custom Image Processor.

        Args:
            model_name: Name of the model type ('resnet', 'vit', 'convnext', etc.)
            image_mean: Custom mean values for normalization
            image_std: Custom std values for normalization
            size: Custom size configuration
            crop_pct: Crop percentage for center crop
            interpolation: Interpolation method ('bilinear', 'bicubic')
            custom_config: Complete custom configuration dictionary
        """
        # Use custom config if provided, otherwise use model-specific config
        if custom_config:
            config = dict(custom_config)
            self.data_config_source = "custom"
        elif self._is_timm_model_name(model_name):
            config = self._config_from_timm_model(model_name)
            self.data_config_source = "timm"
        else:
            # Extract model type from model name for common architectures
            model_type = self._extract_model_type(model_name)
            config = dict(self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS["resnet"]))
            self.data_config_source = model_type

        if image_size is not None:
            config = dict(config)
            config["size"] = {"height": int(image_size), "width": int(image_size)}

        # Override with provided parameters
        self.image_mean = image_mean if image_mean is not None else config["image_mean"]
        self.image_std = image_std if image_std is not None else config["image_std"]
        self.size = size if size is not None else config["size"]
        self.crop_pct = crop_pct if crop_pct != 0.875 else config.get("crop_pct", 0.875)
        self.interpolation = interpolation if interpolation != "bilinear" else config.get("interpolation", "bilinear")

        # Set interpolation mode
        if self.interpolation == "bicubic":
            self.interp_mode = transforms.InterpolationMode.BICUBIC
        else:
            self.interp_mode = transforms.InterpolationMode.BILINEAR

        # Calculate crop size
        self.crop_size = self._calculate_crop_size()

        # Create basic transforms
        self._create_transforms()

    @staticmethod
    def _is_timm_model_name(model_name: str) -> bool:
        model_name_lower = str(model_name).lower()
        return model_name_lower.startswith("timm/") or ".dinov3" in model_name_lower or ".fb_" in model_name_lower

    @staticmethod
    def _config_from_timm_model(model_name: str) -> dict:
        try:
            from timm.data import resolve_model_data_config
        except ImportError:
            return dict(CustomImageProcessor.MODEL_CONFIGS["resnet"])

        timm_name = str(model_name)
        if timm_name.startswith("timm/"):
            timm_name = timm_name.split("/", 1)[1]
        try:
            data_config = resolve_model_data_config(timm_name)
        except Exception:
            return dict(CustomImageProcessor.MODEL_CONFIGS["resnet"])

        input_size = data_config.get("input_size", (3, 224, 224))
        height = int(input_size[-2])
        width = int(input_size[-1])
        return {
            "image_mean": list(data_config.get("mean", (0.485, 0.456, 0.406))),
            "image_std": list(data_config.get("std", (0.229, 0.224, 0.225))),
            "size": {"height": height, "width": width},
            "crop_pct": float(data_config.get("crop_pct", 0.875)),
            "interpolation": str(data_config.get("interpolation", "bilinear")).lower(),
        }

    def _extract_model_type(self, model_name: str) -> str:
        """Extract model type from model name string."""
        model_name_lower = model_name.lower()

        is_timm_dinov3 = model_name_lower.startswith("timm/") or ".dinov3" in model_name_lower
        is_facebook_dinov3 = model_name_lower.startswith("facebook/dinov3-")
        if is_facebook_dinov3:
            return "facebook_dinov3"
        elif is_timm_dinov3 and "convnext" in model_name_lower:
            return "timm_dinov3_convnext"
        elif is_timm_dinov3 and "vit" in model_name_lower:
            return "timm_dinov3_vit"
        elif "resnet" in model_name_lower:
            return "resnet"
        elif "vit" in model_name_lower or "vision" in model_name_lower:
            return "vit"
        elif "convnext" in model_name_lower:
            return "convnext"
        elif "efficientnet" in model_name_lower:
            return "efficientnet"
        elif "swin" in model_name_lower:
            return "swin"
        else:
            # Default to resnet config for unknown models
            return "resnet"

    def _calculate_crop_size(self) -> int | tuple[int, int]:
        """Calculate crop size based on size configuration."""
        if "height" in self.size and "width" in self.size:
            return (self.size["height"], self.size["width"])
        elif "shortest_edge" in self.size:
            return self.size["shortest_edge"]
        else:
            # Default fallback
            return 224

    def _create_transforms(self):
        """Create the transformation pipelines."""
        # Normalization transform
        self.normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)

        # Calculate resize size based on crop percentage
        if isinstance(self.crop_size, tuple):
            resize_size = (int(self.crop_size[0] / self.crop_pct), int(self.crop_size[1] / self.crop_pct))
        else:
            resize_size = int(self.crop_size / self.crop_pct)

        # Basic inference transform (equivalent to AutoImageProcessor)
        self.inference_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=self.interp_mode),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, images: Image.Image | list[Image.Image], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        """
        Process images and return in the same format as AutoImageProcessor.

        Args:
            images: Single PIL Image or list of PIL Images
            return_tensors: Format to return tensors ("pt" for PyTorch)

        Returns:
            Dictionary with 'pixel_values' key containing processed tensors
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Convert images to RGB and process
        processed_images = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed_img = self.inference_transform(img)
            processed_images.append(processed_img)

        # Stack tensors
        pixel_values = torch.stack(processed_images)

        return {"pixel_values": pixel_values}

    def preprocess(
        self, images: Image.Image | list[Image.Image], return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        """Alias for __call__ to match AutoImageProcessor interface."""
        return self.__call__(images, return_tensors)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create ImageProcessor from model name or path (similar to AutoImageProcessor.from_pretrained).

        Args:
            model_name_or_path: Name of the model or path to saved configuration
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            CustomImageProcessor instance configured for the model
        """
        # Remove 'use_fast' if present (compatibility with AutoImageProcessor calls)
        kwargs.pop("use_fast", None)

        # Check if it's a path to a saved configuration
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "preprocessor_config.json")
            if os.path.exists(config_file):
                return cls.from_config(config_file)

        # Otherwise, treat as model name
        return cls(model_name=model_name_or_path, **kwargs)

    def save_config(self, file_path: str):
        """Save current configuration to a JSON file."""
        config = {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size": self.size,
            "crop_pct": self.crop_pct,
            "interpolation": self.interpolation,
            "data_config_source": self.data_config_source,
        }

        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, file_path: str):
        """Load configuration from a JSON file."""
        with open(file_path) as f:
            config = json.load(f)

        return cls(custom_config=config)

    def save_pretrained(self, save_directory: str):
        """
        Save the processor configuration to a directory (compatible with Transformers).

        Args:
            save_directory: Directory to save the configuration
        """
        os.makedirs(save_directory, exist_ok=True)

        config_file = os.path.join(save_directory, "preprocessor_config.json")
        self.save_config(config_file)

        print(f"CustomImageProcessor configuration saved to {config_file}")

    @property
    def model_input_names(self):
        """Return the expected input names (compatibility with Transformers)."""
        return ["pixel_values"]

    def __repr__(self):
        """String representation of the processor."""
        return (
            f"CustomImageProcessor("
            f"image_mean={self.image_mean}, "
            f"image_std={self.image_std}, "
            f"size={self.size}, "
            f"crop_pct={self.crop_pct}, "
            f"interpolation='{self.interpolation}', "
            f"data_config_source='{self.data_config_source}')"
        )

    def to_dict(self):
        """Convert processor configuration to dictionary."""
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "size": self.size,
            "crop_pct": self.crop_pct,
            "interpolation": self.interpolation,
            "data_config_source": self.data_config_source,
        }

    def get_transform_for_training(
        self,
        random_resize_crop: bool = True,
        horizontal_flip: bool = True,
        color_jitter: bool = False,
        random_quarter_turns: bool = False,
        random_resized_crop_scale: tuple[float, float] = (0.08, 1.0),
    ) -> transforms.Compose:
        """
        Get training transform with data augmentation.

        Args:
            random_resize_crop: Whether to use random resize crop
            horizontal_flip: Whether to use random horizontal flip
            color_jitter: Whether to use color jitter
            random_quarter_turns: Whether to rotate by a random multiple of 90 degrees
            random_resized_crop_scale: Scale range for RandomResizedCrop

        Returns:
            Training transform pipeline
        """
        transform_list = []

        if random_resize_crop:
            if isinstance(self.crop_size, tuple):
                transform_list.append(
                    transforms.RandomResizedCrop(
                        self.crop_size,
                        scale=random_resized_crop_scale,
                        interpolation=self.interp_mode,
                    )
                )
            else:
                transform_list.append(
                    transforms.RandomResizedCrop(
                        (self.crop_size, self.crop_size),
                        scale=random_resized_crop_scale,
                        interpolation=self.interp_mode,
                    )
                )
        else:
            # Use resize + center crop for training if random crop is disabled
            resize_size = (
                int(self.crop_size / self.crop_pct)
                if isinstance(self.crop_size, int)
                else (int(self.crop_size[0] / self.crop_pct), int(self.crop_size[1] / self.crop_pct))
            )
            transform_list.extend(
                [transforms.Resize(resize_size, interpolation=self.interp_mode), transforms.CenterCrop(self.crop_size)]
            )

        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if random_quarter_turns:
            transform_list.append(RandomQuarterTurns())

        if color_jitter:
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))

        transform_list.extend([transforms.ToTensor(), self.normalize])

        return transforms.Compose(transform_list)

    def get_transform_for_validation(self) -> transforms.Compose:
        """Get validation transform (same as inference)."""
        return self.inference_transform

    def get_transform_for_test(self) -> transforms.Compose:
        """Get test transform (identical to validation / inference)."""
        return self.get_transform_for_validation()


# Convenience function to create processor
def create_image_processor(model_name: str, **kwargs) -> CustomImageProcessor:
    """
    Convenience function to create a CustomImageProcessor.

    Args:
        model_name: Name of the model
        **kwargs: Additional configuration parameters

    Returns:
        Configured CustomImageProcessor instance
    """
    return CustomImageProcessor.from_pretrained(model_name, **kwargs)
