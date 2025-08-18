# ConvNext Model Integration

This document describes the integration of ConvNext models into the UDC-Model repository.

## Overview

The repository now supports both ResNet and ConvNext models for image classification tasks. The models are implemented using custom base classes that replace HuggingFace transformers dependencies, making them fully compatible with the existing training infrastructure.

## Changes Made

### 1. ConvNext Model Implementation (`model/convnext.py`)

- **Custom Base Classes**: Implemented custom `CustomConfig`, `CustomPreTrainedModel`, and output classes to replace HuggingFace dependencies
- **ConvNext Architecture**: Full implementation including:
  - `ConvNextConfig`: Configuration class with model parameters
  - `ConvNextLayerNorm`: Custom layer normalization supporting different data formats
  - `ConvNextEmbeddings`: Patch embedding layer
  - `ConvNextLayer`: Individual ConvNext block with depthwise convolution
  - `ConvNextStage`: Stage composed of multiple ConvNext layers
  - `ConvNextEncoder`: Full encoder with multiple stages
  - `ConvNextModel`: Base model with pooling
  - `ConvNextForImageClassification`: Classification head with loss computation

### 2. Training Script Updates (`train.py`)

- **Model Selection**: Added support for choosing between ResNet and ConvNext models via `model_type` parameter
- **Conditional Model Creation**: Models are instantiated based on the `model_type` configuration
- **Import Updates**: Added ConvNext model imports

### 3. Configuration Updates (`utils/utils.py`)

- **New Parameter**: Added `model_type` field to `ScriptTrainingArguments` dataclass
- **Validation**: Support for 'resnet' and 'convnext' model types

### 4. Example Configuration (`config/convnext_example.json`)

- **ConvNext Config**: Example configuration file showing how to use ConvNext models
- **Parameter Examples**: Demonstrates proper parameter values for ConvNext training

## Usage

### Using ConvNext Models

To use a ConvNext model instead of ResNet, set the `model_type` parameter in your configuration JSON:

```json
{
    "model_type": "convnext",
    "model": "facebook/convnext-tiny-224",
    "num_labels": 3,
    ...
}
```

### Training Command

```bash
python train.py --config config/convnext_example.json
```

### Model Types Supported

- `"resnet"`: Uses ResNet architecture (default)
- `"convnext"`: Uses ConvNext architecture

### ConvNext Configuration Parameters

Key parameters specific to ConvNext:

- `num_channels`: Input image channels (default: 3)
- `patch_size`: Patch size for embeddings (default: 4)
- `num_stages`: Number of encoder stages (default: 4)
- `hidden_sizes`: Channel dimensions for each stage (default: [96, 192, 384, 768])
- `depths`: Number of layers in each stage (default: [3, 3, 9, 3])
- `hidden_act`: Activation function (default: "gelu")
- `layer_scale_init_value`: Layer scale initialization (default: 1e-6)

## Architecture Details

### ConvNext vs ResNet

| Feature | ResNet | ConvNext |
|---------|--------|----------|
| **Convolution Type** | Standard + Bottleneck | Depthwise Separable |
| **Normalization** | BatchNorm | LayerNorm |
| **Activation** | ReLU | GELU |
| **Architecture** | Residual Blocks | Inverted Bottlenecks |
| **Default Channels** | [256, 512, 1024, 2048] | [96, 192, 384, 768] |

### Custom Implementation Benefits

1. **No HuggingFace Dependency**: Fully self-contained implementation
2. **Custom Loss Functions**: Compatible with existing loss function framework
3. **Consistent Interface**: Same training pipeline for both architectures
4. **Weight Loading**: Support for loading pretrained weights with filtering

## Backward Compatibility

All existing ResNet configurations will continue to work without modification. The default `model_type` is "resnet" to maintain backward compatibility.

## Example Usage

### ResNet (Existing)
```json
{
    "model_type": "resnet",
    "model": "microsoft/resnet-50",
    "num_labels": 3
}
```

### ConvNext (New)
```json
{
    "model_type": "convnext", 
    "model": "facebook/convnext-tiny-224",
    "num_labels": 3
}
```

Both configurations will work with the same training script and infrastructure.