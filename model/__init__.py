"""
Model package — shared base classes for custom vision model architectures.

Provides HuggingFace-compatible base classes for model configuration and output
types, decoupled from the transformers library. These shared classes are used by
both the ResNet (model.ResNet) and ConvNeXt (model.convnext) implementations.

Classes:
    CustomConfig: Base configuration class replacing transformers.PretrainedConfig.
    CustomPreTrainedModel: Base model class replacing transformers.PreTrainedModel.
    BaseModelOutputWithNoAttention: Output type with last_hidden_state.
    BaseModelOutputWithPoolingAndNoAttention: Output type with pooler_output.
    ImageClassifierOutputWithNoAttention: Output type for image classification.
"""

import json

import torch.nn as nn


class CustomConfig:
    """Base configuration class replacing transformers.PretrainedConfig.

    Provides serialization to/from dict, JSON string, and JSON file,
    along with standard HuggingFace config attributes (output_hidden_states,
    use_return_dict, problem_type).
    """

    def __init__(self, **kwargs):
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.use_return_dict = kwargs.pop("use_return_dict", True)
        self.problem_type = kwargs.pop("problem_type", None)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                if isinstance(value, (list, tuple)):
                    output[key] = list(value)
                elif hasattr(value, "to_dict"):
                    output[key] = value.to_dict()
                else:
                    output[key] = value
        return output

    def to_json_string(self, use_diff=True):
        """Serialize this instance to a JSON string."""
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path, use_diff=True):
        """Save this instance to a JSON file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Instantiate a config from a Python dictionary of parameters."""
        return cls(**config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        """Instantiate a config from the path to a JSON file of parameters."""
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_json_string(text)

    @classmethod
    def from_json_string(cls, json_string):
        """Instantiate a config from a JSON string of parameters."""
        config_dict = json.loads(json_string)
        return cls(**config_dict)


class CustomPreTrainedModel(nn.Module):
    """Base model class replacing transformers.PreTrainedModel.

    Provides weight initialization via post_init() and standard HuggingFace
    model attributes (config_class, base_model_prefix, main_input_name).
    """

    config_class = None
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = []

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        """Initialize weights and apply final processing.

        Equivalent to the HuggingFace PreTrainedModel.post_init() method.
        """
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights. Override in subclasses for architecture-specific init."""
        pass


class BaseModelOutputWithNoAttention:
    """Output type with last_hidden_state and optional hidden_states tuple.

    Replaces transformers.modeling_outputs.BaseModelOutputWithNoAttention.
    Supports tuple-like indexing for backward compatibility.
    """

    def __init__(self, last_hidden_state=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")


class BaseModelOutputWithPoolingAndNoAttention:
    """Output type with last_hidden_state, pooler_output, and optional hidden_states.

    Replaces transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention.
    Supports tuple-like indexing for backward compatibility.
    """

    def __init__(self, last_hidden_state=None, pooler_output=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.pooler_output
        elif index == 2:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")


class ImageClassifierOutputWithNoAttention:
    """Output type for image classification with loss, logits, and optional hidden_states.

    Replaces transformers.modeling_outputs.ImageClassifierOutputWithNoAttention.
    Supports tuple-like indexing for backward compatibility.
    """

    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index == 0:
            return self.logits
        elif index == 1:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")
