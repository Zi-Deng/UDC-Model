import math
import torch
import torch.nn as nn
import json
from typing import Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


# Custom base classes to replace HuggingFace transformer base classes
class CustomConfig:
    """
    Custom configuration base class to replace transformers.PretrainedConfig
    """
    def __init__(self, **kwargs):
        # Standard configuration attributes that HuggingFace models expect
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.use_return_dict = kwargs.pop("use_return_dict", True)
        self.problem_type = kwargs.pop("problem_type", None)
        
        # Store any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and methods
            if not key.startswith('_'):
                if isinstance(value, (list, tuple)):
                    output[key] = list(value)
                elif hasattr(value, 'to_dict'):
                    output[key] = value.to_dict()
                else:
                    output[key] = value
        return output

    def to_json_string(self, use_diff=True):
        """
        Serializes this instance to a JSON string.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path, use_diff=True):
        """
        Save this instance to a JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiates a config from a Python dictionary of parameters.
        """
        return cls(**config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file):
        """
        Instantiates a config from the path to a JSON file of parameters.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_json_string(text)

    @classmethod
    def from_json_string(cls, json_string):
        """
        Instantiates a config from a JSON string of parameters.
        """
        config_dict = json.loads(json_string)
        return cls(**config_dict)


class CustomPreTrainedModel(nn.Module):
    """
    Custom base model class to replace transformers.PreTrainedModel
    """
    config_class = None
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = []

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        """
        Initialize weights and apply final processing.
        This is equivalent to the HuggingFace PreTrainedModel.post_init() method.
        """
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights. This should be overridden in subclasses.
        """
        pass


# Custom output classes to replace HuggingFace transformer output classes
class BaseModelOutputWithNoAttention:
    """
    Custom output class to replace transformers.modeling_outputs.BaseModelOutputWithNoAttention
    """
    def __init__(self, last_hidden_state=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
    
    def __getitem__(self, index):
        """Allow tuple-like indexing for backward compatibility"""
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")


class BaseModelOutputWithPoolingAndNoAttention:
    """
    Custom output class to replace transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention
    """
    def __init__(self, last_hidden_state=None, pooler_output=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states
    
    def __getitem__(self, index):
        """Allow tuple-like indexing for backward compatibility"""
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.pooler_output
        elif index == 2:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")


class ImageClassifierOutputWithNoAttention:
    """
    Custom output class to replace transformers.modeling_outputs.ImageClassifierOutputWithNoAttention
    """
    def __init__(self, loss=None, logits=None, hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
    
    def __getitem__(self, index):
        """Allow tuple-like indexing for backward compatibility"""
        if index == 0:
            return self.logits
        elif index == 1:
            return self.hidden_states
        else:
            raise IndexError(f"Index {index} out of range")

class ConvNextConfig(CustomConfig):
    model_type = "convnext"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
        image_size=224,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]

class ConvNextLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class ConvNextEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings
    
class ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = getattr(nn.functional, config.hidden_act)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
    

class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i]
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
    
class ConvNextPreTrainedModel(CustomPreTrainedModel):
    config_class = ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"
    _no_split_modules = ["ConvNextLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ConvNextModel(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    
class ConvNextForImageClassification(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
