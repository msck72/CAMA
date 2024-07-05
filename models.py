import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy
import numpy as np
from flwr.common import parameters_to_ndarrays


def create_model(cfg, model_rate, device= torch.device('cpu'))  :
    print('model being created')
    if(cfg.dataset == 'mnist'):
        return conv(model_rate, [1, 28, 28], 10, cfg.hidden_layers, device)
    elif(cfg.dataset == 'cifar10'):
        return conv(model_rate, [3, 32, 32], 10, cfg.hidden_layers, device)
    else:
        raise ValueError("Sorry no dataset_name is known")
    

def copy_gp_to_lp(global_parameters , local_values_shape):
    new_state_dict = {}
    i = 0
    for k , v in global_parameters.items():
        slices = [slice(0, dim) for dim in local_values_shape[i]]
        new_state_dict[k] = copy.deepcopy(v[slices])
        i += 1
    return new_state_dict


class Conv(nn.Module):
    """Convolutional Neural Network architecture."""

    def __init__(
        self,
        hidden_size,
        data_shape,
        classes_size,
    ):
        super().__init__()
        # self.model_config = model_config
        self.classes_size = classes_size

        blocks = [
            nn.Conv2d(
                data_shape[0], hidden_size[0], 3, 1, 1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        for i in range(len(hidden_size) - 1):
            blocks.extend(
                [
                    nn.Conv2d(
                        hidden_size[i], hidden_size[i + 1], 3, 1, 1,
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(
                    hidden_size[-1], classes_size
                ),
            ]
        )
        self.blocks = nn.Sequential(*blocks)


    def forward(self, input_dict):
        """Forward pass of the Conv.

        Parameters
        ----------
        input_dict : Dict
            Conatins input Tensor that will pass through the network.
            label of that input to calculate loss.
            label_split if masking is required.

        Returns
        -------
        Dict
            The resulting Tensor after it has passed through the network and the loss.
        """
        # output = {"loss": torch.tensor(0, device=self.device, dtype=torch.float32)}
        output = {}
        out = self.blocks(input_dict["img"])
        if "label_split" in input_dict:
            label_mask = torch.zeros(
                self.classes_size, device=out.device
            )
            label_mask[input_dict["label_split"]] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output["score"] = out
        output["loss"] = F.cross_entropy(out, input_dict["label"], reduction="mean")
        return output


def conv(
    model_rate,
    data_shape,
    classes_size,
    hidden_layers,
    device="cpu",
):
    """Create the Conv model."""
    hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_layers]
    model = Conv(hidden_size=hidden_size, data_shape=data_shape, classes_size=classes_size)
    return model.to(device)



def get_state_dict_from_param(model, parameters):
    # Load the parameters into the model
    for param_tensor, param_ndarray in zip(
        model.state_dict(), parameters_to_ndarrays(parameters)
    ):
        model.state_dict()[param_tensor].copy_(torch.from_numpy(param_ndarray))
    # Step 3: Obtain the state_dict of the model
    state_dict = model.state_dict()
    return state_dict