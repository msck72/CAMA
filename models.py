import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy
import numpy as np
from flwr.common import parameters_to_ndarrays


def create_model(cfg, model_rate, device= torch.device('cpu'))  :
    # print('model being created')
    if(cfg.dataset == 'mnist'):
        return conv(model_rate, [1, 28, 28], 10, cfg.hidden_layers, device)
    elif(cfg.dataset == 'cifar10'):
        # print(cfg.hidden_layers)
        # return conv(model_rate, [3, 32, 32], 10, cfg.hidden_layers, device)
        return create_ResNet18(cfg.hidden_layers, model_rate, device)
    else:
        raise ValueError("Sorry no dataset_name is known")
    

def copy_gp_to_lp(global_parameters , local_values_shape):
    new_state_dict = {}
    i = 0
    j = 0
    # for _, v1 in global_parameters.items():
    #     print(f'{v1.shape},   {local_values_shape[j]}')
    #     j+= 1
    
    # print("enetring copying")
    for k , v in global_parameters.items():
        # print(f'{v.shape},    {local_values_shape[i]}')
        if v.shape != torch.Size([]):
            slices = [slice(0, dim) for dim in local_values_shape[i]]
            new_state_dict[k] = copy.deepcopy(v[slices])
        else:
            new_state_dict[k] = copy.deepcopy(v)
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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, hidden_layers , num_blocks, num_classes=10, model_rate=1):
        super(ResNet, self).__init__()
        self.in_planes = hidden_layers[0]
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, hidden_layers[0], kernel_size=7, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_layers[0])
        self.layer1 = self._make_layer(block, hidden_layers[0] , num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_layers[1] , num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_layers[2] , num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_layers[3] , num_blocks[3], stride=2)
        self.linear = nn.Linear(int(hidden_layers[3] * block.expansion), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, int(planes), stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict['img']
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # output = {}
        if "label_split" in input_dict:
            label_mask = torch.zeros(
                self.num_classes, device=out.device
            )
            label_mask[input_dict["label_split"]] = 1
            out = out.masked_fill(label_mask == 0, 0)
        # output['score'] = out
        # output['loss'] = F.cross_entropy(output["score"], input_dict["label"])
        # return output
        return out

def create_ResNet18(hidden_layers, model_rate = 1, device = "cpu"):
    hidden_layers = [int(layer * model_rate) for layer in hidden_layers]
    # print("is it float64??")
    # print(type(hidden_layers))
    # print(hidden_layers[0])
    return ResNet(BasicBlock, hidden_layers=hidden_layers, num_blocks=[2, 2, 2, 2], model_rate=model_rate).to(device)


def get_state_dict_from_param(model, parameters):
    # Load the parameters into the model
    for param_tensor, param_ndarray in zip(
        model.state_dict(), parameters_to_ndarrays(parameters)
    ):
        model.state_dict()[param_tensor].copy_(torch.from_numpy(param_ndarray))
    # Step 3: Obtain the state_dict of the model
    state_dict = model.state_dict()
    return state_dict