import torch
import torch.nn as nn


class HighwayEncoding(nn.Module):

    def __init__(self,
                 data,
                 input_size,
                 gate_bias=-2,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayEncoding, self).__init__()

        self.gpu = data.HP_gpu

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

        if self.gpu:
            self.normal_layer = self.normal_layer.cuda()
            self.gate_layer = self.gate_layer.cuda()


    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)