import torch
import torch.nn as nn

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, h_size):
        super(LSTMCell, self).__init__()

        combined = input_size + h_size

        self.w_f = nn.Linear(combined, h_size)
        self.w_i = nn.Linear(combined, h_size)
        self.w_c = nn.Linear(combined, h_size)
        self.w_o = nn.Linear(combined, h_size)
    
    def forward(self, input, hidden, cell):
        combined = torch.cat([input, hidden], dim = 1)

        f = torch.sigmoid(self.w_f(combined))
        i = torch.sigmoid(self.w_i(combined))
        c_hat = torch.tanh(self.w_c(combined))
        o = torch.sigmoid(self.w_o(combined))

        cell = f * cell + i * c_hat

        h = o * torch.tanh(cell)

        return h, cell
