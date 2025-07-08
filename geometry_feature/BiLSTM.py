from torch import nn
import torch

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):# 特征维度， seq_len， 输出维度
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size * 2, output_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.backward_h0 = nn.Parameter(torch.randn(1, hidden_size)) # 反向隐藏状态初始化
        self.backward_c0 = nn.Parameter(torch.randn(1, hidden_size)) # 反向细胞状态初始化

    def forward(self, input):# (batch_size, seq_len, input_size/features_size)
        batch_size = input.size(0)
        seq_len = input.size(1)

        # 前向状态初始化
        forward_hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        forward_cell_state = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        # 反向状态初始化，通过可学习参数生成，并通过 expand 扩展到当前批次大小。
        backward_hidden_state = self.backward_h0.expand(batch_size, -1)
        backward_cell_state = self.backward_c0.expand(batch_size, -1)
        forward_outputs = []
        backward_outputs = []
        # 前向
        for i in range(seq_len):
            combined = torch.cat((input[:, i, :], forward_hidden_state), dim=1)
            f_t = self.sigmoid(self.Wf(combined))
            i_t = self.sigmoid(self.Wi(combined))
            o_t = self.sigmoid(self.Wo(combined))
            c_hat_t = self.tanh(self.Wc(combined))
            forward_cell_state = f_t * forward_cell_state + i_t * c_hat_t
            forward_hidden_state = o_t * self.tanh(forward_cell_state)
            forward_outputs.append(forward_hidden_state.unsqueeze(1))
        # 反向
        for i in reversed(range(seq_len)):
            combined = torch.cat((input[:, i, :], backward_hidden_state), dim=1)
            f_t = self.sigmoid(self.Wf(combined))
            i_t = self.sigmoid(self.Wi(combined))
            o_t = self.sigmoid(self.Wo(combined))
            c_hat_t = self.tanh(self.Wc(combined))
            backward_cell_state = f_t * backward_cell_state + i_t * c_hat_t
            backward_hidden_state = o_t * self.tanh(backward_cell_state)
            backward_outputs.insert(0, backward_hidden_state.unsqueeze(1))

        forward_outputs = torch.cat(forward_outputs, dim=1)# (batch_size, seq_len, features)
        backward_outputs = torch.cat(backward_outputs, dim=1)# (batch_size, seq_len, features)
        outputs = torch.cat((forward_outputs, backward_outputs), dim=2)

        final_output = self.output_layer(outputs)
        return final_output, (forward_hidden_state, backward_hidden_state)

class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = BiLSTM(input_size=468*3, hidden_size=128, output_size=468)