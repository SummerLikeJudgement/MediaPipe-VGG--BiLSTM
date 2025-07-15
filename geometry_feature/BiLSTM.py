from torch import nn
import torch

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):# 特征维度， 隐藏状态/细胞状态的维度， 输出维度
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
            forward_outputs.append(forward_hidden_state.unsqueeze(1)) # (batch_size, 1, hidden_size)
        # 反向
        for i in reversed(range(seq_len)):
            combined = torch.cat((input[:, i, :], backward_hidden_state), dim=1)
            f_t = self.sigmoid(self.Wf(combined))
            i_t = self.sigmoid(self.Wi(combined))
            o_t = self.sigmoid(self.Wo(combined))
            c_hat_t = self.tanh(self.Wc(combined))
            backward_cell_state = f_t * backward_cell_state + i_t * c_hat_t
            backward_hidden_state = o_t * self.tanh(backward_cell_state)
            backward_outputs.insert(0, backward_hidden_state.unsqueeze(1)) # (batch_size, 1, hidden_size)

        forward_outputs = torch.cat(forward_outputs, dim=1)# (batch_size, seq_len, hidden_size)
        backward_outputs = torch.cat(backward_outputs, dim=1)# (batch_size, seq_len, hidden_size)
        outputs = torch.cat((forward_outputs, backward_outputs), dim=2) # (batch_size, seq_len, hidden_size*2)

        final_output = self.output_layer(outputs)# (batch_size, seq_len, outsize)
        return final_output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):# embed_size = num_heads*head_dim
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embed size needs to be divisible by heads"
        # qkv线性变换
        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)
        # 输出
        self.out = nn.Linear(embed_size, embed_size)
        self.scale = self.head_dim ** 0.5
    def forward(self, x):# x(batch_size, seq_len, features) features=head_dim*num_heads
        batch_size, seq_len, features = x.shape
        # QKV
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        # 划分多头 (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 注意力分数计算
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / self.scale # (batch_size, seq_len, num_heads, num_heads)
        attn_probs = torch.softmax(attn_score, dim=-1) # 对注意力分数归一化，纯数学操作，无参数
        attn_out = torch.matmul(attn_probs, V) # (batch_size, seq_len, num_heads, head_dim)
        # 合并多头
        attn_out = attn_out.contiguous().view(batch_size, seq_len, -1)
        # 输出映射
        out = self.out(attn_out)# (batch_size, seq_len, features)
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = BiLSTM(input_size=468*3, hidden_size=128, output_size=128)
        self.norm = nn.BatchNorm1d(128)
        self.attn = MultiHeadAttention(embed_size=128, num_heads=8)
        self.avgpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):# x(batch_size, seq_len, input_size/features_size)
        x = self.lstm(x) # (batch_size, 128, 468*3)->(batch_size, 128, 128)
        x = self.norm(x.transpose(1, 2))# (batch_size, 128, 128)->(batch_size, 128, 128)
        x = self.attn(x.transpose(1, 2))# (batch_size, 128, 128)->(batch_size, 128, 128)
        x = self.avgpool(x)# (batch_size, 128, 128)->(batch_size, 128, 1)
        return x

if __name__ == '__main__':
    model = LSTM()
    x = torch.randn(2, 128, 468*3)
    y = model(x)
    print(y.shape)
    print(model)