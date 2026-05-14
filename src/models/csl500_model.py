import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class EnhancedGCN(nn.Module):
    def __init__(self, config):
        super(EnhancedGCN, self).__init__()
        self.input_proj = nn.Linear(config.num_joints, config.num_joints * 2)
        self.gcn1 = GCNConv(config.num_joints * 2, config.hidden_dim)
        self.gcn2 = GCNConv(config.hidden_dim, config.hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim)
        self.edge_index = config.pose_edge_index
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual = ResidualBlock(config.hidden_dim, config.hidden_dim)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x = x.view(batch_size * seq_len, num_features)
        x = self.input_proj(x)  # 将17维特征映射到34维
        edge_index = self.edge_index.repeat(1, batch_size * seq_len).to(x.device)

        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(batch_size, seq_len, -1).transpose(1, 2)
        x = self.residual(x)
        x = x.transpose(1, 2)

        return x

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.residual = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.residual(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x.transpose(0, 1)

class IntermediateCSL500Model(nn.Module):
    def __init__(self, config):
        super(IntermediateCSL500Model, self).__init__()
        self.gcn = EnhancedGCN(config)

        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(config.hidden_dim, config.hidden_dim * 2),
            TemporalConvBlock(config.hidden_dim * 2, config.hidden_dim * 4)
        )

        self.lstm = nn.LSTM(config.hidden_dim * 4, config.hidden_dim, batch_first=True, bidirectional=True)

        self.attention = MultiHeadAttention(config.hidden_dim * 2, num_heads=8)

        self.fc1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)

    def forward(self, x, lengths):
        # GCN
        x = self.gcn(x)

        # Temporal convolution
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)

        # Adjust lengths for temporal convolution
        lengths = torch.div(lengths, 4, rounding_mode='floor')

        # Pack padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM
        x, _ = self.lstm(x)

        # Unpack sequence
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = self.layer_norm(x)

        # Self-attention
        x = self.attention(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Test code
if __name__ == "__main__":
    from config.config import Config

    config = Config()
    model = IntermediateCSL500Model(config)

    batch_size = 16
    seq_len = 36
    num_joints = 17
    input_dim = 2

    x = torch.randn(batch_size, seq_len, num_joints * input_dim)
    lengths = torch.randint(10, seq_len, (batch_size,))

    output = model(x, lengths)
    print(f"Output shape: {output.shape}")