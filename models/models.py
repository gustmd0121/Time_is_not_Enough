import torch 
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch import nn, Tensor
import numpy as np
from scipy.signal import stft, istft 
import random
import os
import math 

random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block ,layers, args, num_classes=3):
        super(ResNet1d, self).__init__()
        self.args = args
        self.inplanes = self.args.inplanes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels = self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.args.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        if self.args.use_transformer: 
            #Define transformer encoder layer 
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=512 * block.expansion * 2, 
                nhead=8, 
                dim_feedforward=2048,
                dropout=0.1
            )
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.args.num_layers)
        
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
            
        self.dropout = nn.Dropout(0.2)
        self.fs = self.args.fs 
        self.nperseg = self.args.nperseg
        self.noverlap = self.args.noverlap
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        data = x.cpu().detach().numpy()
        frequencies,times,Sxx = stft(data, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, boundary='zeros')
        img = torch.tensor(Sxx).to(device) 
        spectrogram = img.clone().detach() 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)

        if self.args.use_transformer: 
            x = self.transformer_encoder(x)
            
        x = self.fc(x)

        return x

class TransformerModel(nn.Module):
    def __init__(self, args, num_classes):
        super(TransformerModel, self).__init__()

        self.args = args
        
        #initial linear layer 
        self.initial_linear = None

        # Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.args.d_model, 
            nhead=self.args.nhead, 
            dim_feedforward=self.args.dim_feedforward, 
            dropout=self.args.dropout
        )

        # Stack n_layers of the Transformer Encoder
        self.pos_encoder = PositionalEncoding(self.args.d_model, self.args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.args.num_layers)

        # Fully connected layer
        self.fc = nn.Linear(self.args.d_model, num_classes)

    def forward(self, x):
        if self.initial_linear is None:
            self.initial_linear = nn.Linear(x.shape[2], self.args.d_model).to(x.device)

        x = self.initial_linear(x)
        
        x = x.transpose(0, 1)

        #positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate the output (e.g., mean across time dimension)
        x = x.mean(dim=0)

        # Fully connected layer
        x = self.fc(x)

        return x

class BiLSTMModel(nn.Module):
    def __init__(self, args, num_classes):
        super(BiLSTMModel, self).__init__()

        self.args = args

        # Bi-LSTM layer
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=self.args.d_model, 
            num_layers=self.args.num_layers, 
            batch_first=True, 
            bidirectional=True
        )

        # Fully connected layer
        # Output size is doubled for bidirectional LSTM
        self.fc = nn.Linear(2 * self.args.d_model, num_classes)

    def forward(self, x):
        
        x = x.transpose(1,2)
        # LSTM layer
        # We only use the output of the LSTM, not the hidden and cell states
        lstm_out, _ = self.lstm(x)

        #average of sequence length
        x = lstm_out.mean(dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x

def resnet34(*args, num_classes):
    model = ResNet1d(BasicBlock1d, [3,4,6,3], *args, num_classes)
    return model

def resnet_transformer(*args, num_classes):
    model = ResNet1d(BasicBlock1d, [3,4,6,3], *args, num_classes)
    return model

def transformer(*args, num_classes): 
    model = TransformerModel(*args, num_classes) 
    return model 

def BiLSTM(*args, num_classes): 
    model = BiLSTMModel(*args, num_classes) 
    return model 