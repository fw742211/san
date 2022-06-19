import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, len_ts):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(len_ts, int(0.75 * len_ts))
        self.linear2 = nn.Linear(int(0.75 * len_ts), int(0.5 * len_ts))
        self.linear3 = nn.Linear(int(0.5 * len_ts), int(0.25 * len_ts))

        self.dropout = nn.Dropout(0.0)
        self.act1 = nn.GELU()

    def forward(self, x):
        out = self.linear1(x)
        # out = torch.relu(out)
        out = self.linear2(out)
        # out = torch.relu(out)
        out = self.linear3(out)
        out = torch.relu(out)
        # out = self.act1(out)
        return out


class Decoder(nn.Module):
    def __init__(self, len_ts):
        super(Decoder, self).__init__()
        self.linear2 = nn.Linear(int(0.25 * len_ts), int(0.5 * len_ts))
        self.linear3 = nn.Linear(int(0.5 * len_ts), int(0.75 * len_ts))
        self.linear4 = nn.Linear(int(0.75 * len_ts), len_ts)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        out = self.linear2(x)
        out = self.linear3(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)

        return out


class Adaptation_Network(nn.Module):
    def __init__(self, len_ts):
        super(Adaptation_Network, self).__init__()
        self.encoder = Encoder(len_ts)
        self.decoder = Decoder(len_ts)

    def forward(self, x):
        latent_space = self.encoder(x)
        y = self.decoder(latent_space)
        return y, latent_space


class Discrimator(nn.Module):
    def __init__(self, len_ts):
        super(Discrimator, self).__init__()
        self.linear1 = nn.Linear(len_ts, int(0.75 * len_ts))
        self.linear2 = nn.Linear(int(0.75 * len_ts), int(0.5 * len_ts))
        self.linear3 = nn.Linear(int(0.5 * len_ts), 1)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()

        self.dropout = nn.Dropout(0.0)


    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        # out = self.act1(out)
        out = self.linear2(out)
        out = torch.relu(out)
        # out = self.act2(out)
        out = self.linear3(out)
        out = torch.relu(out)
        # out = self.act3(out)
        out = self.dropout(out)
        return out


