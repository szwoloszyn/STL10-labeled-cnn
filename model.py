import torch
import torch.nn as nn

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class CodyConvNet(nn.Module):
    def __init__(self, image_dim, n_classes, conv_neuron_rate=1, dense_neuron_rate=1, dropout=0.5, dropout2d=0.2):
        super().__init__()

        self.img_channels = 3
        self.kernel_size = 3
        self.pooling_kernel_size = 2
        self.stride = 1
        self.padding = "same"
        self.no_of_pools = 4
        self.dropout = dropout
        self.dropout2d = dropout2d
        self.cnr = conv_neuron_rate
        self.dnr = dense_neuron_rate
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=int(self.cnr*32), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), 
            nn.BatchNorm2d(int(self.cnr*32)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),

            nn.Conv2d(in_channels=int(self.cnr*32), out_channels=int(self.cnr*64), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), 
            nn.BatchNorm2d(int(self.cnr*64)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),

            nn.Conv2d(in_channels=int(self.cnr*64), out_channels=int(self.cnr*128), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), 
            nn.BatchNorm2d(int(self.cnr*128)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),

            nn.Conv2d(in_channels=int(self.cnr*128), out_channels=int(self.cnr*256), kernel_size=3, padding="same"),
            nn.BatchNorm2d(int(self.cnr*256)),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout2d),
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),


            nn.Flatten(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=int(self.cnr*256)*(image_dim//(2**self.no_of_pools))**2, out_features=int(self.dnr*128)),
            nn.ReLU(),

            nn.Dropout(p=max(self.dropout - 0.2, 0.0)),
            nn.Linear(in_features=int(self.dnr*128), out_features=int(self.dnr*64)),
            nn.ReLU(),
            nn.Dropout(p=max(self.dropout - 0.3, 0.0)),
            nn.Linear(in_features=int(self.dnr*64), out_features=n_classes)
        )
    def forward(self, x):
        return self.model(x)