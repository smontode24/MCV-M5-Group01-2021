from torch import nn

class RandomNet:
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        
        # Input block
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=2, padding=1, use_bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1, use_bias=False)(x)
        self.bn2 = nn.BatchNorm2d(32)(x)

        # Residual block 1
        self.conv2_residual = nn.Conv2D(32, 64, kernel_size=3, stride=2, padding=1, use_bias=False, use_bias=False)(x)
        self.bn2_residual = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        # Residual block 2
        self.conv4_residual = nn.Conv2D(64, 128, kernel_size=3, stride=2, padding=1, use_bias=False)(x)
        self.bn4_residual = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2D(128, 128, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2D(128, kernel_size=3, stride=2, padding=1, use_bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2D(128, kernel_size=3, stride=2, padding=1, use_bias=False)
        self.bn8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2D(128, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.conv10 = nn.Conv2D(256, kernel_size=3, stride=2, padding=1, use_bias=False)
        self.bn10 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(256, 10)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def __forward__(self, x):
        # Input block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Residual block 1
        residual = self.conv2_residual(x)
        residual = self.bn2_residual(residual)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        
        x = self.maxpool(x)
        x = x + residual

        # Residual block 2
        residual = self.conv4_residual(x)
        residual = self.bn4_residual(residual)
        
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)

        x = self.maxpool(x)
        x = x + residual

        x = F.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)

        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)

        # Global avg pooling
        x = F.adaptive_avg_pool2d(x, 1).view(-1, 256)

        x = self.dropout(x)
        x = self.fc(x)
        return x