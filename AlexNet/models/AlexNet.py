import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, LRN_size=5, LRN_alpha=1e-4, LRN_beta=0.75, LRN_k=2):
        super(AlexNet, self).__init__()
        self.num_classes = 1000

        self.LRN_size = LRN_size
        self.LRN_alpha = LRN_alpha
        self.LRN_beta = LRN_beta
        self.LRN_k = LRN_k

        self.conv1_1 = self.conv_block(3, 48, 11, 4, 0, True, False)
        self.conv1_2 = self.conv_block(3, 48, 11, 4, 0, True, False)

        self.conv2_1 = self.conv_block(48, 128, 5, 1, 2, True, True)
        self.conv2_2 = self.conv_block(48, 128, 5, 1, 2, True, True)

        self.conv3 = self.conv_block(256, 384, 3, 1, 1, False, True)

        self.conv4_1 = self.conv_block(192, 192, 3, 1, 1, False, False)
        self.conv4_2 = self.conv_block(192, 192, 3, 1, 1, False, False)

        self.conv5_1 = self.conv_block(192, 128, 3, 1, 1, False, True)
        self.conv5_2 = self.conv_block(192, 128, 3, 1, 1, False, True)

        self.fc1 = self.fc_block((6 * 6 * 256), 4096)

        self.fc2 = self.fc_block(4096, self.num_classes)

        self.fc3 = nn.Linear(self.num_classes, 2)

    def forward(self, x):
        x1_1 = self.conv1_1(x)
        x1_2 = self.conv1_2(x)

        x2_1 = self.conv2_1(x1_1)
        x2_2 = self.conv2_2(x1_2)

        x2 = torch.cat([x2_1, x2_2], dim=1)
        x_3 = self.conv3(x2)

        x3_1, x3_2 = torch.split(x_3, split_size_or_sections=192, dim=1)
        x4_1 = self.conv4_1(x_3)
        x4_2 = self.conv4_1(x_3)

        x5_1 = self.conv5_1(x4_1)
        x5_2 = self.conv5_1(x4_2)

        x_5 = torch.cat([x5_1, x5_2], dim=1)
        x_5 = x_5.view(-1, 256 * 6 * 6)

        x_6 = self.fc1(x_5)

        x_7 = self.fc2(x_6)

        logit = self.fc3(x_7)
        return logit

    def init_weights(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        nn.init.constant_(self.conv1_1[0].bias, 0)
        nn.init.constant_(self.conv1_2[0].bias, 0)

        nn.init.constant_(self.conv2_1[0].bias, 1)
        nn.init.constant_(self.conv2_2[0].bias, 1)

        nn.init.constant_(self.conv3[0].bias, 0)

        nn.init.constant_(self.conv4_1[0].bias, 1)
        nn.init.constant_(self.conv4_2[0].bias, 1)

        nn.init.constant_(self.conv5_1[0].bias, 1)
        nn.init.constant_(self.conv5_2[0].bias, 1)

        nn.init.constant_(self.fc1[0].bias, 1)
        nn.init.constant_(self.fc2[0].bias, 1)
        nn.init.constant_(self.fc3.bias, 1)

    def conv_block(self, in_c, out_c, kernel_size, stride, padding, LRN, Pool):
        modules = [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)]

        if LRN:
            modules.append(nn.LocalResponseNorm(self.LRN_size, self.LRN_alpha, self.LRN_beta, self.LRN_k))
            if Pool:
                modules.append(nn.MaxPool2d(kernel_size=3, stride=2))

        modules.append(nn.ReLU())
        return nn.Sequential(*modules)

    def fc_block(self, in_f, out_f):
        modules = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        return modules


aa = AlexNet()
aa.init_weights()
