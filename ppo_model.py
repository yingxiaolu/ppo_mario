import torch.nn as nn
import torch.nn.functional as F
class PPO(nn.Module):
    '''
    return self.actor_linear(x), self.critic_linear(x)
    '''
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) 
        self.linear = nn.Linear(32*6*6, 512)
        

        self.afc1 = nn.Linear(512, 512)
        self.afc2=nn.Linear(512, 512)
        self.afc3=nn.Linear(512, num_actions)
        
        self.cfc1 = nn.Linear(512, 512)
        self.cfc2=nn.Linear(512, 512)
        self.cfc3=nn.Linear(512, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        #随机初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        # x = self.maxpool(F.relu(self.conv1(x)))
        # x = self.maxpool(F.relu(self.conv2(x)))
        # x = self.maxpool(F.relu(self.conv3(x)))
        # x = self.maxpool(F.relu(self.conv4(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        # ax=F.relu(self.afc1(x))
        # ax=F.relu(self.afc2(ax))
        ax=self.afc3(x)
        # cx=F.relu(self.cfc1(x))
        # cx=F.relu(self.cfc2(cx))
        cx=self.cfc3(x)
        
        return ax,cx