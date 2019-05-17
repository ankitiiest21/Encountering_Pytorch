import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)                              # 1 Input Image Channel, 6 Output Channel, 5*5 Square Convolution. 
        self.conv2 = nn.Conv2d(6,16,5)                             # 6 Input Image Channel, 16 Output Channel, 5*5 Square Convolution.
        self.fc1 = nn.Linear(16*5*5,120)                           # Fully Connected Layer 1
        self.fc2 = nn.Linear(120,84)                               # Fully Connected Layer 2
        self.fc3 = nn.Linear(84,10)                                # Fully Connected Layer 3(with the Output)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))              # Max Pooling over a 2*2 Window
        x = F.max_pool2d(F.relu(self.conv2(x)),2)                  # Max Pooling over a Square of dimension 2
        x = x.view(-1,self.num_flat_features(x))                   # Reshaping the Tensor x with the "Flattened Features"
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):         
        size = x.size()[1:]                                       # If x is a 25x3x32x32 Tensor(an image) , then size would be 3x32x32
        num_features = 1                                          
        for s in size:
            num_features*=s
        return num_features                                       # num_features would be 3x32x32 = 3072(Total Number of Pixels in that Image).
                                                                  # Flat_features are “Flattened Features”.
net = Net()
print(net)

params = list(net.parameters())                                   # net.parameters() returns the Learnable Parameters
print(len(params))
print(params[0].size())                                           # Weight of the Conv1 Layer                         
      
      
inputz = torch.randn(1,1,32,32)                                   # LeNet Input Example
out = net(inputz)

output = net(inputz)                               
target = torch.randn(10)                                          # Creates a Random tensor of 10*1 dimensions where each element belongs to (0,1]
target = target.view(1,-1)                                        # "-1" argument in the view(1,-1) specifies number of Columns to be unspecified(n(Row)=1)
criterion = nn.MSELoss()
loss = criterion(output,target)                                   # Loss function takes (Output,Target) pair of inputs
print(loss)

net.zero_grad()                                                   # Gradient Buffers of all Parameters set to Zero
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)                                        # CAUTION!!! Here "None" will be displayed as unless and until we backpropagate we cant make the Gradient Zero.
loss.backward()                                                   # BackPropagate The Error 
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
