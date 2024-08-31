
from torchvision.datasets import MNIST 
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import transforms 
import matplotlib.pyplot as plt 
import torch.optim as optim 
import torch.nn.functional as F 
import torch

transform = transforms.Compose([transforms.ToTensor()])

trainset = MNIST(root="./data",download=True,train=True,transform=transform)
testset = MNIST(root="./data",download=True,train=True,transform=transform)

trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
testloader = DataLoader(testset,batch_size=64,shuffle=True)

trainiter = iter(trainloader)
images,labels = trainiter.next()
print(images.shape)
print(labels.shape)

index = 0
plt.imshow(images[index].numpy().squeeze())
plt.xlabel(labels[index])


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(784,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,64)
        self.final = nn.Linear(64,10)
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.final(x)
        return F.log_softmax(x,dim=1)
net = Net()
net

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# **training model**

for epoch in range(3):
    for data in trainloader:
        x,y = data 
        net.zero_grad()
        output = net(x.view(-1,784))
        loss = F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    print(loss)
                      
        

# evaluating test set 

total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        x,y = data 
        output = net(x.view(-1,784))
        for idx ,i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
            
print("Accuracy :" , round(correct/total,3))        
