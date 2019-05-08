import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision   
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

EPOCH = 1     
BATCH_SIZE = 50
LR = 0.001        
DOWNLOAD_MNIST = True  


train_data = torchvision.datasets.MNIST(
    root='D:/MINST/minst',   
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    
    download=False,          
)

test_data = torchvision.datasets.MNIST(root='D:/MINST/minst', train=False)
print('aaaa')
print(type(train_data))
print(len(train_data))
print('aaa')

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(16, 32, 5, 1, 2), 
            nn.ReLU(),  
            nn.MaxPool2d(2), 
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  
        output = cnn(b_x)         
        loss = loss_func(output, b_y) 
        optimizer.zero_grad()          
        loss.backward()                
        optimizer.step()    
        if step%100==0:          
        	test_output = cnn(test_x[:10])
        	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        	print(pred_y, 'prediction number')
        	print(test_y[:10].numpy(), 'real number') 

torch.save(cnn,'CNN.pkl')