# 코드 설명

## Import libraries

```
import torch
import torchvision
import torchvision.transforms as transforms
```

Python에서 PyTorch을 사용하기 위해 library을 import



## Dataset

```

transform = transforms.Compose(
    [transforms.ToTensor(), # tensor로 변환
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

ToTensor()을 사용하여 transform에 저장되는 배열을 tensor로 저장

CIFAR10에서 dataset을 다운로드

CIFAR10은 많이 사용하기 때문에 파이썬에서 다운로드 받아 저장하는 함수를 만들어 놓음

다른 dataset을 사용하기 위해서는 image을 다운로드 받아 저장하는 과정이 필요하다.



torch.utils.data.DataLoader() 함수는 데이터를 묶는 과정, 섞는 과정, 병령처리 과정에서 데이터를 불러오는 기능을 모두 제공해주는 iterator이다. 코드에서 trainset과 testset에 대해서 batch_size을 4로 설정하였고 trainset은 shuffle을 실행하고 testset에서는 shuffle을 실행하지 않았다.



CIFAR10에서 제공하는 dataset의 class은 10개로 plane, car, bird, cat, deer, dog, frog, horse, ship, truck 중 하나로 class가 결정된다.



# imshow 사진을 화면에 출력

```
# Define the function to show the image
import matplotlib.pyplot as plt
import numpy as np                

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

데이터는 tensor을 사용하여 다운로드 받았고 학습도 tensor을 사용하여 진행한다.

colab에서 화면에 데이터를 출력하기 위해 matplotlib과 numpy library을 import 한다.

화면에 사진을 출력하는 함수 imshow(img)을 정의



dataiter = iter(trainloader)에서 iter(trainloader)은 iterator을 반환한다. 간단하게 batch_size을 4로 설정한 trainset의 iterator을 반환하여 trainset 중에서 shuffle하여 4개의 sample을 dataiter에 저장한다. dataiter.next() 함수는 반복을 얻기 위해 iterator의 메소드 또는 다음 항목을 가져온다. 코드에서는 랜덤하게 사진을 출력하면서 이 함수를 사용한다.



imshow(torchvision.utils.make_grid(images)) 함수를 사용하면 images에 저장된 4개의 사진이 화면에 출력된다.



# Convolution Layer

```
# convolution layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)         
        self.conv2 = nn.Conv2d(6, 16, 5)      
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   
        self.fc2 = nn.Linear(120, 84)           
        self.fc3 = nn.Linear(84, 10)           

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

convolution layer을 class을 사용하여 정의한다. nn.Module은 계층과 output을 반환하는 forward(input) 메서드를 포함하고 있다.



init() 함수는 net = Net()과 같이 class를 선언했을 때 실행되는 함수로 class의 구조를 만든다. class을 선언하면 net에는 net.conv1, net.pool 등이 생성된다. nn.Conv2d은 2차원의 convolution 연산을 하는 함수로 매개변수는 각각 input channel, output channel, kernel size을 의미한다. Image은 r, g, b 3개의 channel을 가지고 있기 때문에 input channel은 3이 되고 output channel을 6, kernel size을 5로 설정하여 5x5 matrix의 kernel filter을 6개 사용한다.



nn.MaxPool2d(2, 2)은 2차원 matrix에서 2x2의 원소 중에서 최댓값만 남기는 함수이다.

<img src="https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png" width="500px" height="200px">



첨부한 사진에서 4x4 matrix을 2x2 maxpool에 입력하면 2x2 matrix가 출력된다. 입력된 matrix에서 2x2마다 최댓값만 남기고 나머지는 삭제하여 matrix의 크기를 1/2로 줄인다.



conv2에는 Conv2d(6, 16, 5)가 입력된다. Input channel은 conv1에서 출력된 6개이고 kernel size을 5, output channel의 개수를 16으로 설정한다. 



self.fc에는 nn.Linear을 사용한다. nn.Linear은 linear regression을 의미하고 input size은 16x5x5을 사용하고 출력은 120이 나오도록 한다. Input size은 kernel size가 5x5이므로 하나의 filter마다 25개의 원소가 존재하고 16개의 channel이 있기 때문에 총 400이 된다. fc1에서는 linear regression을 사용하여 400개의 input node와 120개의 output node가 연결된다. CIFAR10에서 class은 10개 중 하나로 결정되기 때문에 fc3에서 마지막 출력은 10이 된다.



# Optimizer

```
# optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()                               # cross entropy
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # gradient discent
```

criterion으로 crossentropyloss을 사용하고 optimizer로 gradient discent을 사용한다.



```
# training
# batch size is 4, 4장씩 1묶음으로 network로 넣으므로 100장을 돌리려면 25번을 돌려야 한다.
# 1장씩 넣으면 network가 불안정해진다.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):   # batch size을 위함
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

epoch은 학습을 반복하는 횟수를 나타내는 것으로 여기서는 총 2번 실행한다.

반복문에서 enumerate은 아래와 같이 tuple의 형태로 반환하는 함수이다.

![](C:\Users\user\Desktop\20201230_181929_1.png)

i에는 index에 반환되고 data에는 trainLoader의 값이 반환된다.

data에서 image data은 images에 저장하고 class은 labels에 저장한다.

optimizer을 초기화 한 후 class에 inputs을 입력하여 outputs을 구한 후 crossentropyloss을 loss에 저장한다.

loss을 backpropagraion을 실행한 후 optimizier을 실행한다. running_loss은 반복문을 실행할 때마다 더한 후 2000번에 한 번씩 화면에 출력한다. 실행한 결과 학습이 진행되어도 running_loss에 큰 변화가 나타나지 않았다.



# Save weight

```
# save weight
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```



# Test image

```
net = Net()
net.load_state_dict(torch.load(PATH))

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

net을 초기화 한 후 iterater을 testloader에 대한 iterater로 정의한다.



# Test

```
# Accuracy 55%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Accuracy by class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

전체 testset에 대한 Accracy와 class마다 Accracy을 출력한다.

![](C:\Users\user\Desktop\20201230_203949.png)

![](C:\Users\user\Desktop\20201230_203941.png)

epoch가 반복되는 횟수를 2에서 10으로 변경해도 위의 Accuracy에서 크게 달라지지 않았다.



# colab 주소

앞의 코드를 실행한 colab 주소

[colab] https://colab.research.google.com/drive/1vHVmE2qVkxlD0MFabYwIeeFzTaziaSLj



