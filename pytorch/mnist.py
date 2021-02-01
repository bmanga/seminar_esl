import torch
import torch.functional as F
from net_model import Net

import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

transformer = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                                 ])

mnist_data_train = torchvision.datasets.MNIST("../datasets/", transform=transformer, train=True, download=True)
mnist_data_test = torchvision.datasets.MNIST("../datasets/", transform=transformer, train=False, download=True)

train_loader = DataLoader(mnist_data_train, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(mnist_data_test, batch_size=batch_size_test, shuffle=True)

# Look at some examples

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Net().to(device)
loss_fn = torch.nn.CrossEntropyLoss()


# Train the model
model.train()
params = [p for p in model.parameters() if p.requires_grad]

for epoch in range(3):
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    learning_rate *= 0.3
# Evaluate the model
model.eval()
loss = 0
correct = 0
with torch.no_grad():
    for idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        curr_loss = loss_fn(out, target).item()
        loss += curr_loss

        pred = out.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).sum()
        print("\rLoss: {}".format(curr_loss))

loss /= len(test_loader.dataset)
print("Average loss: {}".format(loss))
print("Accuracy: {}", correct / len(test_loader.dataset))








