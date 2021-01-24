# %%
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import model
from tqdm import tqdm
import random
import time
import torchsummary
NUM_WORKERS = 8
BATCH_SIZE = 128
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.GaussianBlur(3, 3),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# %%
train_dataset = CIFAR10(root='./data', download=True, transform=train_transform, train=True)
train_dataset.data = train_dataset.data[:10_000]
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# %%
def train_step(batch, model, optimizer, loss_fn):
    inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    return loss


# %% =================================================================================================
base_model = model.BaseModel(num_classes=10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
base_t_loss = []
base_model.to(DEVICE)

start = time.time()
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}')
    t_loss = []

    for batch in tqdm(train_loader):
        loss = train_step(batch, base_model, optimizer, loss_fn)
        batch_loss = loss.detach().cpu().numpy().item()
        t_loss.append(batch_loss)

    print(f'train loss: {np.mean(t_loss):.4f}')
    base_t_loss.append(np.mean(t_loss))
print(f'Time to train: {time.time() - start}')


# %% =================================================================================================
atrous_model = model.AtrousModel(num_classes=10)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(atrous_model.parameters(), lr=1e-4)
atrous_t_loss = []
atrous_model.to(DEVICE)

start = time.time()
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}')
    t_loss = []

    for batch in tqdm(train_loader):
        loss = train_step(batch, atrous_model, optimizer, loss_fn)
        batch_loss = loss.detach().cpu().numpy().item()
        t_loss.append(batch_loss)

    print(f'train loss: {np.mean(t_loss):.4f}')
    atrous_t_loss.append(np.mean(t_loss))

print(f'Time to train: {time.time() - start}')


# %%
x = np.arange(len(atrous_t_loss))
plt.title('Conv. 7x7    -    Conv. Dilated 3')
plt.plot(x, base_t_loss, label='7x7 train loss')
plt.plot(x, atrous_t_loss, label='Atrous train loss')
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.legend()
plt.savefig('loss.jpg')
