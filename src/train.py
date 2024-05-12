import torch
import torch.nn as nn
from model import TOModel
from hyperparams import HyperParams
from lossfn import CustomLossFN
from dataset import TODataset, uniform_sampler, random_d4_transform
from tqdm import tqdm


print("TRAINING ON: ", HyperParams.DEVICE)
model = TOModel()
model = model.to(HyperParams.DEVICE)

dataset = TODataset(HyperParams.DATA_PATH, uniform_sampler(), transforms=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=HyperParams.BATCH_SIZE, shuffle=True)

criterion = CustomLossFN()
optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams.LR)

best_loss = np.inf

for epoch in range(HyperParams.N_EPOCHS):
    epoch_loss = 0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = data['iters'], data['targets']
        # print(inputs.shape, labels.shape)
        inputs, labels = inputs.to(HyperParams.DEVICE), labels.to(HyperParams.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Iter: {i}, Loss: {epoch_loss/len(dataloader)}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")