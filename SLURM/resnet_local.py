import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from resnet import ResNet50
from tqdm import tqdm

if __name__=="__main__":
    batch_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    model = ResNet50().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    max_epochs = 10
    accuracy = {"train": [], "valid": []}
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch+1}")
        epoch_correct = 0
        epoch_all = 0
        for k, dataloader in loaders.items():
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum()
                all = len(y_batch)
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()                
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")
            accuracy[k].append(epoch_correct/epoch_all)

