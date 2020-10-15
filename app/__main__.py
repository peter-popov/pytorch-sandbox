import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize

from tqdm import tqdm

from app.datasets import SampleDataset
from app.models import SampleModel, SampleVggStyle


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Running on GPU!")
        torch.backends.cudnn.benchmark = True

    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SampleDataset(args.data, train=True, transform=transform)

    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, shuffle=False)

    torch.autograd.set_detect_anomaly(True)

    model = SampleModel(len(dataset.classes))
    model = model.to(device)
    model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters {}".format(total_params))

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        train_loss = train(model, criterion, optimizer, device,
                           dataset=train_dataset, dataloader=train_loader)

        val_loss = validate(model, criterion, device,
                            dataset=val_dataset, dataloader=val_loader)

        print("train loss: {:.4f}, val loss: {:.4f}".format(train_loss, val_loss))

    torch.save(model, "model.pt")

    # testset = SampleDataset(args.data, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=args.batch_size, 
    #     num_workers=args.num_workers, shuffle=False)
    # acc = test(model, device, testset, testloader)

    # print("Model accuracy: {:.4f}".format(acc))



def train(model, criterion, optimizer, device, dataset, dataloader):
    model.train()

    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, "train"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)

    return epoch_loss


def validate(model, criterion, device, dataset, dataloader):
    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, "  val"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)

    return epoch_loss

def test(model, device, dataset, dataloader):
    model.eval()

    correct_test = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, "  test"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            correct_test += outputs.eq(targets).sum().item()

    return correct_test / len(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("data", type=Path, help="path to data for training")
    arg("--batch-size", type=int, default=8)
    arg("--num-workers", type=int, default=0)
    arg("--num-epochs", type=int, default=10)
    arg("--lr", type=float, default=1e-4)
    arg("--wd", type=float, default=0.0)

    main(parser.parse_args())
