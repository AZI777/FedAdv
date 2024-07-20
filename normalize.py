import torch
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms

if __name__ == "__main__":
    data = EMNIST(root='.', split="byclass", train=True, download=True,
                  transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(data, batch_size=16)
    total_count = 0
    total_value = 0
    for input, _ in dataloader:
        total_count += input.numel()
        total_value += torch.sum(input)
    average_value = total_value / total_count

    s = 0
    for input, _ in dataloader:
        s += torch.sum((input - average_value) ** 2).item()
    s /= total_count
    print(average_value,s)
