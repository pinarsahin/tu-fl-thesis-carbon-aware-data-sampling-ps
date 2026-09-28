import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


fds = None
_fds_key = None


def get_transform():
    return Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])


def apply_transforms(batch):
    transform = get_transform()
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


def load_partition_datasets(
    partition_id: int,
    num_partitions: int,
    sample_fraction: float = 0.05,
    partition_method: str = "dirichlet",   # default keeps current behavior
    alpha: float = 0.5,                     # default keeps current value
    classes_per_partition: int = 8,         # only used for pathological
):
    global fds, _fds_key

    # Rebuild fds only when the partition config changes, so switching
    # method/alpha across runs works and does not silently reuse a stale
    # partition. With defaults, behavior is identical to the current setup.
    key = (partition_method, num_partitions, alpha, classes_per_partition)
    if fds is None or _fds_key != key:
        if partition_method == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="character",
                alpha=alpha,
                seed=42,
            )
        elif partition_method == "pathological":
            from flwr_datasets.partitioner import PathologicalPartitioner
            partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                partition_by="character",
                num_classes_per_partition=classes_per_partition,
                class_assignment_mode="first-deterministic",
                shuffle=True,
                seed=42,
            )
            
        else:
            raise ValueError(f"Unknown partition_method: {partition_method}")

        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": partitioner},
        )
        _fds_key = key

    partition = fds.load_partition(partition_id)
    partition = partition.shuffle(seed=42)
    keep_n = max(1, int(len(partition) * sample_fraction))
    partition = partition.select(range(keep_n))
    partition = partition.train_test_split(test_size=0.2, seed=42)
    partition = partition.with_transform(apply_transforms)
    return partition["train"], partition["test"]

def load_centralized_testloader(batch_size: int = 128):
    ds = load_dataset("flwrlabs/femnist")
    test_data = ds["train"].train_test_split(test_size=0.1, seed=42)["test"]
    test_data = test_data.with_transform(apply_transforms)
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)


def make_dataloader(dataset, batch_size: int, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(net, trainloader, epochs, lr, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    return loss / len(testloader), correct / len(testloader.dataset)

def test_per_class(net, testloader, device, num_classes=62):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum().item()
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1
    total = len(testloader.dataset)
    return loss / len(testloader), correct / total, class_correct, class_total
