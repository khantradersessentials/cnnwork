"""
Real dataset loaders using torchvision. No synthetic/fabricated data.

Each dataset uses its own published per-channel mean/std for normalization.
Attacks in attacks.py operate in raw [0,1] pixel space (via NormalizedModel
wrapper in models.py) so epsilon budgets are meaningful in standard L_inf terms.
"""
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

DATASET_STATS = {
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616), "channels": 3, "classes": 10},
    "cifar100": {"mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762), "channels": 3, "classes": 100},
    "mnist":    {"mean": (0.1307,), "std": (0.3081,), "channels": 1, "classes": 10},
    "fmnist":   {"mean": (0.2860,), "std": (0.3530,), "channels": 1, "classes": 10},
    "svhn":     {"mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970), "channels": 3, "classes": 10},
    "stl10":    {"mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713), "channels": 3, "classes": 10},
}


def get_dataloaders(dataset_name, resolution=32, batch_size=32, data_dir="./data",
                     val_split=0.1, num_workers=2, seed=0):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, channels).
    Downloads the real dataset via torchvision (requires internet on first run).
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_STATS.keys())}")

    stats = DATASET_STATS[dataset_name]
    # NOTE: normalization is intentionally NOT baked into these transforms.
    # We keep tensors in [0,1] here and normalize inside the model (see
    # models.NormalizedModel) so that adversarial attacks operate in the
    # true pixel space with a meaningful epsilon (standard practice).
    train_tf = T.Compose([
        T.Resize((resolution, resolution)),
        T.RandomCrop(resolution, padding=max(2, resolution // 16)) if resolution >= 32 else T.Lambda(lambda x: x),
        T.RandomHorizontalFlip() if dataset_name not in ("mnist", "svhn") else T.Lambda(lambda x: x),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
    ])

    if dataset_name == "cifar10":
        train_full = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)
    elif dataset_name == "cifar100":
        train_full = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=eval_tf)
    elif dataset_name == "mnist":
        train_full = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=eval_tf)
    elif dataset_name == "fmnist":
        train_full = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=eval_tf)
    elif dataset_name == "svhn":
        train_full = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=train_tf)
        test_set = torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=eval_tf)
    elif dataset_name == "stl10":
        train_full = torchvision.datasets.STL10(data_dir, split="train", download=True, transform=train_tf)
        test_set = torchvision.datasets.STL10(data_dir, split="test", download=True, transform=eval_tf)
    else:
        raise ValueError(dataset_name)

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, stats["classes"], stats["channels"], stats["mean"], stats["std"]
