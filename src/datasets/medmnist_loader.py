from dataclasses import dataclass
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import medmnist
from medmnist import INFO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SplitLoaders:
    id_train: DataLoader
    id_val: DataLoader
    id_test: DataLoader
    near_ood: DataLoader


def get_transform(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not augment:
        return transforms.Compose(base)
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_split(name: str, split: str, root: str, image_size: int, augment: bool = False):
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass(
        split=split,
        transform=get_transform(image_size, augment),
        download=True,
        root=root,
        size=image_size,
        as_rgb=True,
    )


def filter_by_classes(dataset, classes: list[int]) -> Subset:
    classes_set = set(classes)
    labels = dataset.labels.squeeze()
    indices = [i for i, lbl in enumerate(labels) if int(lbl) in classes_set]
    return Subset(dataset, indices)


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_loaders(
    name: str,
    id_classes: list[int],
    near_ood_classes: list[int],
    batch_size: int = 64,
    root: str = "./data/medmnist",
    image_size: int = 224,
    num_workers: int = 4,
) -> SplitLoaders:
    id_train = _make_loader(
        filter_by_classes(load_split(name, "train", root, image_size, augment=True), id_classes),
        batch_size, shuffle=True, num_workers=num_workers,
    )
    id_val = _make_loader(
        filter_by_classes(load_split(name, "val", root, image_size), id_classes),
        batch_size, shuffle=False, num_workers=num_workers,
    )

    test_raw = load_split(name, "test", root, image_size)
    id_test = _make_loader(
        filter_by_classes(test_raw, id_classes),
        batch_size, shuffle=False, num_workers=num_workers,
    )
    near_ood = _make_loader(
        filter_by_classes(test_raw, near_ood_classes),
        batch_size, shuffle=False, num_workers=num_workers,
    )

    return SplitLoaders(id_train=id_train, id_val=id_val, id_test=id_test, near_ood=near_ood)


def get_far_ood_loaders(
    ood_datasets: list[str],
    batch_size: int = 64,
    root: str = "./data/medmnist",
    image_size: int = 224,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    return {
        name: _make_loader(
            load_split(name, "test", root, image_size),
            batch_size, shuffle=False, num_workers=num_workers,
        )
        for name in ood_datasets
    }


def dataset_info(name: str) -> dict:
    info = INFO[name]
    return {
        "name": name,
        "task": info["task"],
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "labels": info["label"],
    }
