from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import medmnist
from medmnist import INFO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    if augment:
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
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_raw(name: str, split: str, root: str, image_size: int, augment: bool):
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


def _make_loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_class_split_loaders(
    name: str,
    in_dist_classes: list[int],
    near_ood_classes: list[int],
    batch_size: int = 64,
    root: str = "./data/medmnist",
    image_size: int = 224,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    loaders = {}

    for split, augment in [("train", True), ("val", False)]:
        raw = load_raw(name, split, root, image_size, augment=augment)
        subset = filter_by_classes(raw, in_dist_classes)
        loaders[f"in_{split}"] = _make_loader(
            subset, batch_size, shuffle=(split == "train"), num_workers=num_workers
        )

    raw_test = load_raw(name, "test", root, image_size, augment=False)
    loaders["in_test"] = _make_loader(
        filter_by_classes(raw_test, in_dist_classes), batch_size, False, num_workers
    )
    loaders["near_ood"] = _make_loader(
        filter_by_classes(raw_test, near_ood_classes), batch_size, False, num_workers
    )

    return loaders


def get_far_ood_loaders(
    ood_datasets: list[str],
    batch_size: int = 64,
    root: str = "./data/medmnist",
    image_size: int = 224,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    return {
        name: _make_loader(
            load_raw(name, "test", root, image_size, augment=False),
            batch_size,
            False,
            num_workers,
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
