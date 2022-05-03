import logging
import os
from os import listdir
import torch
from skimage import io
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image

logger = logging.getLogger(__name__)

# add hymenoptera dataset
class HymenopteraDataset(Dataset):
    def __init__(self, root_dir='dataset/hymenoptera_data', transform=None, train=True):
        self.categories = {'ants':0, 'bees':1}
        self.root_dir = os.path.join(root_dir, 'train' if train else 'val')
        self.transform = transform
        self.paths = self._read_dataset()

    def _read_dataset(self):
        paths = []
        for c, label in self.categories.items():
            paths_tmp = sorted(listdir(os.path.join(self.root_dir, c)))
            paths.extend([[os.path.join(self.root_dir, c, path), label] for path in paths_tmp])
        return paths
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        
        img_path, label = self.paths[idx]
        # image = io.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "hymenoptera":
        trainset = HymenopteraDataset(transform=transform_train, train=True)
        testset = HymenopteraDataset(transform=transform_test, train=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
