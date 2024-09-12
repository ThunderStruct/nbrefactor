from ..utilities.functional_utils.file_utils import should_overwrite_path
from ..utilities.logger import Logger
import torchvision
from ..utilities.functional_utils.file_utils import recursive_dir_delete
import os
import PIL
import torch
from enum import auto
from torch.utils.data import DataLoader
from ..utilities.metadata.datasource_metadata import DataSourceMetadata
import re
import torchvision.transforms as transforms
from enum import Enum

class VisionDataSource:

    class Dataset(Enum):
        CIFAR10     = auto()
        CIFAR100    = auto()
        MNIST       = auto()
        IMAGENET    = auto()
        CUSTOM      = auto()

    class SegmentableImageFolder(torchvision.datasets.ImageFolder):
        def __init__(self, root, split, segment_size, segment_idx,
                     transform=None, target_transform=None,
                     loader=torchvision.datasets.folder.default_loader,
                     is_valid_file=None, accumulate_classes=True):

            assert split in ['train', 'val'], (
                f'Invalid split {split} provided. Valid values are `train` '
                'and `val`'
            )

            self.split = split
            self.accumulate_classes = accumulate_classes
            self.segment_start = segment_size * segment_idx
            self.segment_end = self.segment_start + segment_size

            split_root = os.path.join(root, split)

            super(VisionDataSource.SegmentableImageFolder,
                  self).__init__(split_root, transform, target_transform,
                                 loader, is_valid_file)


        def find_classes(self, directory):
            def natural_sort(l):
                convert = lambda text: int(text) \
                if text.isdigit() else text.lower()
                alphanum_key = lambda key: [convert(c) \
                                            for c in re.split('([0-9]+)', key)]
                return sorted(l, key=alphanum_key)

            classes = natural_sort(entry.name \
                                   for entry in os.scandir(directory) \
                                   if entry.is_dir())
            if not classes:
                raise FileNotFoundError(f'Couldn\'t find any class folder in ' +
                                        f'{directory}.')

            # filter classes
            # classes = classes[self.segment_start:self.segment_end]
            slice_start = 0 if self.accumulate_classes else self.segment_start
            classes = classes[slice_start:self.segment_end]

            # filter class_to_idx
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

            return classes, class_to_idx


    def __init__(self, path, dataset=None, transform=None, num_workers=4,
                 autoload=False, train_data=None, val_data=None,
                 segment_size=None, segment_idx=None):
        assert dataset is None or isinstance(dataset,
                                             VisionDataSource.Dataset), (
            'The dataset provided is invalid! Make sure it '
            'is of type `VisionDataSource.Dataset`'
        )

        self.dataset = dataset
        self.path = path
        self.transform = transform
        self.num_workers = num_workers
        self.train_data = train_data
        self.val_data = val_data
        self.segment_size = segment_size
        self.segment_idx = segment_idx if segment_idx is not None else 0

        if autoload:
            self.load()

    # --------------------------------------------------------------------------
    # DATASET LOADERS

    def load(self):
        """
        Dataset loading
        (downloads torchvision datasets if not previously downloaded)
        """

        trns = self.transform
        if self.dataset == VisionDataSource.Dataset.CIFAR10:
            self.train_data = torchvision.datasets.CIFAR10(root=self.path,
                                                           train=True,
                                                           download=True,
                                                           transform=trns)
            self.val_data = torchvision.datasets.CIFAR10(root=self.path,
                                                         train=False,
                                                         download=True,
                                                         transform=trns)

        elif self.dataset == VisionDataSource.Dataset.CIFAR100:
            self.train_data = torchvision.datasets.CIFAR100(root=self.path,
                                                            train=True,
                                                            download=True,
                                                            transform=trns)
            self.val_data = torchvision.datasets.CIFAR100(root=self.path,
                                                          train=False,
                                                          download=True,
                                                          transform=trns)
        elif self.dataset == VisionDataSource.Dataset.MNIST:
            self.train_data = torchvision.datasets.MNIST(root=self.path,
                                                         train=True,
                                                         download=True,
                                                         transform=trns)
            self.val_data = torchvision.datasets.MNIST(root=self.path,
                                                       train=False,
                                                       download=True,
                                                       transform=trns)
        elif self.dataset == VisionDataSource.Dataset.IMAGENET:
            self.train_data = torchvision.datasets.ImageNet(root=self.path,
                                                            split='train',
                                                            transform=trns)
            self.val_data = torchvision.datasets.ImageNet(root=self.path,
                                                          split='val',
                                                          transform=trns)
        elif self.dataset == VisionDataSource.Dataset.CUSTOM:
            train_path = os.path.join(self.path, 'train')
            val_path = os.path.join(self.path, 'val')
            self.train_data = torchvision.dataset.ImageFolder(root=train_path,
                                                              transform=trns)
            self.val_data = torchvision.dataset.ImageFolder(root=val_path,
                                                            transform=trns)

        if self.segment_size is None:
            self.segment_size = len(self.val_data.classes)


    # --------------------------------------------------------------------------
    # STATIC UTILITIES


    @staticmethod
    def class_segmentation_factory(path, dataset,
                                   segment_size, segment_idx, transform):
        trns = transform
        ss, si = segment_size, segment_idx

        train_data = VisionDataSource.SegmentableImageFolder(root=path,
                                                             split='train',
                                                             segment_size=ss,
                                                             segment_idx=si,
                                                             transform=trns)
        val_data = VisionDataSource.SegmentableImageFolder(root=path,
                                                           split='val',
                                                           segment_size=ss,
                                                           segment_idx=si,
                                                           transform=trns)

        return VisionDataSource(path, dataset=dataset, transform=transform,
                                train_data=train_data, val_data=val_data,
                                segment_size=ss, segment_idx=si)


    @staticmethod
    def __convert_dataset_to_img(dataset, root_dir, split='train', ext='png'):

        Logger.info(f'Converting compressed dataset to {ext} format')
        Logger.info('This may take a while...')

        # intentionally not joining the root_dir with `Config.BASE_PATH`
        # saving locally on Colab instead to avoid clutter on GDrive when
        # applicable
        full_path = os.path.join(root_dir, split)

        for idx, (image, label) in enumerate(dataset):
            label_dir = os.path.join(full_path, str(label))
            os.makedirs(label_dir, exist_ok=True)
            image_path = os.path.join(label_dir, f'{idx}.{ext}')
            if not isinstance(image, PIL.Image.Image):
                image = torchvision.transforms.ToPILImage()(image)
            image.save(image_path)

        Logger.info(f'Successfully converted dataset! Saved to `{full_path}`')


    @staticmethod
    def __torchvision_downloader(cls, path, convert_to_img):
        """
        """

        train_data = cls(root=path, train=True, download=True)
        val_data = cls(root=path, train=False, download=True)

        if convert_to_img:
            # convert dataset to image format and save for consistency
            # to allow segmentation and consistent handling later

            # delete downloaded compressed dataset
            recursive_dir_delete(path)

            # save segmentable dataset
            VisionDataSource.__convert_dataset_to_img(train_data,
                                                      path,
                                                      split='train')
            VisionDataSource.__convert_dataset_to_img(val_data,
                                                      path,
                                                      split='val')

    @staticmethod
    def download_cifar10(path, force_overwrite=False,
                         allow_segmentation=False):
        """
        allow_segmentation is an inefficient implementation as it converts each
        sample to an image from its original compressed format (higher space
        and time complexity)
        """
        if not should_overwrite_path(path, force_overwrite):
            return

        # download CIFAR-10 dataset
        VisionDataSource.__torchvision_downloader(
            torchvision.datasets.CIFAR10, path, allow_segmentation
        )


    @staticmethod
    def download_cifar100(path, force_overwrite=False,
                          allow_segmentation=False):
        """
        allow_segmentation is an inefficient implementation as it converts each
        sample to an image from its original compressed format (higher space
        and time complexity)
        """
        if not should_overwrite_path(path, force_overwrite):
            return

        # download CIFAR-10 dataset
        VisionDataSource.__torchvision_downloader(
            torchvision.datasets.CIFAR100, path, allow_segmentation
        )


    @staticmethod
    def download_mnist(path, force_overwrite=False, allow_segmentation=False):
        """
        allow_segmentation is an inefficient implementation as it converts each
        sample to an image from its original compressed format (higher space
        and time complexity)
        """
        if not should_overwrite_path(path, force_overwrite):
            return

        # download MNIST dataset
        VisionDataSource.__torchvision_downloader(
            torchvision.datasets.MNIST, path, allow_segmentation
        )


    @staticmethod
    def compute_data_distribution(path, transform=[], batch_size=256):
        """
        Iterates through the dataset to calculates the mean/std, and
        infer the channel dimensions.
        """

        if not os.path.exists(path):
            Logger.warning(f'The given path "{path}" does not exist!')
            return

        # load dataset
        t_comp = transforms.Compose(transform + [transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(root=path,
                                                   transform=t_comp)

        Logger.progress('Computing mean and standard deviation for the dataset')
        Logger.progress('This may take a while...')

        # temporary DataLoader required to loop through the dataset
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)

        # var[X] = E[X**2] - E[X]**2
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        for batch in loader:
            images, labels = batch
            # (B, C, H, W)
            channels_sum += torch.mean(images, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
            num_batches += 1

            # if self.channels is None:
            #     self.channels = images[0].shape[1]

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

        Logger.progress(f'Computed mean: {mean}, std: {std}')

        return mean, std


    # --------------------------------------------------------------------------
    # UTILITIES

    @property
    def metadata(self):
        params = ['path', 'segment_size', 'segment_idx', 'num_workers']

        transforms = [str(t) for t in self.transform.transforms]
        dataset = ''
        if self.dataset is not None:
            dataset = self.dataset.name

        return DataSourceMetadata(path=self.path,
                                  segment_size=self.segment_size,
                                  segment_idx=self.segment_idx,
                                  num_workers=self.num_workers,
                                  transforms=transforms,
                                  dataset=dataset)

    # --------------------------------------------------------------------------
    # OPERATORS

    def __str__(self):
        return str(self.metadata)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.metadata)

    def __eq__(self, other):
        if not isinstance(other, VisionDataSource):
            return False

        return self.metadata == other.metadata


