import torchvision.transforms as transforms


class SnapResize(transforms.Resize):
    """
    Resizes a given img to the nearest power of 2
    """
    def __call__(self, img):
        width, height = img.size
        new_width = 2 ** ((width - 1).bit_length())
        new_height = 2 ** ((height - 1).bit_length())
        return super(SnapResize, self).__call__((new_width, new_height))


def cifar10_transforms(mean=(0.49139968, 0.48215841, 0.44653091),
                        std=(0.24703223, 0.24348513, 0.26158784)):

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])


def cifar100_transforms(mean=(0.50707516, 0.48654887, 0.44091784),
                        std=(0.26733429, 0.25643846, 0.27615047)):


    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])


def mnist_transforms(mean=(0.1306604762738429),
                        std=(0.30810780717887876)):


    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])


def imagenet_transforms(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]):
    t_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    return transforms.Compose(t_list)


def custom_images_transforms(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]):
    """
    Normalization defaults to ImageNet distribution as it is probable the
    custom image dataset shares similarities with ImageNet given the
    latter's sheer volume
    """
    t_list = [
        SnapResize(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    return transforms.Compose(t_list)


