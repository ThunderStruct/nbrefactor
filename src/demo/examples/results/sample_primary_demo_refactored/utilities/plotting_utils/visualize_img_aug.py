from matplotlib import plt as plt


def plot_img_aug(sample_image, augmented_image, aug_title):
    # Plot original and augmented images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(aug_title)
    plt.imshow(augmented_image)
    plt.axis('off')

    plt.show()
    
    

