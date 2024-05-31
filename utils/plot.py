import matplotlib.pyplot as plt

def Display_sample(display_list):
    """
    Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = display_list[i]
        if len(img.shape) == 2:  # for single-channel images (masks)
            plt.imshow(img.cpu().numpy())
        else:  # for multi-channel images (RGB images)
            img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
            plt.imshow(img.cpu().numpy())
        plt.axis('off')
    plt.show()
