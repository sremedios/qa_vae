from matplotlib import pyplot as plt
import numpy as np

def show_image(img_data):
    plt.imshow(img_data, interpolation='nearest', cmap="Greys")
    plt.show()

def dual_show_image(img_data_1, img_data_2):
    fig = plt.figure()
    ax_1 = fig.add_subplot(1,2,1)
    ax_1.set_title("Original")
    ax_1.imshow((img_data_1 * 255).astype(np.uint8), interpolation='nearest', cmap="Greys")

    ax_2 = fig.add_subplot(1,2,2)
    ax_2.set_title("Reconstructed")
    ax_2.imshow((img_data_2 * 255).astype(np.uint8), interpolation='nearest', cmap="Greys")

    plt.show()
