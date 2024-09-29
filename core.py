import matplotlib.pyplot as plt
import cv2


def process_and_show_captured_hand_palm(captured_image):
    """Process captured image and extract features."""

    # Display results alongside original captured image
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Three plots in one row

    # Show the captured palm image
    axs[0].imshow(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Captured Palm Image")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Second Caupted Palm Image")
    axs[1].axis("off")

    # Display the image
    plt.show()
