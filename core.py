import matplotlib.pyplot as plt
import cv2


def process_and_show_captured_hand_palm(captured_image):
    """Process captured image and extract features."""

    # Write me a function to write capture image to disk
    cv2.imwrite("./hand_palm.jpg", captured_image)

    # Display results alongside original captured image
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Three plots in one row

    # Show the captured palm image
    axs[0].imshow(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Captured Palm Image")
    axs[0].axis("off")

    adjust_img = adjust_illumination(captured_image)
    palm_print = extract_palm_print(adjust_img)
    axs[1].imshow(cv2.cvtColor(palm_print, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Palm Print")
    axs[1].axis("off")

    # Display the image
    plt.show()


def adjust_illumination(image):
    """Adjust illumination of the image."""
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)


def extract_palm_print(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight the largest contour (assuming it's the palm)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        highlighted_image = cv2.drawContours(
            image.copy(), [largest_contour], -1, (0, 0, 255), thickness=2
        )
    else:
        highlighted_image = image.copy()  # No palm contour found

    return highlighted_image
