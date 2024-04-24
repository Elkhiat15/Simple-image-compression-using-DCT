import numpy as np
import matplotlib.pyplot as plt
import cv2

# make sure to choose your correct image path
PATH = 'Resources/original.png'
original_image = cv2.imread(PATH)
WIDTH, HEIGHT, _ = original_image.shape

blue, green, red = original_image.copy(), original_image.copy(), original_image.copy()
blue[:, :, (1, 2)] = 0
green[:, :, (0, 2)] = 0
red[:, :, (0, 1)] = 0

cv2.imshow("Blue channel", cv2.resize(blue, (960, 540)))
cv2.imshow("Green channel", cv2.resize(green, (960, 540)))
cv2.imshow("Red channel", cv2.resize(red, (960, 540)))

blue_, green_, red_ = cv2.split(original_image)
cv2.imshow("Blue channel Gray Scaled", cv2.resize(blue_, (960, 540)))
cv2.imshow("Green channel Gray Scaled", cv2.resize(green_, (960, 540)))
cv2.imshow("Red channel Gray Scaled", cv2.resize(red_, (960, 540)))

# Save B, G and R images
'''
cv2.imwrite("Blue_channel.png", blue)
cv2.imwrite("Green_channel.png", green)
cv2.imwrite("Red_channel.png", red)

cv2.imwrite("Blue_channel_GrayScale.png", blue_)
cv2.imwrite("Green_channel_GrayScale.png", green_)
cv2.imwrite("Red_channel_GrayScale.png", red_)
'''


def compressImage(m, show_imgs=False):
    """ compress and decompress image """

    decompressed_image = np.zeros_like(original_image)
    m = m
    compressed_image_width = m * (HEIGHT // 8)
    compressed_image = np.empty((m, compressed_image_width, 3))

    for i in range(0, WIDTH, 8):
        compressed_horizonal_slice = np.empty((m, m, 3))
        for j in range(0, HEIGHT, 8):
            block = original_image[i:i + 8, j:j + 8]
            block = block.astype(np.float32) / 255.0
            # split the three channels
            blue, green, red = np.float32(cv2.split(block))

            # apply 2d DCT to each channel to compress the image
            blue_dct = cv2.dct(blue)
            green_dct = cv2.dct(green)
            red_dct = cv2.dct(red)

            # keep the top-left block and ignore the rest
            blue_dct[m:, :] = 0
            blue_dct[:, m:] = 0
            green_dct[m:, :] = 0
            green_dct[:, m:] = 0
            red_dct[m:, :] = 0
            red_dct[:, m:] = 0

            # apply inverse DCT to each channel to decompress the image
            blue_idct = cv2.idct(blue_dct)
            green_idct = cv2.idct(green_dct)
            red_idct = cv2.idct(red_dct)

            # merge the BGR channels to construct our compressed and decompressed images
            decompressed_block = cv2.merge((blue_idct, green_idct, red_idct))
            decompressed_block = (decompressed_block * 255).clip(0, 255).astype(np.uint8)
            decompressed_image[i:i + 8, j:j + 8] = decompressed_block

            compressed_block = cv2.merge((blue_dct[:m, :m], green_dct[:m, :m], red_dct[:m, :m]))
            compressed_block = (compressed_block * 255).clip(0, 255).astype(np.uint8)

            if j == 0:
                compressed_horizonal_slice = compressed_block
            else:
                compressed_horizonal_slice = np.hstack((compressed_horizonal_slice, compressed_block))

        if i == 0:
            compressed_image = compressed_horizonal_slice
        else:
            compressed_image = np.vstack((compressed_image, compressed_horizonal_slice))

    # show the output images
    if show_imgs:
        cv2.imshow("Original image", cv2.resize(original_image, (960, 540)))
        cv2.imshow(f"Compressed image at m = {m}", compressed_image)
        cv2.imshow(f"Decompressed image at m = {m}", cv2.resize(decompressed_image, (960, 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return compressed_image, decompressed_image


def calculate_psnr(original_image, decompressed_image):
    """ calculate psnr value between two images """
    return cv2.PSNR(original_image, decompressed_image)


# to show image
compressed_image, decompressed_image = compressImage(m=2, show_imgs=True)

# to get compress image size at each value of m and calculate psnr value
img_sizes = []
PSNR_array = []
m_array = [1, 2, 3, 4]
for m_value in m_array:
    compressed_image, decompressed_image = compressImage(m=m_value)
    # cv2.imwrite(f"compressed_image_m({m_value}).png", compressed_image)
    # cv2.imwrite(f"decompressed_image_m({m_value}).png", decompressed_image)

    psnr_value = calculate_psnr(original_image, decompressed_image)
    PSNR_array.append(psnr_value)

    img_sizes.append(compressed_image.size)

# plot m values with compressed images sizes
plt.bar(m_array, img_sizes)
for x, y in zip(m_array, img_sizes):
    plt.text(x - 0.03, y + 5000, y, ha='center', va='bottom')
plt.xlabel("M Value")
plt.ylabel("Image Size (bit)")
plt.title("M value with compressed images sizes")
# plt.savefig("Image_Sizes.jpg")

# plot m values with PSNR values
plt.bar(m_array, PSNR_array)
for x, y in zip(m_array, PSNR_array):
    plt.text(x - 0.03, y + 0.2, round(y, 3), ha='center', va='bottom')
plt.xlabel("M Value")
plt.ylabel("PSNR values")
plt.title("M value with PSNR values")
# plt.savefig("PSNR_Plot.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
