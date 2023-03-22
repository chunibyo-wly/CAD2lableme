import cv2
import numpy as np
from os.path import join, basename
from tqdm import trange


def main():
    for i in trange(6):
        file_name = f"./input/F{i + 1}.png"
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        image = image[np.any(image == 0, axis=1), :][:, np.any(image == 0, axis=0)]
        cv2.imwrite(join("cropped", basename(file_name)), image)


if __name__ == "__main__":
    main()
