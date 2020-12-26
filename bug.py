"""

Author: Arthur Wesley

"""

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import save_img


def main():
    """

    bug here

    :return:
    """

    dataset = image_dataset_from_directory("data",
                                           image_size=(640, 360))

    x, y = next(iter(dataset))

    save_img("bug.png", x[0])


if __name__ == "__main__":
    main()
