import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import config as cfg


def test():
    evaluate = False

    model_path = cfg.MODEL_PATH
    img_w, img_h = cfg.IMG_SHAPE

    model = keras.load_model(model_path)

    # TODO load images & labels
    images = []
    labels = []

    # EVALUATION
    if evaluate:
        score = model.evaluate(images, labels, verbose=2)
        print("loss:", score[0])
        print("acc:", score[1])

    # PREDICTION
    else:
        pts = model.predict(images)
        for i in range(len(images)):
            plt.imshow(images[i])
            plt_pts = np.reshape(pts[i], (-1, 2))
            for x, y in plt_pts:
                plt.plot(x * img_w, y * img_h, 'ro', markersize=0.5)
            plt.show()


if __name__ == "__main__":
    test()
