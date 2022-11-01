import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import config as cfg


def test(test_checkpoint=True):
    evaluate = False

    checkpoint_path = cfg.MODEL_PATH_CKPT+'.h5py'
    model_path = cfg.MODEL_PATH+'.h5py'
    img_w, img_h, _ = cfg.INPUT_SHAPE

    if test_checkpoint:
        model = keras.models.load_model(checkpoint_path)
    else:
        model = keras.models.load_model(model_path)

    print(model.summary())

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
