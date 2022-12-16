from PIL import Image
import matplotlib.pyplot as plt

from utils import prepare_input, interocular_nme
from affine import predict
from config import config as cfg


def test_predict(img_paths, is_file=True):
    preds = predict(img_paths, cfg.DATA.TARGET, is_file=is_file)

    for i in range(len(img_paths)):
        if is_file:
            image = Image.open(img_paths[i])
        else:
            image = img_paths[i]

        plt.imshow(image)
        for x, y in preds[0]:
            plt.plot(x, y, 'r.')
        plt.show()


if __name__ == '__main__':
    img_paths = ['../data/cleft/test_images/Abu Ghader_Karam (39).JPG',
                '../data/cleft/test_images/Al Araj_Ahmad_18_JAN_2020 (1).JPG',
                '../data/cleft/test_images/Abou Sadet_Karim_07_DEC_19 (9).JPG']

    test_predict([img_paths[0]])

