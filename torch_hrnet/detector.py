from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt

class FaceDetector():
    def __init__(self):
        self.mtcnn = MTCNN(select_largest=False, device='cuda:0')

    def detect(self, image):
        return self.mtcnn.detect(image, landmarks=True)

    def __call__(self, image):
        return self.mtcnn(image)

if __name__ == '__main__':
    img_path = ['../data/cleft/test_images/Abu Ghader_Karam (39).JPG',
                '../data/cleft/test_images/Al Araj_Ahmad_18_JAN_2020 (1).JPG',
                '../data/cleft/test_images/Abou Sadet_Karim_07_DEC_19 (9).JPG']

    img_path = img_path[0]

    image = Image.open(img_path)
    detector = FaceDetector()
    boxes, probs, landmarks = detector.detect(image)

    plt.imshow(image)
    plt.plot(boxes[0][0], boxes[0][1], 'r.')
    plt.plot(boxes[0][2], boxes[0][3], 'r.')
    plt.show()

    print(boxes)

