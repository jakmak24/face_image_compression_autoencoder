from imutils import face_utils
import dlib
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

dataset_path = "DATASET_PATH"

p = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

import os

os.makedirs("out",exist_ok=True)

for filename in os.listdir(dataset_path):

    print(filename)
    image = Image.open(dataset_path+"/"+filename)
    image = np.array(image)

    rects = detector(image, 0)

    width, height, channels = image.shape

    result = np.full((width, height), 5)
    result = Image.fromarray(result).convert("L")

    d = ImageDraw.Draw(result)

    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        print(rect.left(), rect.top(), rect.right(), rect.bottom())

        shape = [(x, y) for (x, y) in shape]

        d.rectangle([rect.left(), rect.top(), rect.right(), rect.bottom()], fill=100)
        d.polygon(shape[0:17], fill=150)  # jaw
        d.polygon(shape[17:22], fill=200)  # eyebrowL
        d.polygon(shape[22:27], fill=200)  # eyebrowR
        d.polygon(shape[27:36], fill=200)  # nose
        d.polygon(shape[36:42], fill=255)  # eyeL
        d.polygon(shape[42:48], fill=255)  # eyeR
        d.polygon(shape[48:61], fill=200) #lips
        d.polygon(shape[61:68], fill=180) #lips

        #result = result.filter(ImageFilter.GaussianBlur(1))

        break

    result.save("out/"+filename[:-3]+"png")



