import numpy as np
from PIL import Image
from siamese import Siamese
import os

root_folder = os.path.dirname(
        os.path.abspath(__file__)
        )

def similar(pic_1_path,pic_2_path):
    model_name="classify.onnx"
    model = Siamese(os.path.join(root_folder,"logs",model_name))

    image_1 = pic_1_path
    image_1 = Image.open(image_1)

    image_2 = pic_2_path
    image_2 = Image.open(image_2)
    probability =float(model.detect_image(image_1,image_2)) 
    return probability

