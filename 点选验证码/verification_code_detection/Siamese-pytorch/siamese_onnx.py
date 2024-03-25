import onnxruntime
import numpy as np
from PIL import Image


class SiamOnnx(object):
    def __init__(self,path,providers=None):
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.sess = onnxruntime.InferenceSession(path,providers=providers)
        self.loadSize =512
        self.input_shape = [105,105]

    
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def transform(file):
        if isinstance(file,np.ndarray):
            img=Image.fromarray(file)
        elif isinstance(file,bytes):
            img=Image.open(BytesID(file))