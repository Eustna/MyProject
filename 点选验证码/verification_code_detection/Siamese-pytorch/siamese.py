import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import os
from nets.siamese import Siamese as siamese
import onnx as onx
import onnxruntime as ort

class Siamese(object):

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self,path, **kwargs):
        self._defaults = {
            "model_path"    :path,
            "input_shape"   : (105, 105, 3),
            "cuda"          : True
        }
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model   = siamese(self.input_shape)
        _, file_extension = os.path.splitext(self.model_path)
        if file_extension == '.onnx':
            self.net = onx.load(self.model_path)
            self.sess = ort.InferenceSession(self.model_path)
        else:
            model = siamese(self.input_shape)
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.net = model.eval()
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.net = model.eval()

            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True
                self.net = self.net.cuda()
    
    def letterbox_image(self, image, size):
        image   = image.convert("RGB")
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        
    def detect_image(self, image_1, image_2):
        image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
        image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
        photo_1 = np.asarray(image_1).astype(np.float64) / 255
        photo_2 = np.asarray(image_2).astype(np.float64) / 255

        if self.input_shape[-1]==1:
            photo_1 = np.expand_dims(photo_1, -1)
            photo_2 = np.expand_dims(photo_2, -1)

        with torch.no_grad():
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
            _, file_extension = os.path.splitext(self.model_path)
            if file_extension == '.onnx':
                sess = ort.InferenceSession(self.model_path)
                input_name_1 = sess.get_inputs()[0].name
                input_name_2 = sess.get_inputs()[1].name
                input_data_1 = photo_1.cpu().numpy() 
                input_data_2 = photo_2.cpu().numpy()  
                output = sess.run(None, {input_name_1: input_data_1, input_name_2: input_data_2})
                output = output[0]
                plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
            else:
                # 运行PyTorch模型的推理
                output = self.net([photo_1, photo_2])[0]
                output = torch.nn.Sigmoid()(output)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        # plt.show()
        return output
