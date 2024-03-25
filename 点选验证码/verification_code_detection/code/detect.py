import cv2
from PIL import Image
import time
import onnxruntime as ort
import numpy as np
import os
import sys


root_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
        )
    )

sys.path.insert(0, os.path.join(root_folder,"Siamese-pytorch"))
import predict

def cut_pitcture(input_path, output_path, start_x, start_y, end_x, end_y):
    with Image.open(input_path) as img:
        cropped_img = img.crop((start_x, start_y, end_x, end_y))
        cropped_img.save(output_path)
class getTpInfo():
    def __init__(self):
        with open(class_path, encoding='utf-8') as f:
            t = f.read().split('\n')
        self.alllb = t
        self.Train_model = ort.InferenceSession(model_path)

    def bbbiou(self, rec1, rec2):
        if self.pdisIn(rec1[0], rec1[1], rec1[2], rec1[3], rec2[0], rec2[1], rec2[2], rec2[3]) == False:
            return 0
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S2 + S1 - S_cross)

    def pdisIn(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False


    def hetInfo(self,out):
        out = out[0]
        lzx = 1 / out.shape[0]
        lzy = 1 / out.shape[1]
        kd = []

        for i in range(out.shape[0]):
            zxdwx = lzx * i + lzx / 2
            for i2 in range(out.shape[1]):
                zxdwy = lzy * i2 + lzy / 2
                for k in range(out.shape[2]):
                    if out[i, i2, k, 4] > 0.9:
                        zxx = (out[i, i2, k, 0] - 0.5) + zxdwx
                        zxy = (out[i, i2, k, 1] - 0.5) + zxdwy
                        zxk = (out[i, i2, k, 2] - 0.5) + lzx
                        zxg = (out[i, i2, k, 3] - 0.5) + lzy
                        center_x = (zxx - zxk / 2 + zxx + zxk / 2) / 2
                        center_y = (zxy - zxg / 2 + zxy + zxg / 2) / 2
                        l = [zxx - zxk / 2, zxy - zxg / 2,
                            zxx + zxk / 2, zxy + zxg / 2, out[i, i2, k, 4], center_x, center_y]
                        isokk = 1
                        for idx, ds in enumerate(kd):
                            if self.bbbiou([l[0], l[1], l[2], l[3]], [ds[0], ds[1], ds[2], ds[3]]) < 0.1:
                                continue
                            else:
                                isokk = 0
                                if ds[4] < l[4]:
                                    kd[idx] = l
                        if isokk == 1:
                            kd.append(l)

        return kd


    def getimage(self,input_path,output_path):
        imge = Image.open(input_path).convert('RGB')
        dst = imge.resize((320, 192), Image.BILINEAR)
        output_folder = os.path.dirname(output_path)
        temporary_path= os.path.join(output_folder, "temporary_pic.jpg")
        dst.save(temporary_path)
        dst = np.array(dst).astype(np.float32) / 255
        img = dst.transpose(2, 1, 0).reshape((1, 3, 320, 192))
        return img, imge

    def box_pitcture(self,input_path,output_path):
        input_pic, imge = self.getimage(input_path,output_path)
        d = time.time()
        box_obj = self.Train_model.run(None, {self.Train_model.get_inputs()[0].name: input_pic})
        print("识别用时",time.time()-d, "秒")

        box = self.hetInfo(box_obj) # 获取框的信息

        input_pic = cv2.imread(input_path)
        y = input_pic.shape[0]
        x = input_pic.shape[1]
        position_0=0
        position_1=0
        position_2=0
        for idx, i in enumerate(box):
            if idx==0:
                position_0=i
            if idx == 1:
                position_1=i
            if idx == 2:
                position_2=i
            cv2.rectangle(input_pic, (int(i[0] * x), int(i[1] * y)), (int(i[2] * x), int(i[3] * y)), (0, 0, 255), 2)

        # cv2.imshow("image", input_pic)
        cv2.imwrite(output_path, input_pic)


        if os.path.exists(os.path.join(os.path.dirname(output_path), "temporary_pic.jpg")):
            os.remove(os.path.join(os.path.dirname(output_path), "temporary_pic.jpg"))
        cv2.waitKey(0)
        if position_0 == 0:
            print("没有检测到")
        elif position_1 == 0:
            print("检测到1个")
        elif position_2 == 0:
            print("检测到2个")
        
        return position_0,position_1,position_2




#修改点
model_name="det_model"
InPic_first_name = "pic_261"
InPic_name = InPic_first_name+".png"


model_path = os.path.join(root_folder, "model",f"{model_name}.onnx")
class_path = os.path.join(root_folder, "train","classes.txt")
InPic_path = os.path.join(root_folder, "input_pic",InPic_name)
OutPic_path= os.path.join(root_folder, "output_pic",InPic_name)


picture = getTpInfo()
pic_information=picture.box_pitcture(InPic_path,OutPic_path)

cut_folder=os.path.join(root_folder,"cut_pic")
cut_pic = cv2.imread(InPic_path)
heigth = cut_pic.shape[0]
width = cut_pic.shape[1]

obj=1
obj_center=0
obj_Right=0
obj_Left=0
if pic_information[0] == 0:
    print("没有检测到")
elif pic_information[1] == 0:
    print("检测到1个")
    index = 1
    obj_1=pic_information[0]
    obj_center_1=(pic_information[0][5],pic_information[0][6])
    obj_Left_1  =(pic_information[0][0],pic_information[0][1])
    obj_Right_1 =(pic_information[0][2],pic_information[0][3])


elif pic_information[2] == 0:
    print("检测到2个")
    index = 2
    obj_1=pic_information[0]
    obj_center_1=(pic_information[0][5],pic_information[0][6])
    obj_Left_1  =(pic_information[0][0],pic_information[0][1])
    obj_Right_1 =(pic_information[0][2],pic_information[0][3])

    obj_2=pic_information[1]
    obj_center_2=(pic_information[1][5],pic_information[1][6])
    obj_Left_2  =(pic_information[1][0],pic_information[1][1])
    obj_Right_2 =(pic_information[1][2],pic_information[1][3])

else:
    index=3
    obj_1=pic_information[0]
    obj_center_1=(pic_information[0][5],pic_information[0][6])
    obj_Left_1  =(pic_information[0][0],pic_information[0][1])
    obj_Right_1 =(pic_information[0][2],pic_information[0][3])

    obj_2=pic_information[1]
    obj_center_2=(pic_information[1][5],pic_information[1][6])
    obj_Left_2  =(pic_information[1][0],pic_information[1][1])
    obj_Right_2 =(pic_information[1][2],pic_information[1][3])

    obj_3=pic_information[2]
    obj_center_3=(pic_information[2][5],pic_information[2][6])
    obj_Left_3  =(pic_information[2][0],pic_information[2][1])
    obj_Right_3 =(pic_information[2][2],pic_information[2][3])

might=list()
click_list=[0,0,0]
name_list=list()

import datetime

now = datetime.datetime.now()

for i in range(index):
    cut_pitcture(InPic_path, 
                    os.path.join(root_folder,"cut_pic",f"{InPic_first_name}_cut_{i+1}.png"), 
                    int(pic_information[i][0]*width) , 
                    int(pic_information[i][1]*heigth) , 
                    int(pic_information[i][2]*width) , 
                    int(pic_information[i][3]*heigth))
    
for j in range(3):
    for i in range(index):
        if f"{InPic_first_name}_cut_{i+1}.png" in name_list:
            pred = -2
        else :
            pred=predict.similar(os.path.join(cut_folder,f"cut{j+1}.png"),
                                    os.path.join(root_folder,"cut_pic",f"{InPic_first_name}_cut_{i+1}.png")
                                    )
        might.append(pred)

    index_num=might.index(max(might))
    might.clear()
    click_list[j]=(pic_information[index_num][5],pic_information[index_num][6])
    name_list.append(f"{InPic_first_name}_cut_{index_num+1}.png")


print(click_list)

