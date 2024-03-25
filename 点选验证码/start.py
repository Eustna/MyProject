import pyautogui
import time
import os
import re
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pyautogui.FAILSAFE = False

import cv2
from PIL import Image
import time
import onnxruntime as ort
import numpy as np
import os
import sys

root_folder=os.path.join(os.path.dirname(
        os.path.abspath(__file__)
        ),"verification_code_detection")


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
InPic_first_name = "pic_input"
InPic_name = InPic_first_name+".png"


model_path = os.path.join(root_folder, "model",f"{model_name}.onnx")
class_path = os.path.join(root_folder, "train","classes.txt")
InPic_path = os.path.join(root_folder, "input_pic",InPic_name)
OutPic_path= os.path.join(root_folder, "output_pic",InPic_name)

def get_click_inf():
    picture = getTpInfo()
    pic_information=picture.box_pitcture(InPic_path,OutPic_path)

    cut_folder=os.path.join(root_folder,"cut_pic")
    cut_pic = cv2.imread(InPic_path)
    heigth = cut_pic.shape[0]
    width = cut_pic.shape[1]

    # obj=1
    # obj_center=0
    # obj_Right=0
    # obj_Left=0
    if pic_information[0] == 0:
        print("没有检测到")
    elif pic_information[1] == 0:
        print("检测到1个")
        index = 1
        # obj_1=pic_information[0]
        # obj_center_1=(pic_information[0][5],pic_information[0][6])
        # obj_Left_1  =(pic_information[0][0],pic_information[0][1])
        # obj_Right_1 =(pic_information[0][2],pic_information[0][3])


    elif pic_information[2] == 0:
        print("检测到2个")
        index = 2
        # obj_1=pic_information[0]
        # obj_center_1=(pic_information[0][5],pic_information[0][6])
        # obj_Left_1  =(pic_information[0][0],pic_information[0][1])
        # obj_Right_1 =(pic_information[0][2],pic_information[0][3])

        # obj_2=pic_information[1]
        # obj_center_2=(pic_information[1][5],pic_information[1][6])
        # obj_Left_2  =(pic_information[1][0],pic_information[1][1])
        # obj_Right_2 =(pic_information[1][2],pic_information[1][3])

    else:
        index=3
        # obj_1=pic_information[0]
        # obj_center_1=(pic_information[0][5],pic_information[0][6])
        # obj_Left_1  =(pic_information[0][0],pic_information[0][1])
        # obj_Right_1 =(pic_information[0][2],pic_information[0][3])

        # obj_2=pic_information[1]
        # obj_center_2=(pic_information[1][5],pic_information[1][6])
        # obj_Left_2  =(pic_information[1][0],pic_information[1][1])
        # obj_Right_2 =(pic_information[1][2],pic_information[1][3])

        # obj_3=pic_information[2]
        # obj_center_3=(pic_information[2][5],pic_information[2][6])
        # obj_Left_3  =(pic_information[2][0],pic_information[2][1])
        # obj_Right_3 =(pic_information[2][2],pic_information[2][3])

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
                pred=predict.similar(os.path.join(cut_folder,f"cut_{j+1}.png"),
                                        os.path.join(root_folder,"cut_pic",f"{InPic_first_name}_cut_{i+1}.png")
                                        )
            might.append(pred)

        index_num=might.index(max(might))
        might.clear()
        click_list[j]=(pic_information[index_num][5]*width,pic_information[index_num][6]*heigth)
        name_list.append(f"{InPic_first_name}_cut_{index_num+1}.png")
    return click_list

def get_photo(lt_x,lt_y,rb_x,rb_y):
    left_top_x = lt_x
    left_top_y = lt_y
    right_bottom_x = rb_x
    right_bottom_y = rb_y
    width = right_bottom_x - left_top_x
    height = right_bottom_y - left_top_y
    screenshot = pyautogui.screenshot(region=(left_top_x, left_top_y, width, height))
    screenshot.save(os.path.join(rf,"photo","catch_image.png"))
    img = Image.open(os.path.join(rf,"photo","catch_image.png"))
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img, lang='chi_sim')
    return text

def cut_photo(path,lt_x,lt_y,rb_x,rb_y,name,index=None):
    left_top_x = lt_x
    left_top_y = lt_y
    right_bottom_x = rb_x
    right_bottom_y = rb_y
    width = right_bottom_x - left_top_x
    height = right_bottom_y - left_top_y
    screenshot = pyautogui.screenshot(region=(left_top_x, left_top_y, width, height))
    screenshot.save(os.path.join(path,f"{name}_{index}.png"))
    
def get_max_number(path):
    max_number = -1
    for filename in os.listdir(path):
        if filename.startswith('pic_') and filename.endswith('.png'):
            number = int(re.findall(r'\d+', filename)[0])
            if number > max_number:
                max_number = number
    return max_number

# while True:#获取坐标
#     print(pyautogui.position())
rf= os.path.dirname(
        os.path.abspath(__file__)
        )

index=get_max_number(os.path.join(rf,"cut"))
# watch_video_box=[126+960,587,258+960,611]
# watch_video_bt=[249+960,599]
watch_video_box=[1079,743,1226,771]
watch_video_bt=[1159,752]
youtube_box=[127+960,1032,214+960,1061]
youtube_bt=[459+960,632]
end_box=[97+960,106+40,803+960,173+40]
x_bt=[548+960,27]
skip_video_bt=[852+960,131+40]
ten_box=[32+960,119+40,71+960,153+40]
error_box=[310+960,608,629+960,1060]
check_box=[332+960,686+170,598+960,735+170]
img=[340+960,468+170,582+960,666+170]
cut1=[412+960,740+170,444+960,772+170]
cut2=[444+960,740+170,476+960,772+170]
cut3=[476+960,740+170,508+960,772+170]
next_vedio_bt = [1411,1032]
next_vedio_box = [1332,1018,1513,1051]
refresh_bt = [1033,62]
#record  638

while True:
    time.sleep(1)
    round_time=1
    print("开始检测进入看视频")
    while True:
        round_time += 1
        time.sleep(1)
        text=get_photo(watch_video_box[0],
                       watch_video_box[1],
                       watch_video_box[2],
                       watch_video_box[3])
        print(f"检测值为{text}")
        if "Watch video" in text:
            print("检测进入看视频成功")
            time.sleep(1)
            pyautogui.click(watch_video_bt[0],watch_video_bt[1])
            break
        elif round_time >=2:
            break
            # pyautogui.click(x_bt[0],x_bt[1])
        else:
            text=get_photo(next_vedio_box[0],
                next_vedio_box[1],
                next_vedio_box[2],
                next_vedio_box[3])
            print(f"检测值为{text}")
            if "video" in text:
                print("检测进入看视频成功")
                time.sleep(1)
                pyautogui.click(watch_video_bt[0],watch_video_bt[1])
                break
            elif round_time >=2:
                break
                # pyautogui.click(x_bt[0],x_bt[1])
            else:
                time.sleep(0.5)

    time.sleep(2)
    print("开始等待检测油管")
    round_time=1
    while True:
        text=get_photo(youtube_box[0],
                       youtube_box[1],
                       youtube_box[2],
                       youtube_box[3])
        print(f"检测值为{text}")
        if "b" in text or "u"in text or "Y"in text or "e"in text or "h"in text or "i"in text:#检测油管
            print("检测油管成功，点击播放视频")
            time.sleep(0.5)
            pyautogui.click(youtube_bt[0],youtube_bt[1])
            break
        elif round_time >= 5:
            pyautogui.click(x_bt[0],x_bt[1])
            time.sleep(1)
            pyautogui.click(refresh_bt[0],refresh_bt[1])
            round_time =1
            text=get_photo(watch_video_box[0],
                            watch_video_box[1],
                            watch_video_box[2],
                            watch_video_box[3])
            if "Watch video" in text:
                print("检测进入看视频成功")
                time.sleep(1)
                pyautogui.click(watch_video_bt[0],watch_video_bt[1])
        else:
            time.sleep(0.5)
            round_time += 1
    time.sleep(11)

    print("开始检测是否结束")
    round_time=1
    while True:
        round_time +=1
        text=get_photo(end_box[0],
                       end_box[1],
                       end_box[2],
                       end_box[3])
        print(f"检测值为{text}")
        if "has been counted" in text:
            print("视频结束成功")
            pyautogui.click(next_vedio_bt[0],next_vedio_bt[1])
            break
        else:
            print("视频结束失败")
            time.sleep(1)
            if True:
                print("开始检测视频是否卡住")
                text=get_photo(ten_box[0],
                               ten_box[1],
                               ten_box[2],
                               ten_box[3])
                print(f"检测值为{text}")
                if "10" in text:#检测卡视频
                    print("视频卡住，点击跳过")
                    time.sleep(0.5)
                    pyautogui.click(next_vedio_bt[0],next_vedio_bt[1])
                    break
                elif round_time == 20:
                    print("视频卡住，点击跳过")
                    time.sleep(0.5)
                    pyautogui.click(next_vedio_bt[0],next_vedio_bt[1])
                    break
            if True:
                print("开始检测视频是否异常")
                text=get_photo(error_box[0],
                               error_box[1],
                               error_box[2],
                               error_box[3])
                print(f"检测值为{text}")
                if "Verification" in text:
                    print("视频异常，点击跳过")
                    time.sleep(0.5)
                    pyautogui.click(next_vedio_bt[0],next_vedio_bt[1])
                    break
                elif "over"in text or "Bp" in text:
                    print("视频异常，点击跳过")
                    time.sleep(0.5)
                    pyautogui.click(next_vedio_bt[0],next_vedio_bt[1])
                    break
            if True:
                print("开始检测是否需要验证")
                if "icons" in get_photo(check_box[0],
                                        check_box[1],
                                        check_box[2],
                                        check_box[3]):
                    index +=1
                    cut_photo(os.path.join(rf,"verification_code_detection","input_pic"),
                              img[0],
                              img[1],
                              img[2],
                              img[3],"pic","input")
                    cut_photo(os.path.join(rf,"verification_code_detection","cut_pic"),
                              cut1[0],
                              cut1[1],
                              cut1[2],
                              cut1[3],"cut","1")
                    cut_photo(os.path.join(rf,"verification_code_detection","cut_pic"),
                              cut2[0],
                              cut2[1],
                              cut2[2],
                              cut2[3],"cut","2")
                    cut_photo(os.path.join(rf,"verification_code_detection","cut_pic"),
                              cut3[0],
                              cut3[1],
                              cut3[2],
                              cut3[3],"cut","3")
                    print("截图完毕")

                    time.sleep(0.2)

                    list_of_click=get_click_inf()
                    pyautogui.click(list_of_click[0][0]+img[0],list_of_click[0][1]+img[1])
                    time.sleep(0.2)
                    pyautogui.click(list_of_click[1][0]+img[0],list_of_click[1][1]+img[1])
                    time.sleep(0.2)
                    pyautogui.click(list_of_click[2][0]+img[0],list_of_click[2][1]+img[1])
                    # pyautogui.click(skip_video_bt[0],skip_video_bt[1])#删掉后添加检验