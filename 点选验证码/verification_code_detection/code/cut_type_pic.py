import xml.etree.ElementTree as ET
import os
import re
from PIL import Image
import shutil

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('删除失败 %s. 原因: %s' % (file_path, e))

def rename_files_time_rank(folder_path,type,name,change_tpye=None,number=0):
    #type为需要进行排列的后缀，而change_type为需要转换的后缀，如果只改前缀名，则不填
    import os
    import glob
    if change_tpye == None:
        change_tpye = type
    files_type = glob.glob(os.path.join(folder_path, f"*.{type}"))
    files_type.sort(key=os.path.getmtime)
    for i, file_path in enumerate(files_type):
        new_file_name = f"{name}{str(i).zfill(number)}.{change_tpye}"
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(file_path, new_file_path)
    print("文件已成功重命名。")

import os

def get_subfolder_names(parent_folder):
    # 初始化一个空列表来存储子文件夹的名字
    subfolder_names = []
    # 遍历父文件夹中的所有子文件夹
    for name in os.listdir(parent_folder):
        # 检查这是否是一个文件夹
        if os.path.isdir(os.path.join(parent_folder, name)):
            # 如果是，将其名字添加到列表中
            subfolder_names.append(name)
    # 返回子文件夹的名字列表
    return subfolder_names

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)

def get_max_number(path):
    max_number = -1
    for filename in os.listdir(path):
        if filename.startswith('pic_') and filename.endswith('.xml'):
            number = int(re.findall(r'\d+', filename)[0])
            if number > max_number:
                max_number = number
    return max_number

def cut_pitcture(input_path, output_path, start_x, start_y, end_x, end_y):
    with Image.open(input_path) as img:
        cropped_img = img.crop((start_x, start_y, end_x, end_y))
        cropped_img.save(output_path)

def cut_information(read_path):
    with open(read_path, 'r') as file:
        xml_data = file.read()
    root = ET.fromstring(xml_data)
    objects = root.findall('.//object')
    name = [obj.find('name').text for obj in objects]
    xmin = [int(obj.find('bndbox/xmin').text) for obj in objects]
    ymin = [int(obj.find('bndbox/ymin').text) for obj in objects]
    xmax = [int(obj.find('bndbox/xmax').text) for obj in objects]
    ymax = [int(obj.find('bndbox/ymax').text) for obj in objects]
    return name,xmin,ymin,xmax,ymax

root_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
        )
    )

read_folder=os.path.join(root_folder, "train","labelImg","data")
cut_folder=os.path.join(root_folder, "cut_pic")
input_folder=os.path.join(root_folder, "train","train_pic")

for index in range(1,get_max_number(read_folder)+1):
    read_name=f"pic_{index}.xml"
    pic=f"pic_{index}.png"
    pic_information=cut_information(os.path.join(read_folder,read_name))
    for i in range(3):
        cut_pitcture(os.path.join(input_folder,pic), 
                        os.path.join(cut_folder,f"{pic_information[0][i]}_pic_{index}_cut_{i+1}.png"), 
                        int(pic_information[1][i]) , 
                        int(pic_information[2][i]) , 
                        int(pic_information[3][i]) , 
                        int(pic_information[4][i]))
        resize_image(os.path.join(cut_folder,f"{pic_information[0][i]}_pic_{index}_cut_{i+1}.png"),
                     os.path.join(cut_folder,f"{pic_information[0][i]}_pic_{index}_cut_{i+1}.png"),
                     (105,105))
        

for filename in os.listdir(cut_folder):#再次重新规范下尺寸
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        file_path = os.path.join(cut_folder, filename)
        resize_image(file_path, file_path, (105, 105))  



if True:#将剪切好的数据移动
    src_dir = os.path.join(root_folder,"cut_pic")
    dst_parent_dir = os.path.join(root_folder,"Siamese-pytorch","datasets","images_background")
    for filename in os.listdir(src_dir):
        if '_pic' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            dst_dir_name = filename.split('_pic')[0]
            dst_dir = os.path.join(dst_parent_dir, dst_dir_name)
            os.makedirs(dst_dir, exist_ok=True)
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            shutil.move(src_file, dst_file)

if True:#将移动好的数据重命名
    subfolder_names = get_subfolder_names(os.path.join(root_folder,"Siamese-pytorch","datasets","images_background"))
    for i in range(len(subfolder_names)):
        rename_files_time_rank(os.path.join(os.path.join(root_folder,"Siamese-pytorch","datasets","images_background"),subfolder_names[i]),
                            "png",
                            f"{subfolder_names[i]}_",
                            )

clear_folder(cut_folder)