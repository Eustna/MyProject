import xml.etree.ElementTree as ET
import os
import re

def transform(read_path,out_path):
    with open(read_path, 'r') as file:
        xml_data = file.read()
    root = ET.fromstring(xml_data)
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    objects = root.findall('.//object')
    name = [obj.find('name').text for obj in objects]
    xmin = [int(obj.find('bndbox/xmin').text) for obj in objects]
    ymin = [int(obj.find('bndbox/ymin').text) for obj in objects]
    xmax = [int(obj.find('bndbox/xmax').text) for obj in objects]
    ymax = [int(obj.find('bndbox/ymax').text) for obj in objects]

    x = [(xmin[i] + xmax[i]) / 2 / width for i in range(len(objects))]
    y = [(ymin[i] + ymax[i]) / 2 / height for i in range(len(objects))]

    xx = [(xmax[i]-xmin[i])/ width for i in range(len(objects))]
    yy = [(ymax[i]-ymin[i])/ height for i in range(len(objects))]

    name_list = ['core', 'spaceship', 'video', 'bag', 'hurricane', 'threebuilding', 'gear', 'cap', 'star', 'magnet', 'tube', 'keybag', 'zbell', 'chef', 'twostar', 'cellphone', 'rugby', 'cloud', 'shield', 'bulb', 'key', 'prize', 'cola', 'watch', 'game', 'lifebuoy', 'hammar', 'xband', 'heart', 'moon', 'hand', 'volleyball', 'fire', 'smile', 'usave', 'radio', 'lock', 'reward', 'ladybug','planet']

    with open(out_path, 'w') as f:
        for i in range(len(objects)):
            name[i]=name_list.index(name[i])
            f.write(f"{name[i]} {round(x[i],6)} {round(y[i],6)} {round(xx[i],6)} {round(yy[i],6)}\n")

def get_max_number(path):
    max_number = -1
    for filename in os.listdir(path):
        if filename.startswith('pic_') and filename.endswith('.xml'):
            number = int(re.findall(r'\d+', filename)[0])
            if number > max_number:
                max_number = number
    return max_number


root_folder = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
        )
    )

input_folder=os.path.join(root_folder, "train","labelImg","data")
output_folder=os.path.join(root_folder, "train","label")
for index in range(1,get_max_number(input_folder)+1):
    input_name=f"pic_{index}.xml"
    output_name=f"pic_{index}.txt"
    transform(os.path.join(input_folder,input_name),os.path.join(output_folder,output_name))