'''
Author: your name
Date: 2020-08-12 16:51:39
LastEditTime: 2020-08-12 17:43:23
LastEditors: Please set LastEditors
Description: 评估预测关键点的rmse误差，当然要注意，如果有两个点时，此处是直接跳过，避免误匹配的过程
FilePath: /yolo-with-landmark/evaluation_on_hand_points_2.py
'''
import os
import xml.etree.ElementTree as ET
import numpy as np

gt_xml_path = "/home/fyl/data/RHD_v1-1/RHD_published_v2/zuoyi_changed/Annotations"

pred_txt_path = "/home/fyl/source_code/yolo-with-landmark/test_imgs/predict_points"

prefix = "rhd_training_000"

num_points = 6
img_width = 320

errors = np.array([0])

for _,_,files in os.walk(pred_txt_path):
    for file in files:
        #　预测数值
        print(pred_txt_path+"/"+file)
        
        pred_lines_f = open(pred_txt_path+"/"+file)
        pred_lines = pred_lines_f.readlines()
        print(pred_lines)
        if (len(pred_lines) >= 2 or not pred_lines):
            continue
        pred_str = pred_lines[0].split(" ")
        pred_points = [ float(x)  for x in pred_str ]
        print(pred_points)



        ## gt数值
        f_gt = prefix + file
        f_gt = f_gt.replace("txt", "xml", 1)
        in_file = open("%s/%s"%(gt_xml_path,f_gt))
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            name = obj.find('name').text

            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            xmlbox_points = obj.find('points')
            points = []
            px = ['p'+str(i) for i in [0, 1, 5, 9, 13, 17]]
            # px = ['p'+str(i) for i in [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]]
            for pxx in px:
                # print( xmlbox_points.find(pxx).text)
                p0 = xmlbox_points.find(pxx).text.split(',')
                float_px = float(p0[0])
                float_py = float(p0[1])
                points.append(float_px)
                points.append(float_py)
            # print(points)
            gt = np.array(points)
            pred = np.array(pred_points)
            tmp_err = abs(gt - pred)
            # print(tmp_err)
            # exit(0)
            errors  = np.append(errors, tmp_err)
            # print(errors)

print(errors.shape)

Error = errors * errors

RMSE = np.sqrt(np.sum(Error)/Error.shape[0])

print(RMSE)