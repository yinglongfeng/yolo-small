from nn import  *
import torch
from utils.torch_utils import select_device
from utils.utils import non_max_suppression
import cv2
import numpy as  np
import glob
from hyp import  hyp
import time


# 参数初始化
# 相似度阈值
confThreshold = 0.2  # Confidence threshold

# NMS算法阈值
nmsThreshold = 0.3

# 输入图片的宽和高
inpWidth = 416
inpHeight = 416
factor_w = 1080/416
factor_h = 1920/416

# 导入物体类别class文件，默认支持80种
classesFile = "/home/fyl/source_code/darknet-yolo-v4/data/gesture.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# yolo v3的配置及weights文件

modelConfiguration = "/home/fyl/source_code/darknet-yolo-v4/cfg/yolov4-14-cls.cfg"
modelWeights = "/home/fyl/source_code/darknet-yolo-v4/results/all_hand_yolo_v4/yolov4-14-cls_best1.weights"

# modelConfiguration = "/home/fyl/source_code/darknet-yolo-v4/cfg/yolov3-14cls.cfg"
# modelWeights = "/home/fyl/source_code/darknet-yolo-v4/results/all_hand/yolov3-14cls-60000.backup"



# opencv读取外部模型
net_yolo = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# 画bounding box
def drawPred(classId, conf, left_x, left_y, width, height):
    # Draw a bounding box.
    cv2.rectangle(frame, (left_x, left_y), (width, height), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left_y = max(left_y, labelSize[1])
    cv2.rectangle(frame, (left_x, left_y - round(1.5 * labelSize[1])), (left_x + round(1.5 * labelSize[0]), left_y + baseLine),
               (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left_x, left_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# 使用NMS算法，丢弃低相似度的bounding box
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left_x = int(center_x - width / 2)
                left_y = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left_x, left_y, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    res = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left_x = box[0]
        left_y = box[1]
        width = box[2]
        height = box[3]
        finger_class = classes[classIds[i]]
        drawPred(classIds[i], confidences[i], left_x, left_y, left_x + width, left_y + height)
        box_info = [left_x, left_y, width, height, finger_class]
        res.append(box_info)
    return res




# device = select_device('cpu')
device = torch.device('cuda:0') 


# net_type = "mbv3_large_1"
net_type = "mbv3_large_75"
long_side = -1  # -1 mean origin shape
backone = None

assert net_type in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',
                   "mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light',
                   ]



if net_type.startswith("mbv3_small_1"):
    backone = mobilenetv3_small()
elif net_type.startswith("mbv3_small_75"):
    backone = mobilenetv3_small( width_mult=0.75)
elif net_type.startswith("mbv3_large_1"):
    backone = mobilenetv3_large()
elif net_type.startswith("mbv3_large_75"):
    backone = mobilenetv3_large( width_mult=0.75)
elif net_type.startswith("mbv3_large_f"):
    backone = mobilenetv3_large_full()

if 'light' in net_type:
    net = DarknetWithShh(backone, hyp, light_head=True).to(device)
else:
    net = DarknetWithShh(backone, hyp).to(device)


point_num = hyp['point_num']
weights = "./weights_test/{}_last.pt".format(net_type)

net.load_state_dict(torch.load(weights, map_location=device)['model'])
net.eval()

## 实时视频处理

capture = cv2.VideoCapture(0)

while(1):
# for path in imgs:
#     print(path)
    ret, orig_image = capture.read()
    if not ret:
        continue
    ori_h, ori_w, _ = orig_image.shape
    ori_h = ori_h * 0.7
    ori_w = ori_w * 0.7
    orig_image = cv2.resize(orig_image, (int(ori_w),int(ori_h)))
    frame = orig_image
    ori_h, ori_w, _ = orig_image.shape
    LONG_SIDE = long_side
    if long_side == -1:
        max_size = max(ori_w,ori_h)
        LONG_SIDE = max(32,max_size - max_size%32)

    if ori_h > ori_w:
        scale_h = LONG_SIDE / ori_h
        tar_w = int(ori_w * scale_h)
        tar_w = tar_w - tar_w % 32
        tar_w = max(32, tar_w)
        tar_h = LONG_SIDE


    else:
        scale_w = LONG_SIDE / ori_w
        tar_h = int(ori_h * scale_w)
        tar_h = tar_h - tar_h % 32
        tar_h = max(32, tar_h)
        tar_w = LONG_SIDE

    scale_w = tar_w * 1.0 / ori_w
    scale_h = tar_h * 1.0 / ori_h

    image = cv2.resize(orig_image, (tar_w, tar_h))


    # YOLO results
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net_yolo.setInput(blob)
    start = time.time()
    outs = net_yolo.forward(getOutputsNames(net_yolo))
    boundingbox = []
    boundingbox = postprocess(frame, outs)



    image = image[...,::-1]
    image = image.astype(np.float64)
    # image = (image - hyp['mean']) / hyp['std']
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])

    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image)

    image = image.to(device).float()
    pred = net(image)[0]

    end = time.time()

    print("fps is {}".format(1/(end-start)))


    pred = non_max_suppression(pred,0.25, 0.35,
                                       multi_label=False, classes=0, agnostic= False,land=True ,point_num= point_num)
    try:
        det = pred[0].cpu().detach().numpy()
        orig_image = orig_image.astype(np.uint8)

        det[:,:4] = det[:,:4] / np.array([scale_w, scale_h] * 2)
        det[:,5:5+point_num*2] = det[:,5:5+point_num*2] / np.array([scale_w, scale_h] * point_num)
    except:
        det = []
    for b in det:

        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        # cv2.rectangle(orig_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        # cv2.putText(orig_image, text, (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms

        # print(b[11], b[12])
        # print(b[13], b[14])
        w , h = b[2] - b[0] , b[3] - b[1]
        # if w >64 or h >64 :
        #     for i in range(point_num):
        #         cv2.circle(orig_image, (b[5+i*2], b[5+i*2+1]), 1, (255, 255, 255), 2)
        cv2.circle(orig_image, (b[5], b[6]), 2, (0, 0, 255), 4)
        cv2.circle(orig_image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(orig_image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(orig_image, (b[11], b[12]), 2, (0, 255, 0), 4)
        cv2.circle(orig_image, (b[13], b[14]), 1, (255, 0, 0), 4)
        cv2.circle(orig_image, (b[15], b[16]), 1, (255, 255, 0), 4)
    cv2.imshow("frame", orig_image)
    # cv2.imshow("frame2", orig_image)
    cv2.waitKey(1)
    # save image

    # cv2.imwrite(path.replace("inputs","outputs"), orig_image)
