from nn import  *
import torch
from utils.torch_utils import select_device
from utils.utils import non_max_suppression
import cv2
import numpy as  np
import glob
from hyp import  hyp
import os

import onnx
import onnxruntime

# save_predic_dir = "/home/fyl/source_code/yolo-with-landmark/test_imgs/predict_points/"
# if not os.path.exits():
#     os.makedirs(save_predic_dir)

device = select_device('cpu')


# net_type = "mbv3_large_1"
net_type =  "mbv3_small_75_light"  # mbv3_large_75" # mbv3_small_75_light

long_side = -1  # -1 mean origin shape
backone = None

assert net_type in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',"mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light']

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

print(weights)

net.load_state_dict(torch.load(weights, map_location=device)['model'])
net.eval()

dir = "./test_imgs/inputs/*.jpg"


imgs = glob.glob(dir)



for path in imgs:
    print(path)
    orig_image = cv2.imread(path)
    ori_h, ori_w, _ = orig_image.shape
    LONG_SIDE = long_side
    if long_side == -1:
        max_size = max(ori_w,ori_h)
        LONG_SIDE = max(32,max_size - max_size%32)
        print(LONG_SIDE)

    # if ori_h > ori_w:
    #     scale_h = LONG_SIDE / ori_h
    #     tar_w = int(ori_w * scale_h)
    #     tar_w = tar_w - tar_w % 32
    #     tar_w = max(32, tar_w)
    #     tar_h = LONG_SIDE


    # else:
    #     scale_w = LONG_SIDE / ori_w
    #     tar_h = int(ori_h * scale_w)
    #     tar_h = tar_h - tar_h % 32
    #     tar_h = max(32, tar_h)
    #     tar_w = LONG_SIDE

    # scale_w = tar_w * 1.0 / ori_w
    # scale_h = tar_h * 1.0 / ori_h

    # print(tar_w, tar_h, ori_w, ori_h)

    # image = cv2.resize(orig_image, (tar_w, tar_h))
    scale_w = 1
    scale_h = 1
    image = cv2.resize(orig_image, (ori_w, ori_h))

    image = cv2.resize(image, (320,320))

    orig_image = image




    image = image[...,::-1]
    image = image.astype(np.float64)
    # image = (image - hyp['mean']) / hyp['std']
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)


    # image = torch.from_numpy(image)
    # image = image.to(device).float()
    # pred = net(image)[0]

    # /home/fyl/source_code/yolo-with-landmark/onnx /home/fyl/source_code/yolo-with-landmark/onnx/shetou/mbv3_small_75_light.onnx

    image = image.astype(np.float32)
    session = onnxruntime.InferenceSession("/home/fyl/source_code/yolo-with-landmark/onnx/mbv3_small_75_light.onnx")
    # inputs = [inputs_.name for inputs_ in session.get_inputs()]
    inputs = session.get_inputs()[0].name
    print("inputs: ", inputs)
    outputs = [outputs.name for outputs in session.get_outputs()]
    print("outputs: ", outputs)
    print( "image.shape: " , image.shape) 
    preds = session.run(outputs, {inputs:  np.array(image, dtype=np.float32)})


    def sigmoid(data):
        return 1/(1+ np.exp(-data))    
    def yolo_decode(p, anchors, stride, nx, ny):
        no = 6
        # anchors = np.array([[12,12], [20,20], [32,32]])
        # stride = 8
        anchor_vec = anchors/stride
        # print(anchor_vec)
        anchor_wh = anchor_vec.reshape(1, len(anchors), 1, 1, 2)
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        grid = np.stack((yv, xv), 2).reshape((1, 1, nx, ny, 2))

        io = p # inference output
        io[..., :2] = sigmoid(io[..., :2]) + grid  # xy
        
        io[..., 2:4] = np.exp(io[..., 2:4]) * anchor_wh  # wh yolo method
        # print("fyl anchor wh", anchor_wh)
        # print("io[..., 2:4]: ", io[..., 2:4][0][0][9][9])

        io[..., :4] *=  stride
        io[..., 4:] = sigmoid(io[..., 4:])
        return io.reshape(1, -1, no)# view [1, 3, 13, 13, 85] as [1, 507, 85]

    # 对预测结果中 s32 进行 yolodecode
    pred = preds[2] # s32  [1 18 10 10]
    grid_scale = 10
    channel = 3
    anchors = [192, 320, 480]
    bias = anchors
    img_w = 320
    img_h = 320
    # 第一步 将得到的结果reshape一下
    print("before reshape: ",pred.shape) # (1, 18, 10, 10)
    pred_reshape = pred.reshape((3, 6, grid_scale,grid_scale)) # 
    pred = np.transpose(pred_reshape, (0, 2,3,1))
    print(pred.shape)
    print("after reshape: ", pred.shape) # 3, 10, 10, 6

    nx, ny = 10 , 10
    stride = 32
    # p_s32 = pred_ncnn[2].detach().numpy()
    p_s32 = pred
    print(p_s32[0][0][1]) # [0.75462  -1.1934  0.3045  -1.0056  -8.5776 -0.094802]
    anchors = np.array([[192,192], [320,320], [480,480]])

    # 第二步 将结果进行坐标解算decode
    yolo_s32 = yolo_decode(p_s32, anchors, stride, nx, ny)
    print("yolo_s32.shape: ", yolo_s32.shape, yolo_s32[0][1]) 
    # (1, 300, 6) [21.766      7.4446      260.34      70.239  0.00018824     0.47632]  
    # [     50.202      8.7741      416.73      45.042  0.00078196     0.50011]

    # 对预测结果中 s16 进行 yolodecode
    pred = preds[1] # s32  [1 18 10 10]
    grid_scale = 20
    channel = 3
    anchors = [48, 72, 128]
    bias = anchors
    # print("before reshape: ",pred.shape) # (1, 18, 10, 10)
    pred_reshape = pred.reshape((3, 6, grid_scale,grid_scale)) # 
    pred = np.transpose(pred_reshape, (0, 2,3,1))
    # print("after reshape: ", pred_reshape.shape) # 3, 10, 10, 6

    nx, ny = 20 , 20
    stride = 16
    p_s16 = pred
    anchors = np.array([[48,48], [72,72], [128,128]])
    yolo_s16 = yolo_decode(p_s16, anchors, stride, nx, ny)
    print("yolo_s32.shape: ", yolo_s16.shape, yolo_s16[0][1])

    # 对预测结果中 s8 进行 yolodecode
    pred = preds[0] # s32  [1 18 10 10]
    grid_scale = 40
    channel = 3
    anchors = [12, 20, 32]
    bias = anchors
    # print("before reshape: ",pred.shape) # (1, 18, 10, 10)
    pred_reshape = pred.reshape((3, 6, grid_scale,grid_scale)) # 
    pred = np.transpose(pred_reshape, (0, 2,3,1))
    # print("after reshape: ", pred_reshape.shape) # 3, 10, 10, 6

    nx, ny = 40 , 40
    stride = 8
    p_s8 = pred
    anchors = np.array([[12,12], [20,20], [32,32]])
    yolo_s8 = yolo_decode(p_s8, anchors, stride, nx, ny)
    print("yolo_s32.shape: ", yolo_s8.shape, yolo_s8[0][0])

    # 第三步 找到最大值
    yolo_out = [yolo_s8,yolo_s16,yolo_s32]
    yolo_out = np.concatenate((yolo_s8, yolo_s16), axis=1)
    yolo_out = np.concatenate((yolo_out, yolo_s32), axis=1)
    print(yolo_out.shape)
    confidences = yolo_out[:,:,4][0]
    print(confidences.shape)
    max_conf = max(confidences)
    indx = np.argmax(confidences)
    # indx = 5071
    print("max_conf: ", max_conf, confidences[indx], indx)
    print("max conf and it's x y w h ", yolo_out[:,:,0][0][indx], yolo_out[:,:,1][0][indx], yolo_out[:,:,2][0][indx], yolo_out[:,:,3][0][indx])


    # print(box_score[0])
    
    # for i in range(grid_scale):
    #     for j in range(grid_scale):
    #         conf = sigmoid(box_score[i][j])
    #         if conf > 0.8:
    #             cx = (i + sigmoid(x[i][j]))/grid_scale
    #             cy = (j + sigmoid(y[i][j]))/grid_scale
    #             w = np.exp(pred_w[i][j]) * bias[c]/grid_scale
    #             h = np.exp(pred_h[i][j]) * bias[c]/grid_scale
    #             print(conf, i, j)
    #             print("x: {}, y: {}, w: {}, h: {}".format(x[i][j], y[i][j], pred_w[i][j], pred_h[i][j]))
    #             print("x: {}, y: {}, w: {}, h: {}".format(cx, cy, w, h))
    #             # print("final x: {}, y: {}, w: {}, h: {}".format(cx*img_h, cy*img_h, w*img_w, h*img_h))
        

    # preds = np.concatenate((pred[0], pred[1], pred[2]), axis=0)

    # print(preds.shape)  # torch pred_size: [1 6300 6], onnx pred size [1, 18, 40, 40 ]    [1, 18, 20, 20]       [1, 18, 10, 10]
    # pred = non_max_suppression(pred, 0.3, 0.35, multi_label=False, classes=0, agnostic= False,land=True ,point_num= point_num)
    # print(pred)
    # try:
    #     det = pred[0].cpu().detach().numpy()
    #     orig_image = orig_image.astype(np.uint8)

    #     det[:,:4] = det[:,:4] / np.array([scale_w, scale_h] * 2)
    #     det[:,5:5+point_num*2] = det[:,5:5+point_num*2] / np.array([scale_w, scale_h] * point_num)
    # except:
    #     det = []
    
    # # filename = path[-10,-4]
    # # out_file = open(save_predic_dir+'%s.txt'%(filename), 'w')

    # for b in det:

    #     text = "{:.4f}".format(b[4])
    #     b = list(map(int, b))

    #     cv2.rectangle(orig_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #     cx = b[0]
    #     cy = b[1] + 12
    #     cv2.putText(orig_image, text, (cx, cy),
    #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #     # landms

    #     # print(b[11], b[12])
    #     # print(b[13], b[14])
    #     w , h = b[2] - b[0] , b[3] - b[1]
    #     # if w >64 or h >64 :
    #     #     for i in range(point_num):
    #     #         cv2.circle(orig_image, (b[5+i*2], b[5+i*2+1]), 1, (255, 255, 255), 2)
    #     for i in range(5, 5+point_num*2, 2):
    #         cv2.circle(orig_image, (b[i], b[i+1]), 1, (0,0,255), 4)
    #     # cv2.circle(orig_image, (b[5], b[6]), 1, (0, 0, 255), 4)
    #     # cv2.circle(orig_image, (b[7], b[8]), 1, (0, 255, 255), 4)
    #     # cv2.circle(orig_image, (b[9], b[10]), 1, (255, 0, 255), 4)
    #     # cv2.circle(orig_image, (b[11], b[12]), 1, (0, 255, 0), 4)
    #     # cv2.circle(orig_image, (b[13], b[14]), 1, (255, 0, 0), 4)
    #     # cv2.circle(orig_image, (b[15], b[16]), 1, (255, 255, 0), 4)
    #     # out_file.write("0" + " " + " ".join([  b[i]  for i in range(5, 5+point_num*2)]) + '\n')
    # # save image
    # # out_file.close()
    # cv2.imshow("frame", orig_image)
    # cv2.waitKey(0)
    # cv2.imwrite(path.replace("inputs","outputs"), orig_image)