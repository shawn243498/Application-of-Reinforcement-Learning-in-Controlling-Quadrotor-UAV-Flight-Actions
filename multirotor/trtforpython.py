import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from collections import OrderedDict,namedtuple
from rtsp import ipcamCapture


def TRTInitial():
    print(trt.__version__)

    win_title = 'YOLOv7 tensorRT CUSTOM DETECTOR'

    # put your trt file
    w = './AirSimBest-nms.trt'
    device = torch.device('cuda:0')

    # Infer TensorRT Engine
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()

    names = ['Target', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']
    colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

    # warmup for 10 times
    for _ in range(5):
        tmp = torch.randn(1, 3, 640, 640).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

    return device, binding_addrs, context, bindings, names, colors, win_title

# image box processing
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def postprocess(boxes,r,dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= r
    return boxes

def Detect(I, device, binding_addrs, context, bindings, names, colors, win_title, ori_area, show_detect):
    # t_prev = time.time()
    All = np.empty((2, 2), dtype=int)
    Center = np.empty((1, 2), dtype=int)
    OriArea2 = np.empty((1, 2), dtype=int)

    frame_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    image = frame_rgb.copy()                            # is necercery but didn't how it work
    image, ratio, dwdh = letterbox(image, auto=False)   # image preprocessing
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im.shape

    im = torch.from_numpy(im).to(device)
    im/=255

    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))

    nums = bindings['num_dets'].data
    boxes = bindings['det_boxes'].data
    scores = bindings['det_scores'].data
    classes = bindings['det_classes'].data

    boxes = boxes[0,:nums[0][0]]
    scores = scores[0,:nums[0][0]]
    classes = classes[0,:nums[0][0]]

    for box,score,cl in zip(boxes,scores,classes):
        box = postprocess(box,ratio,dwdh).round().int()
        name = names[cl]

        if name == 'Target' and score > 0.6:
            show_detect = True
            name += ' ' + str(round(float(score), 3))

            l_x = int(box[0])
            l_y = int(box[1])
            r_x = int(box[2])
            r_y = int(box[3])
            o_w = int(r_x - l_x)
            o_h = int(r_y - l_y)

            area = o_w * o_h
            # If the detect box is too large, the system will filter its target information.
            # System will choose the target which has biggest detect box to be the track target.
            if area < 80000 and area > ori_area:
                if area > ori_area:
                    ori_area = area
                    OriArea2[0, 0] = o_w
                    OriArea2[0, 1] = o_h
                    All[0, 0] = l_x
                    All[0, 1] = l_y
                    All[1, 0] = r_x
                    All[1, 1] = r_y
                    CenterX = int((l_x+r_x)/2)
                    CenterY = int((l_y+r_y)/2)
                    Center[0, 0] = CenterX
                    Center[0, 1] = CenterY

    if show_detect is True:
        cv2.circle(frame_rgb, (Center[0, 0], Center[0, 1]), 5, (255, 0, 255), -1)
        cv2.rectangle(frame_rgb, (All[0, 0], All[0, 1]), (All[1, 0], All[1, 1]), (0, 255, 255), 2)

    ## FPS
    # s = float(time.time() - t_prev)
    # fps = int(1 / s)
    # cv2.putText(frame_rgb, f'fps :  {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img2 = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    return img2, OriArea2, All

# for Test
if __name__ == '__main__':
    device, binding_addrs, context, bindings, names, colors, win_title = TRTInitial()

    URL = 0
    # 連接攝影機
    ipcam = ipcamCapture(URL)
    # 啟動子執行緒
    ipcam.start()
    # 暫停1秒，確保影像已經填充
    time.sleep(1)
    # 使用無窮迴圈擷取影像，直到按下Esc鍵結束
    while True:
        ori_area = 0
        show_detect = False

        I = ipcam.getframe()
        img2, ori_area, All = Detect(I, device, binding_addrs, context, bindings, names, colors, win_title, ori_area, show_detect)
        cv2.imshow(win_title, img2)
        if cv2.waitKey(10) == 27:
            ipcam.stop()
            cv2.destroyAllWindows()
            break