import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.models as models

IMG_SIZE    = 640
S_LARGE     = 20
S_MEDIUM    = 40
S_SMALL     = 80
B           = 3
CLASS_NAMES = ["car", "pedestrian", "truck", "bicyclist"]
C           = len(CLASS_NAMES)

VEHICLE_IDS    = [0, 2]  # car, truck
PEDESTRIAN_IDS = [1, 3]  # pedestrian, bicyclist

VIDEO_PATH  = "test.mp4"
OUTPUT_PATH = "output.mp4"
CONF_THRESH = 0.4
NMS_THRESH  = 0.1

ALL_ANCHORS    = torch.load("anchors.pth")
ANCHORS_SMALL  = ALL_ANCHORS[0:3].numpy()
ANCHORS_MEDIUM = ALL_ANCHORS[3:6].numpy()
ANCHORS_LARGE  = ALL_ANCHORS[6:9].numpy()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True))

class SPPBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool1 = nn.MaxPool2d(5,  stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(9,  stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(13, stride=1, padding=6)
        self.conv  = conv_bn_relu(in_ch * 4, out_ch, kernel=1, padding=0)
    def forward(self, x):
        return self.conv(torch.cat([x, self.pool1(x), self.pool2(x), self.pool3(x)], dim=1))


#MODEL
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1,
                                    resnet.relu,  resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.lat5    = conv_bn_relu(2048, 256, kernel=1, padding=0)
        self.lat4    = conv_bn_relu(1024, 256, kernel=1, padding=0)
        self.lat3    = conv_bn_relu( 512, 256, kernel=1, padding=0)
        self.up5to4  = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4to3  = nn.Upsample(scale_factor=2, mode='nearest')
        self.refine4 = conv_bn_relu(256, 256)
        self.refine3 = conv_bn_relu(256, 256)
        self.spp     = SPPBlock(256, 256)

        def make_head():
            return nn.Sequential(
                conv_bn_relu(256, 512),
                conv_bn_relu(512, 256, kernel=1, padding=0),
                conv_bn_relu(256, 512),
                nn.Dropout2d(0.2),
                nn.Conv2d(512, B * (5 + C), 1, bias=True))

        self.head_large  = make_head()
        self.head_medium = make_head()
        self.head_small  = make_head()

    def _decode_head(self, x, S):
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1).contiguous().view(b, S, S, B, 5 + C)
        return torch.cat([torch.sigmoid(x[..., 0:2]), x[..., 2:4],
                          torch.sigmoid(x[..., 4:5]), torch.sigmoid(x[..., 5:])], dim=-1)

    def forward(self, x):
        x  = self.stem(x);   x  = self.layer1(x)
        c3 = self.layer2(x); c4 = self.layer3(c3); c5 = self.layer4(c4)
        p5 = self.spp(self.lat5(c5))
        p4 = self.refine4(self.lat4(c4) + self.up5to4(p5))
        p3 = self.refine3(self.lat3(c3) + self.up4to3(p4))
        return (self._decode_head(self.head_large(p5),  S_LARGE),
                self._decode_head(self.head_medium(p4), S_MEDIUM),
                self._decode_head(self.head_small(p3),  S_SMALL))


def decode_scale(pred, anchors_np, S):
    boxes, class_ids, scores = [], [], []
    for i in range(S):
        for j in range(S):
            for b in range(B):
                obj    = float(pred[i, j, b, 4])
                cls_p  = pred[i, j, b, 5:]
                cls_id = int(np.argmax(cls_p))
                score  = obj * float(cls_p[cls_id])
                if score < CONF_THRESH:
                    continue
                aw, ah = anchors_np[b]
                bx = (j + float(pred[i,j,b,0])) / S
                by = (i + float(pred[i,j,b,1])) / S
                bw = aw * np.exp(np.clip(float(pred[i,j,b,2]), -4, 4))
                bh = ah * np.exp(np.clip(float(pred[i,j,b,3]), -4, 4))
                x1 = (bx - bw/2) * IMG_SIZE
                y1 = (by - bh/2) * IMG_SIZE
                x2 = (bx + bw/2) * IMG_SIZE
                y2 = (by + bh/2) * IMG_SIZE
                boxes.append([x1, y1, x2, y2])
                class_ids.append(cls_id)
                scores.append(score)
    return boxes, class_ids, scores


def decode_all(pl, pm, ps):
    boxes, class_ids, scores = [], [], []
    for pred, anchors, S in [(pl, ANCHORS_LARGE,  S_LARGE),
                              (pm, ANCHORS_MEDIUM, S_MEDIUM),
                              (ps, ANCHORS_SMALL,  S_SMALL)]:
        b, c, s = decode_scale(pred, anchors, S)
        boxes += b; class_ids += c; scores += s
    return boxes, class_ids, scores


def nms(boxes, scores):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1)
        iou=(w*h)/(areas[i]+areas[order[1:]]-w*h+1e-7)
        order=order[1:][iou<=NMS_THRESH]
    return keep


model = Model().to(DEVICE)
model.load_state_dict(torch.load("best.pth", map_location=DEVICE))
model.eval()
print("Model loaded successfully")


#VIDEO
cap    = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps    = int(cap.get(cv2.CAP_PROP_FPS))
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
frame_delay = max(1, int(1000/fps))
frame_count = 0

#COUNT
count_log = open("counts.txt", "w")
count_log.write("frame,vehicles,pedestrians,total\n")

print(f"Video: {W}x{H} @ {fps}fps — press ESC to stop")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_r  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_in = (img_r.astype(np.float32)/255.0 - IMG_MEAN) / IMG_STD
        img_t  = torch.from_numpy(img_in).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pl, pm, ps = model(img_t)
        pl = pl[0].float().cpu().numpy()
        pm = pm[0].float().cpu().numpy()
        ps = ps[0].float().cpu().numpy()

        boxes, class_ids, scores = decode_all(pl, pm, ps)

        final_boxes, final_cls = [], []
        for cls in set(class_ids):
            mask = [k for k, c in enumerate(class_ids) if c == cls]
            cb   = [boxes[k] for k in mask]
            cs   = [scores[k] for k in mask]
            kept = nms(cb, cs)
            for k in kept:
                x1=int(cb[k][0]*W/IMG_SIZE); y1=int(cb[k][1]*H/IMG_SIZE)
                x2=int(cb[k][2]*W/IMG_SIZE); y2=int(cb[k][3]*H/IMG_SIZE)
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
                if x2>x1 and y2>y1:
                    final_boxes.append([x1,y1,x2,y2])
                    final_cls.append(cls)

        # Count
        vehicle_count    = sum(1 for c in final_cls if c in VEHICLE_IDS)
        pedestrian_count = sum(1 for c in final_cls if c in PEDESTRIAN_IDS)

        
        count_log.write(f"{frame_count},{vehicle_count},{pedestrian_count},{vehicle_count+pedestrian_count}\n")

        #Boxes
        for (x1,y1,x2,y2), cls in zip(final_boxes, final_cls):
            color = (0, 200, 255) if cls in VEHICLE_IDS else (0, 255, 100)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, CLASS_NAMES[cls], (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (260,80), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Vehicles:    {vehicle_count}",
                    (20,38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        cv2.putText(frame, f"Pedestrians: {pedestrian_count}",
                    (20,68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)

        out.write(frame)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(frame_delay) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
count_log.close()
print(f"\nDone — {frame_count} frames saved to {OUTPUT_PATH}")
print(f"Counts saved to counts.txt")