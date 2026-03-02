import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, models

IMG_SIZE = 128
S = 8
B = 2

CLASS_NAMES = ["car", "pedestrian", "truck", "bicyclist", "light"]
IGNORE_IDS = [3, 4]
C = len(CLASS_NAMES)

VIDEO_PATH = "test.mp4"      # your video
OUTPUT_PATH = "output.mp4"    # saved result

# model
def build_yolo_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    outputs = layers.Conv2D(B * (5 + C), 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

# decode
def decode_predictions(pred_tensor, conf_thresh=0.3):
    boxes, class_ids, scores = [], [], []

    for i in range(S):
        for j in range(S):
            for b in range(B):
                offset = b * (5 + C)
                bx, by, bw, bh = pred_tensor[i, j, offset:offset+4]
                obj_score = pred_tensor[i, j, offset+4]
                class_probs = pred_tensor[i, j, offset+5:offset+5+C]

                class_id = np.argmax(class_probs)
                score = obj_score * class_probs[class_id]

                if score > conf_thresh:
                    bw, bh = bw**2, bh**2

                    x1 = (j + bx - bw/2) * (IMG_SIZE / S)
                    y1 = (i + by - bh/2) * (IMG_SIZE / S)
                    x2 = (j + bx + bw/2) * (IMG_SIZE / S)
                    y2 = (i + by + bh/2) * (IMG_SIZE / S)

                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(class_id)
                    scores.append(score)

    return boxes, class_ids, scores

# load
model = build_yolo_model()
model.load_weights("trainweights.weights.h5")


cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    pred = model.predict(img_input, verbose=0)[0]
    boxes, class_ids, scores = decode_predictions(pred)

    for (x1, y1, x2, y2), c, s in zip(boxes, class_ids, scores):

        if c in IGNORE_IDS:
            continue

        x1 = int(x1 * W / IMG_SIZE)
        x2 = int(x2 * W / IMG_SIZE)
        y1 = int(y1 * H / IMG_SIZE)
        y2 = int(y2 * H / IMG_SIZE)

        label = f"{CLASS_NAMES[c]} {s:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)

    cv2.imshow("YOLO Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
