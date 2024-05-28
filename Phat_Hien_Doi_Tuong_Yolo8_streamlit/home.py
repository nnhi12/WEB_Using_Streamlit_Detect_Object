import streamlit as st
from PIL import Image
import cv2
import numpy as np

postprocessing = 'yolov8'
mywidth  = 640
myheight = 640
background_label_id = -1

def load_yolo_model():
    classes = None
    with open('./Phat_Hien_Doi_Tuong_Yolo8_streamlit/object_detection_classes_yolo.txt', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    try:
        if st.session_state["LoadModel"] == True:
            print('Đã load model')
            pass
    except:
        st.session_state["LoadModel"] = True
        st.session_state["Net"] = cv2.dnn.readNet('./Phat_Hien_Doi_Tuong_Yolo8_streamlit/yolov8n.onnx')
        print('Load model lần đầu')

    st.session_state["Net"].setPreferableBackend(0)
    st.session_state["Net"].setPreferableTarget(0)
    outNames = st.session_state["Net"].getUnconnectedOutLayersNames()

    return classes, outNames

def postprocess(frame, outs, classes, outNames, confThreshold=0.5, nmsThreshold=0.4):
    frame = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = st.session_state["Net"].getLayerNames()
    lastLayerId = st.session_state["Net"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net"].getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # NMS is used inside Region layer only on DNN_BACKEND_OPENcv2 for another backends we need NMS in sample
    # or NMS is required if the number of outputs > 1
    if len(outNames) > 1 or lastLayer.type == 'Region' and 0 != cv2.dnn.DNN_BACKEND_OPENcv2:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return frame

def predict_image(image_file, classes, outNames):
    image = Image.open(image_file)
    st.image(image, caption=None)
    # Convert to cv2 for later use
    frame = np.array(image)
    frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
    frameHeight, frameWidth, _ = frame.shape
    if st.button('Predict'):
        # Process image.
        inpWidth = mywidth if mywidth else frameWidth
        inpHeight = myheight if myheight else frameHeight
        blob = cv2.dnn.blobFromImage(frame.copy(), size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)
        
        # Run a model
        st.session_state["Net"].setInput(blob, scalefactor=0.00392, mean=[0, 0, 0])
        outs = st.session_state["Net"].forward(outNames)
        img = postprocess(frame, outs, classes, outNames)
        st.image(img, caption=None, channels="BGR")

def main():
    classes, outNames = load_yolo_model()
    image_file = st.file_uploader("Upload Images", type=["bmp", "png", "jpg", "jpeg"])
    if image_file is not None:
        predict_image(image_file, classes, outNames)

