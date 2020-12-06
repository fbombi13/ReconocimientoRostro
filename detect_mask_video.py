# import the necessary packages
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response

app = Flask(__name__)

confidence_level = 0.5
prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
model_name = "mask_detector.model"
source_camera = 0  # It will take the first camera in your computer, if is not working use the number 1.
image_pivot = "caras.jpg"


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_level:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # predictions
    return locs, preds


def model_validation(frame):
    # load our serialized face detector model from disk
    faceNet = cv2.dnn.readNet(prototxt_path, weights_path)
    # load the face mask detector model from disk
    mask_net = load_model(model_name)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, mask_net)


def online_video_recognition():
    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=source_camera).start()
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, mask_net)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Con tapabocas" if mask > withoutMask else "Sin tapabocas"
            color = (0, 255, 0) if label == "Con tapabocas" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imwrite(image_pivot, frame)

        # key = cv2.waitKey(1) & 0xFF

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(image_pivot, 'rb').read() + b'\r\n')


@app.route('/')
def index():
    """Video streaming"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(online_video_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/recognition')
def api_recognition():
    return Response(online_video_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
