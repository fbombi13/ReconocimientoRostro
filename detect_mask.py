# import the necessary packages
import cv2
import imutils
import numpy as np
from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

face_net = None
mask_net = None
confidence_level = 0.5
prototxt_path = r"face_detector\deploy.prototxt"
weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
model_name = "mask_detector.model"
source_camera = 0  # It will take the first camera in your computer, if is not working use the number 1.
image_pivot = "caras.jpg"
red_color = (0, 0, 255)
green_color = (0, 255, 0)
con_tapabocas = "Con tapabocas"
sin_tapabocas = "Sin tapabocas"


def detect_and_predict_mask(frame):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    results = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_level:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)
        for prediction in preds:
            (mask, withoutMask) = prediction
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = con_tapabocas if mask > withoutMask else sin_tapabocas
            results.append(label)
    # return a 3-tuple of the face locations and their corresponding predictions and results
    return locs, preds, results


def model_initialization():
    # load our serialized face detector model from disk
    face_network = cv2.dnn.readNet(prototxt_path, weights_path)
    # load the face mask detector model from disk
    mask_network = load_model(model_name)

    return face_network, mask_network


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

        # detect faces in the frame and determine if they are wearing a face mask or not

        (locs, preds, results) = detect_and_predict_mask(frame)

        # loop over the detected face locations and their corresponding locations
        for (box, pred, result) in zip(locs, preds, results):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
            color = green_color if result == con_tapabocas else red_color

            # include the probability in the label
            label = "{}: {:.2f}%".format(result, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imwrite(image_pivot, frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(image_pivot, 'rb').read() + b'\r\n')


@app.route('/')
def index():
    """Video streaming"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(online_video_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/recognition', methods=['POST'])
def api_recognition():
    data = request.args.get('image')
    image = cv2.imread(data)
    (locs, preds, results) = detect_and_predict_mask(image)

    return Response(results, mimetype='text/plain')


if __name__ == '__main__':
    face_net, mask_net = model_initialization()
    app.run()
