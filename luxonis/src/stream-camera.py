from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
import os
import numpy as np
import depthai as dai

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
globalFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to

# warmup
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def compute_image():
    # grab global references to the video stream, output frame, and
    # lock variables
    global globalFrame, lock

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(1920, 1080)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Create output
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

            # Retrieve 'bgr' (opencv format) frame
            # cv2.imshow("bgr", inRgb.getCvFrame())
            with lock:
                globalFrame = inRgb.getCvFrame()

            # if cv2.waitKey(1) == ord('q'):
            # 		break


def generate():
    # grab global references to the output frame and lock variables
    global globalFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if globalFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", globalFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/save", methods=['POST'])
def save():
    cv2.imwrite('image.jpg', globalFrame)
    # return the response generated along with the specific media
    # type (mime type)
    return Response("{}",
                    status=200,
                    mimetype='application/json')


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default='0.0.0.0',
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=8080,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("--rotate", type=int, default=180,
                    help="angle of rotation of the image")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    ap.add_argument('-c', '--confidence', type=float, default=0.5,
                    help='minimum probability to filter weak detections')
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=compute_image)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args['host'], port=args['port'], debug=True,
            threaded=True, use_reloader=False)
