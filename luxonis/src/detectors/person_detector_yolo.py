"""
Tiny-yolo-v4 device side decoding demo
The code is the same as for Tiny-yolo-V3, the only difference is the blob file.
The blob was compiled following this tutorial: https://github.com/TNTWEN/OpenVINO-YOLOV4
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import os
from .utils import getPath

labelsPath = getPath("coco.names")
labelMap = open(labelsPath).read().strip().split("\n")


def compute_image(callback):
    syncNN = True

    nnPath = getPath('yolo-v3.blob')
    if not Path(nnPath).exists():
        raise FileNotFoundError(
            f'Required file/s not found')

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(416, 416)
    camRgb.setInterleaved(False)
    camRgb.setFps(1)

    # Network specific settings
    detectionNetwork = pipeline.createYoloDetectionNetwork()
    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setAnchors(
        np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    detectionNetwork.setAnchorMasks(
        {"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    detectionNetwork.setIouThreshold(0.5)

    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    camRgb.preview.link(detectionNetwork.input)

    # Create outputs
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    if syncNN:
        detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    nnOut = pipeline.createXLinkOut()
    nnOut.setStreamName("detections")
    detectionNetwork.out.link(nnOut.input)

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        # Start pipeline
        device.startPipeline()

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(
            name="detections", maxSize=4, blocking=False)

        frame = None
        detections = []

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def displayFrame(name, frame):
            for detection in detections:
                bbox = frameNorm(
                    frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, labelMap[detection.label], (bbox[0] +
                                                               10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%",
                            (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.imshow(name, frame)
            callback(frame)

        startTime = time.monotonic()
        counter = 0

        while True:
            if syncNN:
                inRgb = qRgb.get()
                inDet = qDet.get()
            else:
                inRgb = qRgb.tryGet()
                inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            if inDet is not None:
                detections = inDet.detections
                counter += 1

            if frame is not None:
                displayFrame("rgb", frame)
