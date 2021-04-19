import cv2
import depthai as dai
import numpy as np
import time

from .utils import getPath
from .pose import getKeypoints, getValidPairs, getPersonwiseKeypoints


def compute_image(callback):

    # Start defining a pipeline
    pipeline = dai.Pipeline()

        # Define a source - color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(456, 256)
    camRgb.setInterleaved(False)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setFps(20)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)
    
    pose_nn = pipeline.createNeuralNetwork()
    pose_nn.setBlobPath(getPath("human-pose-estimation-0001.blob"))
    pose_nn.setNumInferenceThreads(2)
    # Specify that network takes latest arriving frame in non-blocking manner
    pose_nn.input.setQueueSize(1)
    pose_nn.input.setBlocking(False)
    camRgb.preview.link(pose_nn.input)

    pose_nn_xout = pipeline.createXLinkOut()
    pose_nn_xout.setStreamName("pose")
    pose_nn.out.link(pose_nn_xout.input)



    colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]

    POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
                  [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

    running = True
    pose = None
    keypoints_list = None
    detected_keypoints = None
    personwiseKeypoints = None

    # Pipeline is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        device.startPipeline()

        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        pose = device.getOutputQueue(name="pose", maxSize=4, blocking=False)

        startTime = time.monotonic()


        while True:
            inRaw = pose.tryGet()
            inRgb = qRgb.tryGet()

            if inRaw is not None and inRgb is not None:
                frame = qRgb.getCvFrame()

                heatmaps = np.array(inRaw.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
                pafs = np.array(inRaw.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
                heatmaps = heatmaps.astype('float32')
                pafs = pafs.astype('float32')
                outputs = np.concatenate((heatmaps, pafs), axis=1)
                 
                new_keypoints = []
                new_keypoints_list = np.zeros((0, 3))
                keypoint_id = 0

                h, w = frame.shape[:2]  # 256, 456


                for row in range(18):
                    probMap = outputs[0, row, :, :]
                    probMap = cv2.resize(probMap, (w, h))  # (456, 256)
                    keypoints = getKeypoints(probMap, 0.3)
                    new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
                    keypoints_with_id = []

                    for i in range(len(keypoints)):
                        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                        keypoint_id += 1

                    new_keypoints.append(keypoints_with_id)

                valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
                newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

                detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)

                if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                    for i in range(18):
                        for j in range(len(detected_keypoints[i])):
                            cv2.circle(frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
                    for i in range(17):
                        for n in range(len(personwiseKeypoints)):
                            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                            if -1 in index:
                                continue
                            B = np.int32(keypoints_list[index.astype(int), 0])
                            A = np.int32(keypoints_list[index.astype(int), 1])
                            cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
            
                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))