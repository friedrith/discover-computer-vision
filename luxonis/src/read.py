import cv2
import sys

from detectors.person_detector_tiny_yolo import compute_image as compute_image_tiny_yolo
from detectors.person_detector_yolo import compute_image as compute_image_yolo
from detectors.camera import compute_image as compute_image_basic
from detectors.depth_camera import compute_image as compute_image_depth
from detectors.person_depth import compute_image as compute_person_depth
from detectors.spatial_location_calculator import compute_image as compute_location


def display_image(frame):
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        sys.exit()


compute_location(display_image)
