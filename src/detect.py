#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from std_msgs.msg import Float64
from rostopic import get_topic_type
from matplotlib import pyplot as plt 
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        self.previous_spped = -1
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup        
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=1
            )

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            # rospy.get_param("~output_topic"), BoundingBoxes, queue_size=1
            rospy.get_param("~output_topic"), Float64, queue_size=1
        )
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=1
            )
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def callback(self, data):
        """adapted from yolov5/detect.py"""
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            # print(data.header)
            # print(data.height, ' * ', data.width) # 480 * 848
            # print(data.encoding) # bgr8
            # print(data.step) # 2544 = 848 * 3
            # print(data.is_bigendian) # 0
            # print(type(data.data)) # <class 'bytes'>
            # print(len(data.data)) # 1221120
            
            data.height = 640
            data.width = 640
            data.step = 1920
            data.encoding = 'rgb'
            data.is_bigendian = 0

            original_im = np.frombuffer(data.data, dtype=np.uint8)

            im = np.reshape(original_im, (data.height, data.step), 'F')            
            im = np.reshape(im, (data.height, data.width, 3), 'F')
            im = im[:, :, [2, 1, 0]]

            # im = np.reshape(original_im, (data.height, data.width, 3))
        
        im, im0 = self.preprocess(im)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       

            # Stream results
            im0 = annotator.result()

        # Publish prediction
        if len(bounding_boxes.bounding_boxes):
            for boundingbox in bounding_boxes.bounding_boxes:
                print(boundingbox.Class)
                if boundingbox.Class != self.previous_spped:
                    if boundingbox.Class.startswith("Speed Limit"):
                        speed = int(boundingbox.Class.split()[2])
                        self.pred_pub.publish(speed)
                        self.previous_spped = boundingbox.Class
                    elif boundingbox.Class == "Stop" or boundingbox.Class == "Red Light":
                        self.pred_pub.publish(0)
                        self.previous_spped = boundingbox.Class

        # Publish & visualize images
        if self.view_image:
            cv2.imshow("Detection", im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))

    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()
