import math
import time
import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO
import torch
from libs.utils import draw_detections, sigmoid
from PIL import Image, ImageDraw

class YOLOSeg:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, input_height=768, input_width=1280):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_height = input_height
        self.input_width = input_width
        self.input_shape = [1, 3, self.input_height, self.input_width]

        # Initialize model
        self.model = YOLO(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def segment_objects(self, image):
        image = self.prepare_input(image)
        outputs = self.inference(image)
        self.boxes, self.scores, self.class_ids, mask_pred, self.ids = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred)
        self.class_ids = self.class_ids.numpy().astype(int)
        self.boxes = self.boxes.numpy()
        self.scores = self.scores.numpy()
        self.ids = self.ids.numpy()
        return self.boxes, self.scores, self.class_ids, self.mask_maps, self.ids

    def inference(self, input_tensor):
        outputs = self.model.track(
            input_tensor, 
            imgsz=(self.input_height, self.input_width), 
            persist=True,
            iou=self.iou_threshold,
            conf=self.conf_threshold
            )
        return outputs

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        # input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        return input_img

    def process_box_output(self, results):
        box_predictions = results.boxes.xyxy
        mask_predictions = results.masks.xy
        # Get the class with the highest confidence
        class_ids = results.boxes.cls
        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)
        return boxes, results.boxes.conf, class_ids, mask_predictions, results.boxes.id

    def process_mask_output(self, mask_output):
        mask_width = self.img_width
        mask_height = self.img_height
        masks = []
        for mask in mask_output:
            img = Image.new("L", [self.img_width, self.img_height], 0)
            ImageDraw.Draw(img).polygon(mask.ravel().tolist(), outline=1, fill=1)
            masks.append(np.array(img))
        masks = np.array(masks)
        #print(masks.shape)
        masks = masks.reshape((-1, mask_height, mask_width))
        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            #print(np.count_nonzero(scale_crop_mask))
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
        return mask_maps

    def extract_boxes(self, boxes):
        with torch.inference_mode():
            boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, self.ids)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, self.ids, mask_maps=self.mask_maps)

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes