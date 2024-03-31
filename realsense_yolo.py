from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import torch

mouse_pt = (0, 0)

def show_distance(event, x, y, args, params):
    global mouse_pt
    mouse_pt = (x, y)

cv.namedWindow("Color")
cv.setMouseCallback("Color", show_distance)

# Global Variables
image_size = (1280, 720)

# Initialize RealSense pipeline
pipe = rs.pipeline()

# Setup Config
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, image_size[0], image_size[1], rs.format.y8, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start Pipeline
pfp = pipe.start(cfg)

# Get Camera
device = pfp.get_device()
depth_sensor = device.query_sensors()[0]

# Alignment
align = rs.align(rs.stream.depth)

# YOLO Model
model = YOLO('yolov8n-seg.pt')
target = 'person'

while True:
    # Obtain Frames
    frames = pipe.wait_for_frames()
    ir_left = frames.get_infrared_frame(1)
    color = frames.get_color_frame()

    # Aligned frames
    aligned_frames = align.process(frames)
    aligned_color = aligned_frames.get_color_frame()

    # Covert to readable values
    cv_ir_left = np.asanyarray(ir_left.get_data())
    cv_color = np.asanyarray(aligned_color.get_data())

    # Apply Color Map to IR
    cv_ir_left = cv.applyColorMap(cv_ir_left, colormap=cv.COLORMAP_CIVIDIS)

    # Detect
    results = model(cv_color, verbose=False)

    # Work with results
    img_masks = []
    for r in results:
        annotator = Annotator(cv_color.copy())
        
        boxes = r.boxes
        masks = r.masks

        if boxes is not None and masks is not None:
            for box, mask in zip(boxes, masks):
                b = box.xyxy[0]
                c = model.names[int(box.cls)]

                if c == target:

                    # Obtain masks
                    mask_notator = Annotator(np.zeros(cv_color.shape, dtype=np.uint8))
                    l_img = LetterBox(mask.shape[1:])(image=mask_notator.result())
                    im_gpu = torch.as_tensor(l_img, dtype=torch.float16, device=mask.data.device).permute(2, 0, 1).flip(0).contiguous() / 255
                    mask_notator.masks(mask.data, colors=[[255, 255, 255]], im_gpu=im_gpu, alpha=1)
                    
                    img_mask = np.where(mask_notator.result()[:, :, 0] > 0, True, False)

                    img_masks.append(img_mask)
                    
                    # Annotate boxes
                    annotator.box_label(b, c, color=colors(int(box.cls), True))

    anno_color = annotator.result()

    # Draw mask frame
    seg_img = np.zeros(cv_color.shape, dtype=np.uint8)
    for img_mask in img_masks:
        seg_img[img_mask] = cv_color[img_mask]

    # Annotate Mouse Position
    cv.circle(cv_color, mouse_pt, 4, (0, 0, 255))
    cv.circle(cv_ir_left, mouse_pt, 4, (0, 0, 255))

    # Display Frames
    cv.imshow('IR', cv_ir_left)
    cv.imshow('Color', anno_color)
    cv.imshow('Segmentation', seg_img)

    # CV Wait Key
    key = cv.waitKey(1)

    # Quit
    if key == ord('q'):
        print('Exiting')
        break

pipe.stop()
cv.destroyAllWindows()