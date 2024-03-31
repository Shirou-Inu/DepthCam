from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import torch
from depth_map import depthMap
import matplotlib.pyplot as plt

# Global Variables
image_size = (1280, 720)
cal_path = 'calibration/realsense_calibration.npz'
model_name = 'yolov8n-seg.pt'
model_target = 'person'

# Functions
def drawRadar(detected_targets, max_width, max_depth):
    scale_factor = 10

    width = int((max_width * 2) / scale_factor)
    mid_x = int(width / 2)
    height = int(max_depth / scale_factor)
    radar = np.zeros((height, width, 3), dtype=np.uint8)

    cv.circle(radar, (int(width/2), height), 1, [0, 255, 0], 5)
    
    unknown_count = 0
    for target in detected_targets:
        if target:
            x = int(mid_x + (target[0]/ scale_factor))
            z = int(height - (target[2] / scale_factor))
            
            cv.circle(radar, (x, z), 1, [0, 0, 255], 5)
            cv.putText(radar, '{:.2f}'.format(target[0]), (x + 10, z), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv.putText(radar, '{:.2f}'.format(target[1]), (x + 10, z + 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv.putText(radar, '{:.2f}'.format(target[2]), (x + 10, z + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        else:
            unknown_count += 1

    return radar

# Initialize RealSense pipeline
pipe = rs.pipeline()

# Setup Config
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, image_size[0], image_size[1], rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, image_size[0], image_size[1], rs.format.y8, 30)
cfg.enable_stream(rs.stream.color, image_size[0], image_size[1], rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, image_size[0], image_size[1], rs.format.z16, 30)

# Start Pipeline
pfp = pipe.start(cfg)

# Get Camera
device = pfp.get_device()
depth_sensor = device.query_sensors()[0]

# Setup Camera
depth_sensor.set_option(rs.option.emitter_enabled, 1)
depth_sensor.set_option(rs.option.exposure, 33000)

# Setup Alignment
align = rs.align(rs.stream.depth) # Only for realsense due to the stereo camera is grayscale

# Load Calibration File
calibration = np.load(cal_path)

# Create Depth Map Object
dm = depthMap(calibration, image_size)

# Check for CUDA device and set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Create Model Object
model = YOLO(model_name).to(device)

while True:
    # Obtain Frames
    frames = pipe.wait_for_frames()
    ir_left = frames.get_infrared_frame(1)
    ir_right = frames.get_infrared_frame(2)
    aligned_frames = align.process(frames)
    color = aligned_frames.get_color_frame()

    # Convert to readable values
    cv_ir_left = np.asanyarray(ir_left.get_data())
    cv_ir_right = np.asanyarray(ir_right.get_data())
    cv_color = np.asanyarray(color.get_data())

    # Compute Depth
    depth_map = dm.computeDepth(cv_ir_left, cv_ir_right)

    # Find all person in frame
    results = model(cv_color, verbose=False)

    # Work with results
    result = results[0]
    boxes = result.boxes
    masks = result.masks

    target_details = []

    annotator = Annotator(cv_color.copy())
    if boxes is not None and masks is not None:
        for box, mask in zip(result.boxes, result.masks):
            b = box.xyxy[0]
            c = model.names[int(box.cls)]

            if c == model_target:
                # Obtain target mask
                mask_notator = Annotator(np.zeros(cv_color.shape, dtype=np.uint8))
                l_img = LetterBox(mask.shape[1:])(image=mask_notator.result())
                im_gpu = torch.as_tensor(l_img, dtype=torch.float16, device=mask.data.device).permute(2, 0, 1).flip(0).contiguous() / 255
                mask_notator.masks(mask.data, colors=[[255, 255, 255]], im_gpu=im_gpu, alpha=1)

                img_mask = np.where(mask_notator.result()[:, :, 0] > 0, True, False)

                # Draw segmentation
                l_img = LetterBox(mask.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(l_img, dtype=torch.float16, device=mask.data.device).permute(2, 0, 1).flip(0).contiguous() / 255
                annotator.masks(mask.data, colors=[colors(int(box.cls), True)], im_gpu=im_gpu, alpha=0.5)

                # Determine appropriate x, y, and z values for detected target
                target_dm = depth_map[np.where(img_mask)]
                target_dm = target_dm[target_dm[:, 2] < 10000]
                if target_dm.shape[0] > 0:
                    target_xyz = np.median(target_dm, axis=0)

                    target_details.append((target_xyz[0], target_xyz[1], target_xyz[2]))

                    label = '{}: {:.2f}mm'.format(c, target_xyz[2])
                else:
                    target_details.append(None)
                    label = '{}: Unknown Depth'.format(c)

                # Draw box
                annotator.box_label(b, label, color=colors(int(box.cls), True))

    color_results = annotator.result()
    radar_img = drawRadar(target_details, 3000, 6000)

    # Display Frame
    cv.imshow('Cam View', color_results)
    cv.imshow('Radar View', radar_img)

    # Show depth map
    # plt.imshow(depth_map[:, :, 2], cmap='plasma')
    # plt.colorbar()
    # plt.show()

    # CV Wait Key
    key = cv.waitKey(1)

    # Quit
    if key == ord('q'):
        print('Exiting')
        break

pipe.stop()
cv.destroyAllWindows()