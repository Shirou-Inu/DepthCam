import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from datetime import datetime

# Debugging
mouse_pt = (0, 0)

def show_distance(event, x, y, args, params):
    global mouse_pt
    mouse_pt = (x, y)

cv.namedWindow("Color")
cv.setMouseCallback("Color", show_distance)

# Test Path
cal_path = 'test_images/'

# Setup Pipeline
pipe = rs.pipeline()

# Setup Config
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start Pipeline
pfp = pipe.start(cfg)

# Get Camera
device = pfp.get_device()
depth_sensor = device.query_sensors()[0]

# Setup Camera
IR = 1
depth_sensor.set_option(rs.option.emitter_enabled, IR)
depth_sensor.set_option(rs.option.exposure, 33000)

while True:
    # Obtain Frames
    frames = pipe.wait_for_frames()
    ir_left = frames.get_infrared_frame(1)
    ir_right = frames.get_infrared_frame(2)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()

    # Covert to readable values
    cv_ir_left = np.asanyarray(ir_left.get_data())
    cv_ir_right = np.asanyarray(ir_right.get_data())
    cv_color = np.asanyarray(color.get_data())
    cv_depth = np.asanyarray(depth.get_data())

    # Resize Frames
    resized_left = cv.resize(cv_ir_left, (640, 480))
    resized_right = cv.resize(cv_ir_right, (640, 480))
    resized_color = cv.resize(cv_color, (640, 480))
    resized_ref_depth = cv.resize(cv_depth, (640, 480))
    ir_frames = np.hstack((resized_left, resized_right))

    # Normalize
    norm_ref_depth = cv.applyColorMap(np.uint8(cv.normalize(resized_ref_depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)), colormap=cv.COLORMAP_PLASMA)

    # Show distance at mouse point
    ref_dist = resized_ref_depth[mouse_pt[1], mouse_pt[0]]
    cv.circle(resized_color, mouse_pt, 4, (0, 0, 255))
    # cv.circle(norm_ref_depth, mouse_pt, 4, (0, 0, 255))
    cv.putText(resized_color, 'Ref: {}mm'.format(ref_dist), (mouse_pt[0], mouse_pt[1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    # Display Frames
    cv.imshow('IR', ir_frames)
    cv.imshow('Color', resized_color)
    cv.imshow('Depth Reference', norm_ref_depth)

    # Save key
    key = cv.waitKey(1)

    # Toggle IR
    if key == ord('i'):
        IR = (IR + 1) % 2
        print(IR)
        depth_sensor.set_option(rs.option.emitter_enabled, IR)

    # Save Frame
    if key == ord('s'):
        print('Saving')
        time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        cv.imwrite('{}{}_left.jpg'.format(cal_path, time), cv_ir_left)
        cv.imwrite('{}{}_right.jpg'.format(cal_path, time), cv_ir_right)
        cv.imwrite('{}{}_color.jpg'.format(cal_path, time), cv_color)
        np.save('{}{}_depth.npy'.format(cal_path, time), cv_depth)

    # Start Calibration
    if key == ord('q'):
        print('Exiting')
        break

# Stop Pipeline
pipe.stop()

# Close CV Windows
cv.destroyAllWindows()