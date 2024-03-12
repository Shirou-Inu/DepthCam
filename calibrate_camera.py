import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from datetime import datetime
import glob

# Control Variable
CALIBRATE = True # True - Take calibration pictures and compute calibration values
CAM_AVAILABLE = False # Skips taking pictures

# Calibration Path
cal_path = 'calibration_images/'
save_path = 'calibration/stereo_calibration.npz'

# Checkerboard size: Number of points in checkerboard row and column
# Square size: Size of each square in checkerboard in meters
def computeCalibration(left_images, right_images, checkerboard_size, square_size):
    # Arrays to store object points and image points from all images
    object_points = []  # 3D points in real world space
    left_image_points = []  # 2D points in image plane
    right_image_points = []

    # Prepare grid and object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Scale by the size of the squares

    # Arrays to store calibration parameters
    camera_matrix_left = None
    dist_coeffs_left = None
    camera_matrix_right = None
    dist_coeffs_right = None

    for left_img, right_img in zip(left_images, right_images):
        left = cv.imread(left_img)
        right = cv.imread(right_img)

        # Convert images to grayscale
        gray_left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_left, corners_left = cv.findChessboardCorners(gray_left, checkerboard_size, None)
        ret_right, corners_right = cv.findChessboardCorners(gray_right, checkerboard_size, None)
        
        if ret_left and ret_right:
            # If corners are found, append object points and image points
            object_points.append(objp)
            left_image_points.append(corners_left)
            right_image_points.append(corners_right)

    # Perform stereo calibration
    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv.stereoCalibrate(
        object_points, left_image_points, right_image_points,
        None, None, None, None,
        gray_left.shape[::-1],
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        flags=cv.CALIB_FIX_INTRINSIC
    )

    # Save the calibration results
    np.savez(save_path, ret=ret,
            camera_matrix_left=camera_matrix_left, dist_coeffs_left=dist_coeffs_left,
            camera_matrix_right=camera_matrix_right, dist_coeffs_right=dist_coeffs_right,
            R=R, T=T, E=E, F=F)
    
    return ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F

try:
    if CALIBRATE and CAM_AVAILABLE:
        # Setup Pipeline
        pipe = rs.pipeline()

        # Setup Config
        cfg = rs.config()
        cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

        # Start Pipeline
        pfp = pipe.start(cfg)

        # Get Camera
        device = pfp.get_device()
        depth_sensor = device.query_sensors()[0]

        # Setup Camera
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        depth_sensor.set_option(rs.option.exposure, 33000)
except:
    print('No camera found')
    CAM_AVAILABLE = False

if CALIBRATE:
    while True and CAM_AVAILABLE:
        # Obtain Frames
        frames = pipe.wait_for_frames()
        ir_left = frames.get_infrared_frame(1)
        ir_right = frames.get_infrared_frame(2)

        # Covert to readable values
        cv_ir_left = np.asanyarray(ir_left.get_data())
        cv_ir_right = np.asanyarray(ir_right.get_data())

        # Normalize
        norm_ir_left = np.uint8(cv.normalize(cv_ir_left, cv_ir_left, alpha=255, beta=0, norm_type=cv.NORM_MINMAX))
        norm_ir_right = np.uint8(cv.normalize(cv_ir_right, cv_ir_right, alpha=255, beta=0, norm_type=cv.NORM_MINMAX))

        # Display Preview Frames
        preview_left = cv.resize(norm_ir_left, (640, 480))
        preview_right = cv.resize(norm_ir_right, (640, 480))
        preview = np.hstack((preview_left, preview_right))
        cv.imshow('Press s to save', preview)

        # Save Frame
        if cv.waitKey(2) == ord('s'):
            print('Saving')
            time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            cv.imwrite('{}{}_left.jpg'.format(cal_path, time), cv_ir_left)
            cv.imwrite('{}{}_right.jpg'.format(cal_path, time), cv_ir_right)

        # Start Calibration
        if cv.waitKey(1) == ord('q'):
            break

    if CAM_AVAILABLE:
        # Stop Pipeline
        pipe.stop()

    # Close CV Windows
    cv.destroyAllWindows()

    # Load Calibration Images
    print('Loading Calibration Images')
    left_images = glob.glob('{}*_left.jpg'.format(cal_path))  # Pattern to load left images
    right_images = glob.glob('{}*_right.jpg'.format(cal_path))  # Pattern to load right images
    
    # Stereo Calibration
    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = computeCalibration(left_images, right_images, (7, 5), 0.03)

    # Print the calibration results
    print("Stereo Calibration Complete")
    print("\nLeft Camera Matrix:\n", camera_matrix_left)
    print("\nLeft Distortion Coefficients:\n", dist_coeffs_left)
    print("\nRight Camera Matrix:\n", camera_matrix_right)
    print("\nRight Distortion Coefficients:\n", dist_coeffs_right)
    print("\nRotation Matrix:\n", R)
    print("\nTranslation Vector:\n", T)
    print("\nEssential Matrix:\n", E)
    print("\nFundamental Matrix:\n", F)

else:
    print('Calibration Values')
    with np.load(save_path) as calibration:
        camera_matrix_left = calibration['camera_matrix_left']
        dist_coeffs_left = calibration['dist_coeffs_left']
        camera_matrix_right = calibration['camera_matrix_right']
        dist_coeffs_right = calibration['dist_coeffs_right']
        R = calibration['R']
        T = calibration['T']
        E = calibration['E']
        F = calibration['F']

    print("\nLeft Camera Matrix:\n", camera_matrix_left)
    print("\nLeft Distortion Coefficients:\n", dist_coeffs_left)
    print("\nRight Camera Matrix:\n", camera_matrix_right)
    print("\nRight Distortion Coefficients:\n", dist_coeffs_right)
    print("\nRotation Matrix:\n", R)
    print("\nTranslation Vector:\n", T)
    print("\nEssential Matrix:\n", E)
    print("\nFundamental Matrix:\n", F)
