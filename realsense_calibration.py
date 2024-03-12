import pyrealsense2 as rs
import numpy as np
import cv2

save_path = 'calibration/realsense_calibration.npz'

# Initialize RealSense pipeline
pipeline = rs.pipeline()

# Setup Config
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# Start Pipeline
pipeline.start(cfg)

profile = pipeline.get_active_profile()

left_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 1))
left_intrinsics = left_profile.get_intrinsics()

right_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
right_intrinsics = right_profile.get_intrinsics()

depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

l2r_extrinsics = left_profile.get_extrinsics_to(right_profile)
l2r_rotation = np.array(l2r_extrinsics.rotation).reshape(3, 3)
l2r_translation = np.array(l2r_extrinsics.translation)
d2c_extrinsics = depth_profile.get_extrinsics_to(color_profile)
d2c_rotation = np.array(d2c_extrinsics.rotation).reshape(3, 3)
d2c_translation = np.array(d2c_extrinsics.translation)

# print('Left Intrinsics: ', left_intrinsics)
# print('Right Intrinsics: ', right_intrinsics)
# print('Depth Intrinsics: ', depth_intrinsics)
# print('Color Intrinsics: ', color_intrinsics)

# print('Left to Right Extrinsics')
# print('Rotation: ', l2r_rotation)
# print('Translation: ', l2r_translation)
# print('Depth to Color Extrinsics')
# print('Rotation: ', d2c_rotation)
# print('Translation: ', d2c_translation)

pipeline.stop()

# Write Intrinsics and Extrinsics
camera_matrix_left = np.array([[left_intrinsics.fx, 0, left_intrinsics.ppx],
                               [0, left_intrinsics.fy, left_intrinsics.ppy],
                               [0, 0, 1]])
dist_coeffs_left = np.array([left_intrinsics.coeffs])
camera_matrix_right = np.array([[right_intrinsics.fx, 0, right_intrinsics.ppx],
                               [0, right_intrinsics.fy, right_intrinsics.ppy],
                               [0, 0, 1]])
dist_coeffs_right = np.array([right_intrinsics.coeffs])
R = np.array(l2r_rotation)
T = np.array([l2r_translation]).T

print("\nLeft Camera Matrix:\n", camera_matrix_left)
print("\nLeft Distortion Coefficients:\n", dist_coeffs_left)
print("\nRight Camera Matrix:\n", camera_matrix_right)
print("\nRight Distortion Coefficients:\n", dist_coeffs_right)
print("\nRotation Matrix:\n", R)
print("\nTranslation Vector:\n", T)

np.savez(save_path,
        camera_matrix_left=camera_matrix_left, dist_coeffs_left=dist_coeffs_left,
        camera_matrix_right=camera_matrix_right, dist_coeffs_right=dist_coeffs_right,
        R=R, T=T)
        