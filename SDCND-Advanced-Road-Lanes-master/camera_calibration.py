import pickle
import cv2
import numpy as np
import glob
import os
import argparse

"""
Script used to calibrate the camera from the images of a chessboard stored in a given folder. Saves the result
into a pickle file.
"""

def calibrate_camera(
    file_path = os.path.join('camera_cal', 'calibration*.jpg'),
    board_shape = (9, 6),
    corners_folder = os.path.join('output_images', 'calibration')):

    img_files = glob.glob(file_path)

    if len(img_files) == 0:
        print('No calibration image found')
        return None

    print('Calibrating camera with chessboard images (Board shape: {})...'.format(board_shape))

    obj_points = [] # 3D points in real world space
    img_points = [] # 2D point in image plane

    # Common coordinates for the images
    _obj_points = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
    _obj_points[:,:2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2)

    if corners_folder is not None and not os.path.isdir(corners_folder):
        os.makedirs(corners_folder)

    img_shape = None
    
    for img_file in img_files:
        print('Processing {}...'.format(img_file))
        img = cv2.imread(img_file)

        # Grayscale conversion
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = img_gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img_gray, board_shape, None)
        
        if ret == True:
            img_points.append(corners)
            obj_points.append(_obj_points)

            # Draws the corners into the image and saves it out
            if corners_folder is not None:
                cv2.drawChessboardCorners(img, board_shape, corners, ret)
                out_file = os.path.join(corners_folder, os.path.split(img_file)[1][:-4] + '_corners.jpg')
                print('Saving corners image to {}'.format(out_file))
                cv2.imwrite(out_file, img)
        else:
            print('No corner found in image {}'.format(img_file))

    if img_shape is not None and len(img_points) > 0:
        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

        return {
            'mtx': mtx,
            'dist': dist
        }
    else:
        return None

def load_calibration_data(file_path = os.path.join('camera_cal', 'calibration.p')):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def dump_calibration_data(data, output = os.path.join('camera_cal', 'calibration.p')):
    with open(output, 'wb') as f:
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Camera calibration')

    parser.add_argument(
        '--filepath',
        type=str,
        default=os.path.join('camera_cal', 'calibration*.jpg'),
        help='Path pattern for the calibration images'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Output path for the pickle file containing calibration parameters'
    )

    parser.add_argument(
        '--corners_folder',
        type=str,
        default=os.path.join('output_images', 'calibration'),
        help='Folder for saving images with detected corners'
    )
    
    parser.add_argument(
        '--cols',
        type=int,
        default=9,
        help='Number of column corners in the chessboard'
    )

    parser.add_argument(
        '--rows',
        type=int,
        default=6,
        help='Number of row corners in the chessboard'
    )

    args = parser.parse_args()

    if args.corners_folder.lower() == 'none' or args.corners_folder.lower() == 'false':
        args.corners_folder = None

    calibration_data = calibrate_camera(file_path=args.filepath, board_shape=(args.cols, args.rows), corners_folder=args.corners_folder)

    if calibration_data is not None:
        print('Calibration data: {}'.format(calibration_data))
        dump_calibration_data(calibration_data, args.output)
    else:
        print('Could not calibrate the camera.')