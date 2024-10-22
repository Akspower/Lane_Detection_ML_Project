import os
import cv2
import argparse

import numpy as np

from glob import glob
from tqdm import tqdm
from img_processor import ImageProcessor
from lane_detector import LaneDetector

"""
Script used to process a folder with images to test the pipeline
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image processor')

    parser.add_argument(
        '--dir',
        type=str,
        default='test_images',
        help='File pattern for images to process'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output_images',
        help='Folder where to save the processed images'
    )

    parser.add_argument(
        '--calibration_data_file',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Pickle file containing calibration data'
    )

    parser.add_argument(
        '--w_width',
        type=int,
        default=30,
        help='Sliding window width'
    )

    parser.add_argument(
        '--w_height',
        type=int,
        default=80,
        help='Sliding window height'
    )

    parser.add_argument(
        '--w_margin',
        type=int,
        default=35,
        help='Sliding window margin from center'
    )

    args = parser.parse_args()

    img_files = []
    
    for ext in ('*.png', '*.jpg'):
        img_files.extend(glob(os.path.join(args.dir, ext)))

    img_processor = ImageProcessor(args.calibration_data_file)
    lane_tracker = LaneDetector(window_width=args.w_width, window_height=args.w_height, margin=args.w_margin, smooth_frames = 0)

    for img_file in tqdm(img_files, unit=' images', desc ='Image processing'):

        print(img_file)
        input_img = cv2.imread(img_file)

        undistorted_img, thresholded_img, processed_img = img_processor.process_image(input_img)
        
        out_file_prefix = os.path.join(args.output, os.path.split(img_file)[1][:-4])
        cv2.imwrite(out_file_prefix + '_processed.jpg', processed_img)

        cv2.imwrite(out_file_prefix + '_undistorted.jpg', undistorted_img)
        cv2.imwrite(out_file_prefix + '_thresholded.jpg', thresholded_img)
        cv2.imwrite(out_file_prefix + '_color.jpg', img_processor.color_thresh(undistorted_img) * 255)
        cv2.imwrite(out_file_prefix + '_gradient.jpg', img_processor.gradient_thresh(undistorted_img) * 255)
        
        warped_img = img_processor.warp_image(undistorted_img)
      
        processed_src = np.copy(undistorted_img)
        processed_dst = np.copy(warped_img)

        src, dst = img_processor._warp_coordinates(input_img)

        src = np.array(src, np.int32)
        dst = np.array(dst, np.int32)

        cv2.polylines(processed_src, [src], True, (0,0,255), 2)
        cv2.polylines(processed_dst, [dst], True, (0,0,255), 2)
        
        cv2.imwrite(out_file_prefix + '_persp_src.jpg', processed_src)
        cv2.imwrite(out_file_prefix + '_persp_dst.jpg', processed_dst)

        lanes_centroids, polyfit, curvature, deviation, fail_code = lane_tracker.detect_lanes(processed_img)
        processed_img = lane_tracker.draw_windows(processed_img, lanes_centroids, polyfit, blend = True)
        
        cv2.imwrite(out_file_prefix + '_windows.jpg', processed_img)

        lanes_img = lane_tracker.draw_lanes(input_img, polyfit)
        lanes_img = img_processor.unwarp_image(lanes_img)

        lanes_img = cv2.addWeighted(undistorted_img, 1.0, lanes_img, 1.0, 0)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 0)
        cv2.putText(lanes_img, 'Left Curvature: {:.1f}, Right Curvature: {:.1f}'.format(curvature[0], curvature[1]), (30, 60), font, 1, font_color, 2)
        cv2.putText(lanes_img, 'Center Offset: {:.2f} m'.format(deviation), (30, 90), font, 1, font_color, 2)

        if fail_code > 0:
            failed_text = 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code])
            print(failed_text)
            cv2.putText(lanes_img, failed_text, (30, 120), font, 1, (0, 0, 255), 2)

        cv2.imwrite(out_file_prefix + '_lanes.jpg', lanes_img)