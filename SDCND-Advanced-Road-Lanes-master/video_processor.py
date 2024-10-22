import argparse
import os
import cv2
import numpy as np
import time

from img_processor import ImageProcessor
from lane_detector import LaneDetector

from moviepy.editor import VideoFileClip

class VideoProcessor:
    """
    Class used to process a video file and that produces an annotated version with the detected lanes.
    """

    def __init__(self, calibration_data_file, smooth_frames = 5, debug = False, failed_frames_dir = 'failed_frames'):
        self.img_processor = ImageProcessor(calibration_data_file)
        self.lane_tracker = LaneDetector(smooth_frames = smooth_frames)
        self.frame_count = 0
        self.fail_count = 0
        self.debug = debug
        self.failed_frames_dir = failed_frames_dir
        self.last_frame = None
        self.processed_frames = None

    def process_frame(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        undistorted_img, thresholded_img, warped_img = self.img_processor.process_image(img)
        
        _, polyfit, curvature, deviation, fail_code = self.lane_tracker.detect_lanes(warped_img)
        
        fill_color = (0, 255, 0) if fail_code == 0 else (0, 255, 255)

        lane_img = self.lane_tracker.draw_lanes(undistorted_img, polyfit, fill_color = fill_color)
        lane_img = self.img_processor.unwarp_image(lane_img)

        out_image = cv2.addWeighted(undistorted_img, 1.0, lane_img, 1.0, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 0)

        curvature_text = 'Left Curvature: {:.1f}, Right Curvature: {:.1f}'.format(curvature[0], curvature[1])
        offset_text = 'Center Offset: {:.2f} m'.format(deviation)

        cv2.putText(out_image, curvature_text, (30, 60), font, 1, font_color, 2)
        cv2.putText(out_image, offset_text, (30, 90), font, 1, font_color, 2)

        if fail_code > 0:
            self.fail_count += 1
            failed_text = 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code])
            cv2.putText(out_image, failed_text, (30, 120), font, 1, (0, 0, 255), 2)
            if self.debug:
                print(failed_text)
                cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '.png'), img)
                cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '_lanes.png'), out_image)
                if self.last_frame is not None: # Saves also the previous frame for comparison
                    cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '_prev.png'), self.last_frame)

        self.frame_count += 1
        self.last_frame = img

        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

        return out_image

    def process_frame_split(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        undistorted_img, thresholded_img, processed_img = self.img_processor.process_image(img)

        lanes_centroids, polyfit, curvature, deviation, fail_code = self.lane_tracker.detect_lanes(processed_img)
        
        fill_color = (0, 255, 0) if fail_code == 0 else (0, 255, 255)

        lane_img = self.lane_tracker.draw_lanes(undistorted_img, polyfit, fill_color = fill_color)
        lane_img = self.img_processor.unwarp_image(lane_img)

        out_image = cv2.addWeighted(undistorted_img, 1.0, lane_img, 1.0, 0)

        color_img = self.img_processor.color_thresh(undistorted_img) * 255
        gradient_img = self.img_processor.gradient_thresh(undistorted_img) * 255
      
        processed_src = np.array(cv2.merge((thresholded_img, thresholded_img, thresholded_img)), np.uint8)
        src, _ = self.img_processor._warp_coordinates(img)

        src = np.array(src, np.int32)
        
        cv2.polylines(processed_src, [src], True, (0,0,255), 2)

        window_img = np.copy(processed_img)

        window_img = self.lane_tracker.draw_windows(window_img, lanes_centroids, polyfit, blend = True)

        lanes_img = np.copy(out_image)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (0, 255, 0)

        curvature_text = 'Left Curvature: {:.1f}, Right Curvature: {:.1f}'.format(curvature[0], curvature[1])
        offset_text = 'Center Offset: {:.2f} m'.format(deviation)

        cv2.putText(out_image, curvature_text, (30, 60), font, 1, font_color, 2)
        cv2.putText(out_image, offset_text, (30, 90), font, 1, font_color, 2)

        if fail_code > 0:
            self.fail_count += 1
            failed_text = 'Detection Failed: {}'.format(LaneDetector.FAIL_CODES[fail_code])
            cv2.putText(out_image, failed_text, (30, 120), font, 1, (0, 0, 255), 2)
            if self.debug:
                print(failed_text)
                cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '.png'), img)
                cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '_lanes.png'), out_image)
                if self.last_frame is not None: # Saves also the previous frame for comparison
                    cv2.imwrite(os.path.join(self.failed_frames_dir, 'frame' + str(self.frame_count) + '_failed_' + str(fail_code) + '_prev.png'), self.last_frame)

        self.frame_count += 1

        undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        gradient_img = cv2.cvtColor(gradient_img, cv2.COLOR_GRAY2RGB)
        thresholded_img= cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2RGB)
        processed_src= cv2.cvtColor(processed_src, cv2.COLOR_BGR2RGB)
        processed_img= cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        window_img= cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB) 
        lanes_img= cv2.cvtColor(lanes_img, cv2.COLOR_BGR2RGB)
        out_image= cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

        return (undistorted_img, color_img, gradient_img, thresholded_img, processed_src, processed_img, window_img, lanes_img, out_image)

    def process_video(self, file_path, output, t_start = None, t_end = None):

        input_clip = VideoFileClip(file_path)
        
        if t_start is not None:
            input_clip = input_clip.subclip(t_start = t_start, t_end = t_end)
        
        output_clip = input_clip.fl_image(self.process_frame)
        output_clip.write_videofile(output, audio = False)

    def process_frame_stage(self, img, idx):
        
        if self.processed_frames is None:
            self.processed_frames = []

        if idx == 0:
            result = self.process_frame_split(img)
            self.processed_frames.append(result)
        else:
            self.frame_count += 1

        return self.processed_frames[self.frame_count - 1][idx]

    def process_video_split(self, file_path, output, t_start = None, t_end = None):

        input_clip = VideoFileClip(file_path)
        
        if t_start is not None:
            input_clip = input_clip.subclip(t_start = t_start, t_end = t_end)

        out_file_prefix = os.path.split(output)[1][:-4]
        
        idx = 0
        output_clip = input_clip.fl_image(lambda img: self.process_frame_stage(img, idx))
        output_clip.write_videofile(out_file_prefix + '_undistoreted.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_color.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_gradient.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_thresholded.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_processed_src.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_processed_dst.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_window.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_lanes.mp4', audio = False)
        idx += 1
        self.frame_count = 0
        output_clip.write_videofile(out_file_prefix + '_final.mp4', audio = False)

        print('Number of failed detection: {}'.format(self.fail_count))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video processor')

    parser.add_argument(
        'file_path',
        type=str,
        help='Video file path'
    )

    parser.add_argument(
        '--calibration_data_file',
        type=str,
        default=os.path.join('camera_cal', 'calibration.p'),
        help='Pickle file containing calibration data'
    )

    parser.add_argument(
        '--failed_frames_dir',
        type=str,
        default='failed_frames',
        help='Directory to save the failed frames when debug is enabled'
    )

    parser.add_argument(
        '--smooth',
        type=int,
        default=5,
        help='Number of frames to smooth'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=None,
        help='Time start'
    )

    parser.add_argument(
        '--end',
        type=float,
        default=None,
        help='Time start'
    )

    parser.add_argument(
        '--split',
        type=bool,
        default=False,
        help='Produce videos for each stage of the pipeline'
    )

    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='Debugging information'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.failed_frames_dir):
        os.makedirs(args.failed_frames_dir)

    date_time_str = time.strftime('%Y%m%d-%H%M%S')

    output = os.path.split(args.file_path)[1][:-4] + '_processed_' + date_time_str + '.mp4'

    video_processor = VideoProcessor(args.calibration_data_file, smooth_frames = args.smooth, debug = args.debug)

    if args.split:
        video_processor.process_video_split(args.file_path, output, t_start = args.start, t_end = args.end)
    else:
        video_processor.process_video(args.file_path, output, t_start = args.start, t_end = args.end)
