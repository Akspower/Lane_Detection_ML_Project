import pickle
import cv2
import numpy as np
import camera_calibration as cc

class ImageProcessor:
    """
    Class used to process an image for the LaneDetector. Applies both color and gradient thresholding and produces a set of
    images (undistored, thresholded and warped) that can be used for debugging.
    """

    def __init__(self, calibration_data_file):

        # Camera calibration data
        calibration_data = cc.load_calibration_data(file_path = calibration_data_file)
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

        # Gradient and color thresholding parameters
        self.sobel_kernel = 5
        self.grad_x_thresh = (15, 255) # Sobel x threshold
        self.grad_y_thresh = (25, 255) # Sobel y threshold
        self.grad_mag_thresh = (40, 255) # Sobel mag threshold
        self.grad_dir_thresh = (0.7, 1.3) # Sobel direction range
        self.grad_v_thresh = (180, 255) # HSV, V channel threshold to filter gradient

        self.r_thresh = (195, 255) # RGB, Red channel threshold
        self.s_thresh = (100, 255) # HSL, S channel threshold
        self.l_thresh = (195, 255) # HSL, L channel threshold
        self.b_thresh = (150, 255) # LAB, B channel threshold
        self.v_thresh = (140, 255) # HSV, V channel threshold

        # Perspective transformation parameters
        # slope = (y2 - y1) / (x2 - x1)
        # intercept = y1 - slope * x1
        # top left, top right = (570, 470), (722, 470)
        # bottom left, bottom right = (220, 720), (1110, 720)
        self.persp_src_left_line = (-0.7142857143, 877.142857146) # Slope and intercept for left line
        self.persp_src_right_line = (0.6443298969, 4.793814441) # Slope and intercept for right line
        self.persp_src_top_pct = 0.645 # Percentage from the top
        self.persp_src_bottom_pct = 0.02 # Percentage from bottom
        self.persp_dst_x_pct = 0.22 # Destination offset percent
        self.persp_src = None
        self.persp_dst = None

    def _warp_coordinates(self, img):

        if self.persp_src is None or self.persp_dst is None:

            cols = img.shape[1]
            rows = img.shape[0]

            src_top_offset = rows * self.persp_src_top_pct
            src_bottom_offset = rows * self.persp_src_bottom_pct
            left_slope, left_intercept = self.persp_src_left_line
            right_slope, right_intercept = self.persp_src_right_line

            top_left = [(src_top_offset - left_intercept) / left_slope, src_top_offset]
            top_right = [(src_top_offset - right_intercept) / right_slope, src_top_offset]
            bottom_left = [(rows - src_bottom_offset - left_intercept) / left_slope, rows - src_bottom_offset]
            bottom_right = [(rows - src_bottom_offset - right_intercept) / right_slope, rows - src_bottom_offset]

            #Top left, Top right, Bottom right, Bottom left        
            src = np.float32([top_left, top_right, bottom_right, bottom_left])

            dst_x_offset = cols * self.persp_dst_x_pct
    
            top_left = [dst_x_offset, 0]
            top_right = [cols - dst_x_offset, 0]
            bottom_left = [dst_x_offset, rows]
            bottom_right = [cols - dst_x_offset, rows]
            
            dst = np.float32([top_left, top_right, bottom_right, bottom_left])
                            
            self.persp_src = src
            self.persp_dst = dst
        
        return self.persp_src, self.persp_dst

    def _sobel(self, img, orient = 'x', sobel_kernel = 3):
        # Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        else:
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
        return sobel

    def _apply_thresh(self, img, thresh = [0, 255]):
        result = np.zeros_like(img)
        result[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return result

    def unwarp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)

        warp_m = cv2.getPerspectiveTransform(dst, src)
        unwarped = cv2.warpPerspective(img, warp_m, img_shape)

        return unwarped

    def warp_image(self, img):

        img_shape = img.shape[1::-1]

        src, dst = self._warp_coordinates(img)

        warp_m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, warp_m, img_shape)

        return warped

    def undistort_image(self, img):

        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def sobel_abs_thresh(self, sobel, thresh=[0,255]):
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = self._apply_thresh(scaled_sobel, thresh)

        return binary_output

    def sobel_mag_thresh(self, sobel_x, sobel_y, thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobel_x**2 + sobel_y**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 

        binary_output = self._apply_thresh(gradmag, thresh)
        
        return binary_output

    def sobel_dir_thresh(self, sobel_x, sobel_y, thresh=(0, np.pi/2)):
        # Take the absolute value of the x and y gradients
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)

        # Calculate the direction of the gradient 
        abs_grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)

        binary_output = self._apply_thresh(abs_grad_dir, thresh)
       
        return binary_output

    def gradient_thresh(self, img):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_ch = hsv_img[:,:,2]
        v_binary = self._apply_thresh(v_ch, self.grad_v_thresh)

        sobel_x = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'x')
        sobel_y = self._sobel(gray_img, sobel_kernel = self.sobel_kernel, orient = 'y')

        sobel_x_binary = self.sobel_abs_thresh(sobel_x, thresh = self.grad_x_thresh)
        sobel_y_binary = self.sobel_abs_thresh(sobel_y, thresh = self.grad_y_thresh)

        sobel_mag_binary = self.sobel_mag_thresh(sobel_x, sobel_y, thresh = self.grad_mag_thresh)
        sobel_dir_binary = self.sobel_dir_thresh(sobel_x, sobel_y, thresh = self.grad_dir_thresh)

        sobel_binary = np.zeros_like(sobel_x_binary)

        sobel_binary[(((sobel_x_binary == 1) & (sobel_y_binary == 1)) | (sobel_dir_binary == 1)) & (sobel_mag_binary == 1) & (v_binary == 1)] = 1
        
        return sobel_binary

    def color_thresh(self, img):

        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        r_ch = img[:,:,2]
        r_binary = self._apply_thresh(r_ch, self.r_thresh)

        l_ch = hls_img[:,:,1]
        l_binary = self._apply_thresh(l_ch, self.l_thresh)

        s_ch = hls_img[:,:,2]
        s_binary = self._apply_thresh(s_ch, self.s_thresh)

        b_ch = lab_img[:,:,2]
        b_binary = self._apply_thresh(b_ch, self.b_thresh)

        v_ch = hsv_img[:,:,2]
        v_binary = self._apply_thresh(v_ch, self.v_thresh)

        result = np.zeros_like(s_binary)

        # B and V for yellow, R and L for white, S and V for both
        result[((b_binary == 1) & (v_binary == 1)) | ((r_binary == 1) & (l_binary == 1)) | ((s_binary == 1) & (v_binary == 1))] = 1

        return result
        
    def threshold_image(self, img):

        gradient_binary = self.gradient_thresh(img)
        color_binary = self.color_thresh(img)

        result = np.zeros_like(gradient_binary)
        result[(gradient_binary == 1) | (color_binary) == 1] = 255

        return result

    def process_image(self, img):
        """
        Process the given image appling undistorsion from the camera calibration data, thresholds the result and then
        warps the image for an bird-eye view of the road.
        """

        undistorted_img = self.undistort_image(img)

        thresholded_img = self.threshold_image(undistorted_img)

        warped_img = self.warp_image(thresholded_img)

        return undistorted_img, thresholded_img, warped_img