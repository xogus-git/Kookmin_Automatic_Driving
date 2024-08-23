import cv2
import numpy as np
from constants import *

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Lanelines.

        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.width = None
        self.height = None
        self.clear_visibility = True
        self.dir = []

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = NWINDOWS
        # Width of the the windows +/- margin
        self.margin = MARGIN
        # Mininum number of pixels found to recenter window
        self.minpix = MINPIX

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = int(img.shape[0]//self.nwindows)
        self.width = img.shape[1]
        self.height = img.shape[0]

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))
        

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:(midpoint-50)])
        rightx_base = np.argmax(histogram[midpoint+50:]) + midpoint+50

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        left_find = True
        right_find = True
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)
    
            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = int(np.mean(good_left_x))
            
            if len(good_right_x) > self.minpix:
                rightx_current = int(np.mean(good_right_x))
            

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) < len(righty):
            leftx= [x - 800 for x in rightx]
            lefty = righty
        else:
            rightx = [x + 800 for x in leftx]
            righty = lefty

        

        if len(lefty) > 500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        
        
        
        
        

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))
        
        miny = 0
        maxy = self.height
        ploty = np.linspace(miny, maxy, img.shape[0])
        try:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except:
            print("not detected")
            return img, False, 0

        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        pos = self.measure_center()
        if (pos >= 0 and pos <= WIDTH):
            cv2.circle(out_img, (pos, HEIGHT), 2, (255, 0, 0))
        return out_img, True, pos

    def measure_center(self):
        xl = np.dot(self.left_fit, [HEIGHT**2, HEIGHT, 1])
        xr = np.dot(self.right_fit, [HEIGHT**2, HEIGHT, 1])
        return int((xl + xr) // 2)

if __name__ == '__main__':
    from PerspectiveTransformation import *
    from Tools.Thresholding import *
    from Canny import *
    from Hough import *


    transform = PerspectiveTransformation()
    canny = CannyEdge()
    hough = Hough()
    laneLines = LaneLines()
    os.makedirs("./LaneLine_img", exist_ok=True)
    file_lst = os.listdir("./img")
    for file in file_lst:
        img = cv2.imread("./img/{}".format(file))
        img = canny.forward(img)
        img = transform.forward(img)
        img = hough.forward(img)
        img, _, _ = laneLines.forward(img)
        cv2.imwrite("./LaneLine_img/{}".format(file), img)

    

    
