from PerspectiveTransformation import *
from Canny import *
from Hough import *
from LaneLines import *


class LaneDetector:
    def init(self):
        self.transform = PerspectiveTransformation()
        self.canny = CannyEdge()
        self.hough = Hough()
        self.LaneLines = LaneLines()

    def forward(self, img):
        canny_img = self.canny.forward(img)
        transform_img = self.transform.forward(canny_img)
        hough_img = self.hough.forward(transform_img)
        _, pos, detected = self.LaneLines.forward(img)

        return pos, detected
