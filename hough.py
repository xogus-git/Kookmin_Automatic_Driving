import cv2
from Canny import *
from PerspectiveTransformation import *
class Hough:
    def __init__(self):
        self.canny = CannyEdge()
    
    def forward(self, img):
        
        lines = cv2.HoughLinesP(img, 0.5, np.pi/180, 100)
        out_img = np.zeros_like(img)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out_img, (x1, y1), (x2, y2), 150, 4)
        
        return out_img



if __name__ == '__main__':
    img = cv2.imread('./img/60.jpg')
    hough = Hough()
    canny = CannyEdge()
    roi = ROI()
    transform = PerspectiveTransformation()

    img = canny.forward(img)
    #img = transform.forward(img)
    #img = roi.forward(img)
    img = hough.forward(img)

    cv2.imshow('hough transform',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
