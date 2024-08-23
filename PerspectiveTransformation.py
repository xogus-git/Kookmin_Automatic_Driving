import cv2
import numpy as np
from constants import *

# global parameter

class PerspectiveTransformation:
    def __init__(self):
        self.pts1 = np.float32([TRANSFORM_TOP_LEFT, TRANSFORM_BOTTOM_LEFT, TRANSFORM_TOP_RIGHT, TRANSFORM_BOTTOM_RIGHT])
        self.pts2 = np.float32([[0, 0], [0, TRANSFORMHEIGHT], [TRANSFORMWIDTH, 0], [TRANSFORMWIDTH, TRANSFORMHEIGHT]])
        self.matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.matrix_inv = cv2.getPerspectiveTransform(self.pts2, self.pts1)

    def forward(self, img):
        transformed_frame = cv2.warpPerspective(img, self.matrix,[TRANSFORMWIDTH, TRANSFORMHEIGHT])
        return transformed_frame
    
    def backward(self, img):
        inv_transformed_frame = cv2.warpPerspective(img, self.matrix_inv, [WIDTH, HEIGHT])
        return inv_transformed_frame


if __name__ == '__main__':
    
    img = cv2.imread('./img/0.jpg')
    img_dot = np.copy(img)
    perspectiveTransformation = PerspectiveTransformation()
    transformed_frame = perspectiveTransformation.forward(img)
    inv_transformed_frame = perspectiveTransformation.backward(transformed_frame)
    cv2.circle(img_dot, TRANSFORM_TOP_LEFT, 5, (0, 0, 255), -1)
    cv2.circle(img_dot, TRANSFORM_BOTTOM_LEFT, 5, (0, 0, 255), -1)
    cv2.circle(img_dot, TRANSFORM_TOP_RIGHT, 5, (0, 0, 255), -1)
    cv2.circle(img_dot, TRANSFORM_BOTTOM_RIGHT, 5, (0, 0, 255), -1)
    cv2.imshow('frame', img_dot)
    
    cv2.imshow('transformed_frame', transformed_frame)
    cv2.imshow('transformed_frame_inv', inv_transformed_frame)
    print(img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    """
    import os
    folder_dir = "./img"
    file_lst = os.listdir(folder_dir)
    transform = PerspectiveTransformation()
    for file in file_lst:
        img = cv2.imread('./img/{}'.format(file))
        transform_img = transform.forward(img)
        cv2.imwrite("./transform_img/{}".format(file), transform_img)
    """

