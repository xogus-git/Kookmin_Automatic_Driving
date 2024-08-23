import cv2
import numpy as np
import os
class CannyEdge:
    def forward(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)
        return edge_img



if __name__ == '__main__':
    from PerspectiveTransformation import *
    os.makedirs("./canny_img", exist_ok=True)
    transform = PerspectiveTransformation()
    canny = CannyEdge()
    folder_dir = "./img"
    file_lst = os.listdir(folder_dir)
    for file in file_lst:
        img = cv2.imread('./img/{}'.format(file))
        img = transform.forward(img)
        out_img = canny.forward(img)
        #out_img = cv2.hconcat([img, out_img])
        cv2.imwrite("./canny_img/{}".format(file), out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
