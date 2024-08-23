import cv2
import numpy as np
cap = cv2.VideoCapture(0)
corner_lst = []

CHECKERBOARD = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
while True:
    ret, frame = cap.read()
    # 체커보드 코너 찾기
    retval, corners = cv2.findChessboardCorners(frame, [10, 7], 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_FAST_CHECK + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mtx = None
    dist = None
    if retval:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        frame = cv2.drawChessboardCorners(frame, [10, 7], corners, retval)
        
    cv2.imshow('Real-time Video', frame)


    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, 
                                                       gray.shape[::-1], 
                                                       None,
                                                       None)
        np.save("mtx", mtx)
        np.save("dist", dist)
        break
        


cap.release()
cv2.destroyAllWindows()