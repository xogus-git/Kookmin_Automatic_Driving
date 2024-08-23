import cv2
import os
filepath = './input.mp4'
savepath = "./img"
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open:", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(video.get(cv2.CAP_PROP_FPS))
print("length :", length)
print("width : ", width)
print("height : ", height)
print("fps :", fps)

try:
    if not os.path.exists(savepath):
        os.makedirs(savepath)
except OSError:
    print('Error: Creating directory' + savepath)


count = 0
ret = True
while(video.isOpened() and ret):
    ret, image = video.read()
    if not ret:
        break
    if(int(video.get(1)) % fps == 0):
        save_filepath = os.path.join(savepath, "{}.jpg".format(count))
        cv2.imwrite("img/{}.jpg".format(count), image)
    count += 1

video.release()