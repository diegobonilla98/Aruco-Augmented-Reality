import cv2
import numpy as np

aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

cam = cv2.VideoCapture(0)

video = cv2.VideoCapture('./data/video_test_1.mp4')
_, image = video.read()

image_height, image_width = image.shape[:2]
src_pts = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])

while True:
    ret_cam, frame = cam.read()
    ret_vid, image = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = gray.shape

    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray, aruco_dictionary)

    if len(corners) == 0:
        continue

    corner = corners[0]
    for point in corner[0]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    dest_pts = np.float32(corner[0])
    M = cv2.getPerspectiveTransform(src_pts, dest_pts)

    transformed = np.zeros((frame_width, frame_height), dtype=np.uint8)
    result = cv2.warpPerspective(image, M, transformed.shape)

    result = np.where(result == 0, frame, result).astype(np.uint8)

    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
