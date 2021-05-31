import cv2
import mediapipe as mp
import numpy as np
import time

video = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
specs = mpDraw.DrawingSpec(thickness=0, circle_radius=1)
facee_mesh = mp.solutions.face_mesh
mahface = facee_mesh.FaceMesh()

while True:
    ret, frame = video.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mahface.process(rgb)

    blank_image = np.zeros((1000,1000,3), np.uint8)

    if results.multi_face_landmarks:
        for mark in results.multi_face_landmarks:
            list = []
            for id, lm in enumerate(mark.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                list.append((cx, cy))
            left_pts = [list[93], list[132], list[58], list[172], list[136], list[150],list[149], list[176], list[148]]
            right_pts = [list[377], list[400], list[379], list[365], list[397], list[288], list[361], list[323], list[93]]
            for i in range(len(left_pts) - 1):
                cv2.line(frame, left_pts[i], left_pts[i + 1], (277, 188 ,0), 35)
            for n in range(len(right_pts) - 1):
                cv2.line(frame, right_pts[n], right_pts[n + 1], (277, 188 ,0), 35)

            e = len(left_pts) - 1
            while e > 0:
                cv2.line(frame, right_pts[e], left_pts[e], (277, 188 ,0), thickness=20)
                cv2.line(frame, right_pts[-e], left_pts[-e], (277, 188 ,0), thickness=20)
                cv2.line(frame, right_pts[e], list[93], (277, 188 ,0), thickness=20)
                cv2.line(frame, left_pts[e], list[323], (277, 188 ,0), thickness=20)
                e = e - 1
            cv2.line(frame, list[148], list[152], (277, 188 ,0), thickness=35)
            cv2.line(frame, list[152], list[377], (277, 188 ,0), thickness=35)
    cv2.imshow("FaceLeak", frame)
    cv2.waitKey(1)