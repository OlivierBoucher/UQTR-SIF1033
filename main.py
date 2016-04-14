# Travail pratique 2 - SIF1033
# Olivier Boucher
# Catherine Beliveau
# Joel Gbalou
# Maxime Rioux

import numpy as np
import cv2, sys, os

faces_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyes_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_classifier = cv2.CascadeClassifier("haarcascade_mouth.xml")
nose_classifier = cv2.CascadeClassifier("haarcascade_nose.xml")

cap = cv2.VideoCapture("meunier.mov")

skip = 0

while cap.isOpened():
    ret, frame = cap.read()

    # Used to skip 1 in 2 frames
    skip += 1
    if skip % 2 == 0:
        skip = 0
        continue

    faces = faces_classifier.detectMultiScale(frame, 1.3, 4)

    for (x, y, w, h) in faces:
        # NOTE(Olivier): Provide a minimum width to eliminate noise
        if w > 100 and h > 100:
            # NOTE(Olivier): Crop the face from the picture
            face = frame[y:(y + h), x:(x + w)]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            eyes = eyes_classifier.detectMultiScale(gray)

            # NOTE(Olivier): Distinguish right and left eyes based on position
            left_eye_candidates = []
            right_eye_candidates = []
            for (ex, ey, ew, eh) in eyes:
                if ey < h / 2:
                    if ex + ew < w / 2:
                        left_eye_candidates.append((ex, ey, ew, eh))
                    elif ex > w / 2:
                        right_eye_candidates.append((ex, ey, ew, eh))

            # NOTE(Olivier): Find the best candidates based on basic geometry
            eye_pair_candidates = []
            if len(left_eye_candidates) > 0 and len(right_eye_candidates) > 0:
                for (lx, ly, lw, lh) in left_eye_candidates:
                    for (rx, ry, rw, rh) in right_eye_candidates:
                        # Discard if the rectangle is not the same size for both eyes
                        if abs(lw - rw) > 10 or abs(lh - rh) > 10:
                            break
                        # Calculate a ratio for sorting based on x position
                        ratio = abs(lx - rx)
                        eye_pair_candidates.append(((lx, ly, lw, lh), (rx, ry, rw, rh), ratio))

            # NOTE(Olivier): Sort the candidates based on the ratio we calculated
            eye_pair_candidates.sort(key=lambda eye_pair: eye_pair[2])

            # NOTE(Olivier): If we found matching candidates, we display a rectangle over them
            if len(eye_pair_candidates) > 0:
                left_eye = eye_pair_candidates[0][0]
                right_eye = eye_pair_candidates[0][1]

                cv2.rectangle(face, (left_eye[0], left_eye[1]), (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]),
                              (0, 255, 0), 2)
                cv2.rectangle(face, (right_eye[0], right_eye[1]),
                              (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3]), (0, 255, 0), 2)

            # NOTE(Olivier): Find the mouth
            mouths = mouth_classifier.detectMultiScale(gray)
            mouths_candidate = []
            for (mx, my, mw, mh) in mouths:
                # NOTE(Olivier): The position must be near the bottom of the image and the width must a least be 30 pixels
                if my > h / 1.55 and mw > 30:
                    mouths_candidate.append((mx, my, mw, mh))
            # NOTE(Olivier): Sort he candidates based on their y position
            mouths_candidate.sort(key=lambda mouth: mouth[1], reverse=True)

            # NOTE(Olivier): If we found a matching candidate, we display a rectangle over it
            if len(mouths_candidate) > 0:
                mouth = mouths_candidate[0]
                cv2.rectangle(face, (mouth[0], mouth[1]), (mouth[0] + mouth[2], mouth[1] + mouth[3]), (255, 0, 0), 2)

            cv2.imshow("face", face)

    cv2.imshow('main', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
