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

cap = cv2.VideoCapture("C:\\Users\\baudisso\\Desktop\\UQTR-SIF1033\\meunier.mp4")

btw_puppils_ratios = []
btw_eyes_ratios = []
nose_to_mouth_ratios = []

face_absence_cnt = 0
last_detection_was_successful = False
frames_elapsed = 0l

while cap.isOpened():
    ret, frame = cap.read()
    frames_elapsed += 1
    faces = faces_classifier.detectMultiScale(frame, 1.3, 4)

    # NOTE(Olivier): Provide a minimum width to eliminate noise
    # TODO(Olivier): Adjust from hardcoded values to percentage from frame size
    faces = filter(lambda l: l[2] > 100 and l[3] > 100, faces)

    if len(faces) == 0:
        face_absence_cnt += 1

    for (x, y, w, h) in faces:
        right_eye = None
        left_eye = None
        mouth = None
        nose = None
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

        # NOTE(Olivier): Find the nose
        noses = nose_classifier.detectMultiScale(gray)
        nose_candidates = []
        for (nx, ny, nw, nh) in noses:
            if (h * 0.25) < ny < (h * 0.75):
                if (w * 0.25) < nx < (w * 0.75):
                    # NOTE(Olivier): Assign some sort of rank based on position from center of the face
                    indice = abs((w/2)-nx) + abs((h/2)-ny)
                    nose_candidates.append(((nx, ny, nw, nh), indice))

        # NOTE(Olivier): Sort based on distance from the center
        nose_candidates.sort(key=lambda nose: nose[1])

        # NOTE(Olivier): If we found a matching candidate, we display a rectangle over it
        if len(nose_candidates) > 0:
            nose = nose_candidates[0][0]
            cv2.rectangle(face, (nose[0], nose[1]), (nose[0] + nose[2], nose[1] + nose[3]), (0, 0, 255), 2)

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

        # NOTE(Olivier): Begin face analysis based on nose, eyes and mouth
        if right_eye is not None:
            d_btw_puppils = (right_eye[0] + right_eye[2]/2) - (left_eye[0] + left_eye[2]/2)
            # NOTE(Olivier): This measure is more accurate, meunier > 0.10 and its son < 0.10
            d_btw_eyes = (right_eye[0] - (left_eye[0] + left_eye[2]))

            btw_puppils_ratios.append(float(d_btw_puppils)/float(w))
            btw_eyes_ratios.append(float(d_btw_eyes)/float(w))

        if mouth is not None and nose is not None:
            nose_mouth_y = (mouth[1] + mouth[3]/2) - (nose[1] + nose[3]/2)
            nose_to_mouth_ratios.append(float(nose_mouth_y)/float(mouth[2]))

        cv2.imshow("face", face)

    # NOTE(Olivier): 29 fps -> if there was no face detected for more than a second
    if face_absence_cnt > 29:
        if not last_detection_was_successful and len(nose_to_mouth_ratios) <= 15:
                face_absence_cnt = 0
        else:
            avg_puppils_ratio = reduce(lambda n, m: n+m, btw_puppils_ratios)/len(btw_puppils_ratios) if len(btw_puppils_ratios) > 15 else 0
            avg_eyes_ratio = reduce(lambda n, m: n+m, btw_eyes_ratios)/len(btw_eyes_ratios) if len(btw_eyes_ratios) > 15 else 0
            avg_nose_mouth = reduce(lambda n, m: n+m, nose_to_mouth_ratios)/len(nose_to_mouth_ratios) if len(nose_to_mouth_ratios) > 15 else 0

            if avg_nose_mouth > 0:
                print '[' + str(frames_elapsed/29) + ' seconds in] Nose to mouth: ' + str(avg_nose_mouth) + ' | Eyes: ' + str(avg_eyes_ratio) + ' | Puppils: ' + str(avg_puppils_ratio)

            # Reset the average counters
            last_detection_was_successful = True
            btw_puppils_ratios = []
            btw_eyes_ratios = []
            nose_to_mouth_ratios = []
            face_absence_cnt = 0

    cv2.imshow('main', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
