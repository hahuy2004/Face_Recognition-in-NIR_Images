from yoloface import face_analysis
import cv2
import numpy as np

face = face_analysis()

def yoloface_detection(frame):
    img, box, conf = face.face_detection(
        frame_arr=frame, frame_status=True, model='full')
    # Box là list bounding boxes của khuôn mặt
    # box[i] = [x, y, w, h]
    cropped_images = []
    faces_rect = np.array([np.array(xi) for xi in box])
    for (x, y, w, h) in faces_rect:  # Phát hiện và cắt nhiều khuôn mặt trong mỗi khung hình
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), thickness=2)
        cv2.imwrite('yolo.jpg', frame)
        crop_img = frame[y:y+w, x:x+h]
        if len(crop_img) != 0:
            cropped_images.append(crop_img)
    return cropped_images
