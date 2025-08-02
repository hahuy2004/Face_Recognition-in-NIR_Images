from yoloface_face_detection import yoloface_detection
import cv2
import os

def get_crop_images(crop_images, path, original_filename):
    """Lưu danh sách ảnh khuôn mặt với tên dựa trên ảnh gốc"""
    for i, image in enumerate(crop_images):
        name_base = os.path.splitext(original_filename)[0]
        file_path = os.path.join(path, f"{name_base}.jpg")
        cv2.imwrite(file_path, image)

def get_all_faces(images_folders_path, output_base_path):
    for split in os.listdir(images_folders_path):  # train, val, test
        split_path = os.path.join(images_folders_path, split)
        if not os.path.isdir(split_path):
            continue

        for person_id in os.listdir(split_path):  # 1, 2, ..., 25
            person_folder_path = os.path.join(split_path, person_id)
            if not os.path.isdir(person_folder_path):
                continue

            for file in os.listdir(person_folder_path):
                if not file.lower().endswith(".jpg"):
                    continue

                image_path = os.path.join(person_folder_path, file)
                img = cv2.imread(image_path)

                # Gọi YOLOFace
                yolo_cropped = yoloface_detection(img)

                # Tạo thư mục lưu kết quả
                save_folder = os.path.join(output_base_path, split, person_id)
                os.makedirs(save_folder, exist_ok=True)

                get_crop_images(yolo_cropped, save_folder, file)

def main():
    input_path = '/content/TD_NIR_A_Set'
    output_path = '/content/yolo_cropped_images'
    get_all_faces(input_path, output_path)

if __name__ == '__main__':
    main()