import os
import sys

import numpy as np
import cv2
from PIL import Image
import pickle


def face_capture():
    # For each person enter name and one numeric face id
    name = input("Enter name: ")
    ref_id = input("Enter user id: ")

    # Check if Image Dataset exists
    if not os.path.isdir('imageDataset'):
        os.mkdir('imageDataset/')

    # Create a directory in image directory to save captured images
    if not os.path.isdir('imageDataset/' + name + '/'):
        os.mkdir('imageDataset/' + name + '/')

    # Check if pickle file is present if not create one
    if os.path.exists('ref_name.pkl'):
        f = open("ref_name.pkl", "rb")
        ref_dict = pickle.load(f)
        f.close()
    else:
        ref_dict = {}

    ref_dict[ref_id] = name
    f = open("ref_name.pkl", "wb")
    pickle.dump(ref_dict, f)
    f.close()

    print("\n [INFO] Press 'ESC' to quit, 'c' to capture face")

    # Face count
    count = 0
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)  # Flip Horizontally
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=30,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            resize_roi_gray = cv2.resize(roi_gray, (180, 180))
        cv2.imshow('Image', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # press 'ESC' to quit
            break
        elif k == ord("c"):  # press 'c' for capturing face
            count += 1
            cv2.imwrite("imageDataset/" + name + '/' + name + "_" + ref_id + "_" + str(count) + ".jpg", resize_roi_gray)
        elif count >= 30:
            break


def train_recognizer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'imageDataset')
    # Path of the images in Dataset
    x_train = []
    y_labels = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                path = os.path.join(root, file)
                pil_image = Image.open(path).convert("L")  # Grayscale
                img_numpy = np.array(pil_image, 'uint8')
                img_id = int(os.path.split(file)[-1].split("_")[1])
                x_train.append(img_numpy)
                y_labels.append(img_id)

    print("\n [INFO] Training the recognizer. Please wait a few seconds....")
    recognizer.train(x_train, np.array(y_labels))

    # Check if Trainer directory exists
    if not os.path.isdir('Trainer'):
        os.mkdir('Trainer')

    # save the model in trainer.yml file
    recognizer.save('Trainer/trainer.yml')

    # print No of faces trained
    print("\n [INFO] {0} faces trained. Exiting program".format(len(np.unique(y_labels))))


def open_recognizer():
    recognizer.read('Trainer/trainer.yml')
    f = open("ref_name.pkl", "rb")
    ref_dict = pickle.load(f)
    f.close()

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)  # Flip horizontally
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=30,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            img_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            # print(type(img_id))
            name = ref_dict[str(img_id)]
            conf = round(100 - conf)
            print(str(img_id) + " " + name + " " + str(conf))

        sys.stdout.flush()
        #os.system('clear')
        cv2.imshow('Camera', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    print("1. Capture picture for database \n2. Train Recognizer \n3. Open Recognizer \n")
    i = int(input("What to Do?(1/2/3): "))

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set video width
    cap.set(4, 480)  # Set video height
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if i == 1:
        face_capture()
    elif i == 2:
        train_recognizer()
    elif i == 3:
        open_recognizer()

    # Cleaning up
    cap.release()
    cv2.destroyAllWindows()
