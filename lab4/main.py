import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_dataset(csv_path, image_folder, dnn_mode=False):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for index, row in df.iterrows():
        image_path = f"{image_folder}/{row['filename']}"
        image = cv2.imread(image_path)
        if image is not None:
            if dnn_mode:
                image = cv2.resize(image, (400,400), interpolation=cv2.INTER_CUBIC)
                image = image / 255.0
            images.append(image)
            labels.append(row['label'])
        else:
            print(f"Warning: Unable to load image {image_path}")
    return images, labels, df['filename']

def match_features(img1, img2, sift):
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75* n.distance:  
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if len(good_matches) > 4:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        num_inliers = sum(matches_mask)
    else:
        H, matches_mask, num_inliers = None, None, 0

    return good_matches, matches_mask, num_inliers, H

def sift_ransac_stats(train_images, test_images, train_filenames, test_filenames, sift):
    for test_idx, test_img in enumerate(test_images):
        best_match = None
        best_num_inliers = 0
        best_train_idx = -1

        for train_idx, train_img in enumerate(train_images):
            good_matches, matches_mask, num_inliers, H = match_features(test_img, train_img, sift)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_match = (good_matches, matches_mask, H)
                best_train_idx = train_idx

        if best_train_idx != -1:
            train_img = train_images[best_train_idx]
            good_matches, matches_mask, H = best_match

            img_matches = cv2.drawMatches(test_img, sift.detectAndCompute(test_img, None)[0],
                                        train_img, sift.detectAndCompute(train_img, None)[0],
                                        good_matches, None, matchesMask=matches_mask,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imwrite(f"ransac_matches/{test_filenames[test_idx]}-{train_filenames[best_train_idx]}.jpg", img_matches)

            print(f"Number of Inliers: {best_num_inliers}")
            print("Homography Matrix:")
            print(H)
            print("-" * 50)

def train_neural_network(X_train, y_train):
    model = Sequential([
        Input(shape=(400,400,3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

# Main workflow
if __name__ == "__main__":
    train_csv = "train.csv" 
    train_image_folder = "train_images"
    test_csv = "test.csv"
    test_image_folder = "test_images"

    test_images, test_labels, _ = load_dataset(test_csv, test_image_folder, True)
    train_images, train_labels, _ = load_dataset(train_csv, train_image_folder, True)
    
    train_srac, _, train_filenames = load_dataset(train_csv, train_image_folder)
    test_srac, _, test_filenames = load_dataset(test_csv, test_image_folder)
    sift = cv2.SIFT_create()
    sift_ransac_stats(train_srac, test_srac, train_filenames, test_filenames, sift)

    if not os.path.isfile("dnn_model.keras"):
        model = train_neural_network(np.array(train_images), np.array(train_labels))
        model.save("dnn_model.keras")

    else:
        model = tf.keras.models.load_model("dnn_model.keras")

    test_loss, test_acc = model.evaluate(np.array(test_images), np.array(test_labels))
    print(test_acc)
    