import sys
def print_help_exit():
    print("Usage:\n-h or help -- Prints this section")
    print("path/to/training/images -- Absolute or relative path to your folder of training images divided into sub-folders")
    print("modelName.format -- The name and format in which the model should be saved as\n")
    print("Optional arguments:")
    print("epochs -- Integer value for epochs when training. Default is 16")
    print("-v or validate -- Will validate the model rather than predicting from a live video feed")
    print("path/to/validation/images -- Absolute or relative path to your folder of validation images divided into sub-folders")
    print("-c or custom -- Will use a custom data loading method over the Tensorflow one. Warning: it's very slow and its usage is not recommended")
    exit()

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == ("-h" or "help"):
        print_help_exit()
    if len(sys.argv) > 7:
        print("Too many arguments! Use -h or help.")
        print_help_exit()

    train_image_folder = str(sys.argv[1])
    model_name = str(sys.argv[2])
    csvs_folder = "csvs"

    import model_works, preprocess
    if len(sys.argv) == 7 and str(sys.argv[6]) == ("-c" or "custom"):
        train_csv = "train.csv"
        preprocess.label_folders(train_image_folder, csvs_folder, train_csv)
        train_images, train_labels = preprocess.load_data(f"{csvs_folder}/{train_csv}", train_image_folder)
        model = model_works.make_model(inputData = train_images, inputLabels = train_labels, saveModelName = model_name)

        if str(sys.argv[4]) == ("-v" or "validate"):
            validate_csv = "validate.csv"
            preprocess.label_folders(validation_image_folder, csvs_folder, validate_csv)
            validation_images, validation_labels = preprocess.load_data(f"{csvs_folder}/{validate_csv}", validation_image_folder)
            test_loss, test_acc = model_works.validate_model(model, validateData = validation_images, validateLabels = validation_labels)
            print(f"Stats for model {model_name}:\nTest loss: {test_loss}\nTest accuracy: {test_acc}")
            exit()

    epochs = 16
    if len(sys.argv) == 4:
        epochs = sys.argv[3]

    model, class_names = model_works.make_model(train_image_folder, epochs, model_name)

    if len(sys.argv) == 6 and str(sys.argv[4]) == ("-v" or "validate"):
        validation_image_folder = str(sys.argv[5])
        test_loss, test_acc = model_works.validate(model, validation_image_folder)
        print(f"Stats for model {model_name}:\nTest loss: {test_loss}\nTest accuracy: {test_acc}")
        exit()

    import cv2
    from PIL import Image
    import numpy as np
    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')
        im = im.resize((48,48))
        img_array = np.array(im)

        print(model_works.predict_image(model, class_names, img_array))

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()