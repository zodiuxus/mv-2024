import sys
import argparse

def args_parse():
    argparser = argparse.ArgumentParser(description="Emotion recognition over video.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_args = argparser.add_argument_group("model")
    mutex_args = argparser.add_mutually_exclusive_group(required=True)
    optional_args = argparser.add_argument_group("other")

    model_args.add_argument("model", help="loads an existing model (e.g. mymodel.keras) if found, otherwise it trains a model and saves it with the given name")
    model_args.add_argument("train", help="used to load the training dataset and its classes")
    model_args.add_argument("-e", "--epochs", help="number of epochs to train the model for", default=16)
    mutex_args.add_argument("-s", "--source", help="can be an integer to use a live video feed, or a path to use a pre-recorded video")
    mutex_args.add_argument("-v", "--validate", help="will validate the model instead of predicting per frame; the path to the validation dataset needs to be passed")
    optional_args.add_argument("-fc", "--cascade", help="loads a pre-trained face cascade file", default="haarcascade_frontalface_alt.xml")
    optional_args.add_argument("-c", "--custom", help="will use a custom data loading method over the Tensorflow one; warning: it's very slow and usage is not recommended", action="store_true", default=False)

    return argparser.parse_args()

if __name__ == '__main__':
    args = args_parse()

    model_name = args.model
    train_image_folder = args.train
    epochs = args.epochs

    source = str(args.source)
    if source.isdigit():
        source = int(source) # such an odd hack, but it works fine

    validate = args.validate

    cascade = str(args.cascade)

    custom_loading = args.custom

    import model_works, preprocess
    if custom_loading:
        csvs_folder = "csv"
        train_csv = "train.csv"
        preprocess.label_folders(train_image_folder, csvs_folder, train_csv)
        train_images, train_labels = preprocess.load_data(f"{csvs_folder}/{train_csv}", train_image_folder)
        model = model_works.make_model(inputData = train_images, inputLabels = train_labels, saveModelName = model_name)

        if validate:
            validate_csv = "validate.csv"
            preprocess.label_folders(validate, csvs_folder, validate_csv)
            validation_images, validation_labels = preprocess.load_data(f"{csvs_folder}/{validate_csv}", validate)
            test_loss, test_acc = model_works.validate_model(model, validateData = validation_images, validateLabels = validation_labels)
            print(f"Stats for model {model_name}:\nTest loss: {test_loss}\nTest accuracy: {test_acc}")
            exit()

    model, class_names = model_works.make_model(train_image_folder, epochs, model_name)

    if validate:
        test_loss, test_acc = model_works.validate(model, validate)
        print(f"Stats for model {model_name}:\nTest loss: {test_loss}\nTest accuracy: {test_acc}")
        exit()

    import cv2
    from PIL import Image
    import numpy as np
    video = cv2.VideoCapture(source)
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(cascade)):
        print("Error loading face cascade file.")
        exit()

    while True:
        _, frame_orig = video.read()

        frame_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)
        for (x,y,w,h) in faces:
            frame = frame_orig[y:y+h, x:x+w]

            im = Image.fromarray(frame, 'RGB')
            im = im.resize((48,48))
            img_array = np.array(im)

            class_predicted, confidence = model_works.predict_image(model, class_names, img_array)

            cv2.rectangle(frame_orig, (x,y), (x+w,y+h), (247, 83, 182))
            cv2.putText(frame_orig, f"{class_predicted}: {confidence:.2f}%", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (247, 83, 182), 1)

        cv2.imshow("Capture", frame_orig)
        cv2.imwrite(f"{source}_predictions.jpg", frame_orig)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
