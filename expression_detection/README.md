# Expression recognition using Tensorflow

## Usage

There are 2 mandatory arguments when running the program:
 - `model` - The name (or location) of your model file. If it doesn't exist, the program will save it there when it finishes training the model. A pre-trained model `128_epochs.keras` is provided in this repository
 - `train` - A path to the training images folder that has subfolders of images, divided by class

And 5 more optional arguments:
 - `-e` or `--epochs` - Specifies how many epochs the training should go over. By default this is 16
 - `-fc` or `--cascade` - Path to a face cascade file. By default it looks for `haarcascade_frontalface_alt.xml`
 - `-v` or `--validate` - Accepts the same argument as `model`, but for the validation folder. It will output to a text file
 - `-s` or `--source` - Specifies either a path to a static image or video, or a number for a camera source. By default its value is 0
 - `c` or `--custom` - Will use my custom method for loading data rather than the Tensorflow one. Warning: it's very slow, but it checkpoints its progress to pick up where it left off

Examples: 
```
python main.py 128_epochs.keras images/train
python main.py 128_epochs.keras images/train -s videos/people_doing_things.mp4 -v images/validation
```

## Functionality

## Statistics and analysis

### Dataset
The dataset was taken from [kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

### What next?
