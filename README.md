# Tom-Jerry-Emotion-Detection
This is my solution for Tom and Jerry Emotion recognition challenge conducted by HackerEarth. The Dataset consists of a video and a csv file which has emotion for 298 frames of the video. Our task is to find the emotion of tom or jerry [priority: Tom > Jerry] in the image for 185 test frames. <br>
There are totally 5 emotions:
1. Angry
2. Happy
3. Sad
4. Surprised
5. Unknown

## Implemetation Details
Since we only have a limited amount of data for training, I have synthesised more images by using the various data augmentation techniques like rotation, shearing, shift-left, shift-right, flip, gaussian blur and adding random noise. <br>
Once the data was augmented, I found that the data was heavily imbalanced. So I downsampled the required class and made sure all the classes had more or less the same number of images. <br>
Finally, I tried training the dataset and found an accuracy spike from 25% before data augmentation and balancing to 40%. <br>

## How to run

```.bash
python data-ready.py
```
This generates frames from the training video and saves it as a h5py file. <br>
 
```.bash
python data-augment.py
```

This uses the generated h5py file and augments each images and increases the size of the dataset.<br>

```.bash
python data-balanced.py
```

This downsamples the dataset and balances out the dataset.<br>

```.bash
python train.py
```

This uses the augmented and balanced data generated and runs ResNet classifier model on them. As of now, the classifier model is ResNet101, which gives an accuracy of 40%. Model can be changed in model.py and used in the train.py file. <br>

