##TESTING THE DATASET AND PREDICTING THE RESULTS
#predict.py
import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from tensorflow.keras.utils import CustomObjectScope
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from data import load_data, tf_dataset
#from train import iou
from bcancer import * 
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path,"./BCSS-master/images/*")))
    masks = sorted(glob(os.path.join(path,"./BCSS-master/masks/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

if __name__ == "__main__":
    ## Dataset
    path = "."
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("./BCSS-master/files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        #cv2.imwrite("./results/"+str(i)+".png", y_pred)
        h, w, _ = x.shape
        white_line = np.ones((h, 10, 3)) * 255.0
        print(mask_parse(y_pred).shape)
        print(mask_parse(y).shape)
        print(x.shape)
        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{i}.png", image)
    print(type(image))