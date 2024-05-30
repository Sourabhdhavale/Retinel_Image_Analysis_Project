import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.utils import CustomObjectScope

print(tf.test.is_built_with_cuda())

##data.py
##read the images

imsize = 512       #image size
bridge_layer = 512 # bridge layer fileters in U-NET 
num_filters = [32, 64, 128, 256] # Filters in U-net

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path,"./BCSS-master/images/*")))
    masks = sorted(glob(os.path.join(path,"./BCSS-master/masks_tumor/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x,(imsize, imsize))
    x = x/255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (imsize, imsize))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([imsize, imsize, 3])
    y.set_shape([imsize, imsize, 1])
    return x, y

def tf_dataset(x, y, batch=16):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

#model.py
#building the u-net arch



def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)


    return x

def build_model():
    size = imsize
    
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, bridge_layer)

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

##Train.py



def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

if __name__ == "__main__":
    ## Dataset
   # np.random.seed(42);
  #np.random.set_seed(42);
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    #sess = tf.compat.v1.Session(config=config)
    path = "."
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 20

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    if os.path.isfile("./BCSS-master/files/model.h5"):
        with CustomObjectScope({'iou': iou}):
            model = tf.keras.models.load_model("./BCSS-master/files/model.h5")
    else:
        model = build_model()		
    print("Model Buid....")
    model.summary()
    
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0), optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("./BCSS-master/files/model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("./BCSS-master/files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1
    print("Training ....")
    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)