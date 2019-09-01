
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

#############
# Libraries #
#############

import os
import cv2
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

##################
# Preparing data #
##################

IMG_HEIGHT = 258
IMG_WIDTH = 540
N_CHANNELS = 1

train_paths_list = glob.glob("./train/*.png")
target_paths_list = glob.glob("./train_cleaned/*.png")
test_paths_list = glob.glob("./test/*.png")

def load_images(paths):
    images = []
    for i, path in enumerate(paths):
        img = cv2.imread(path, 0) # 0 to return greyscale image
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        images.append(img) 
    images = np.array(images, dtype = np.float32)
    images = np.expand_dims(images, axis = -1)
    return images

x_train = load_images(train_paths_list)
y_train = load_images(target_paths_list)
x_test = load_images(test_paths_list)


x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size = 0.15,
        random_state = 2019
)

####################
# Specifying model #
####################
 
inp = Input(shape = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))     

# encoder
x = Conv2D(64, (3, 3), activation = "relu", padding = "same")(inp)
x = MaxPooling2D((2, 2), padding = "same")(x)

# decoder
x = Conv2D(64, (3, 3), activation = "relu", padding = "same")(x)
x = UpSampling2D((2, 2))(x)
outp = Conv2D(1, (3, 3), activation = "sigmoid", padding = "same")(x)

model = Model(inp, outp)

model.compile(
        loss = "mse",
        optimizer = Adam(lr = 1e-3),
        metrics = ["accuracy"]
)

model.summary()

##################
# Training model #
##################

callbacks_list = [
        EarlyStopping(
                monitor = "val_loss",
                mode = "min",
                patience = 5,
                verbose = 1,

        )
]

fit_log = model.fit(
        x_train,
        y_train,
        batch_size = 20,
        epochs = 200,
        validation_data = (x_val, y_val),
        callbacks = callbacks_list
)

##############
# Evaluation #
##############

plt.plot(fit_log.history["loss"])
plt.plot(fit_log.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper right")
plt.show()

##############
# Prediction #
##############

preds = model.predict(x_test)

# visualize 3 test set images and their predictions
plt.subplots(6, 2, figsize = (15, 20))
for i in range(1, 12, 2):
    plt.subplot(6, 2, i)
    plt.imshow(x_test[i].squeeze(), cmap = "gray")
    plt.axis("off")
    plt.subplot(6, 2, i + 1)
    plt.imshow(preds[i].squeeze(), cmap = "gray")
    plt.axis("off")
plt.tight_layout()    
    
preds = preds.squeeze() # remove extra dimension

ids = []
vals = []
for i, f in enumerate(test_paths_list):
    file = os.path.basename(f)
    imgid = int(file[:-4])
    test_img = cv2.imread(f, 0)
    height, width = test_img.shape[0], test_img.shape[1]
    preds_reshaped = cv2.resize(preds[i], (width, height))
    for r in range(height):
        for c in range(width):
            ids.append(str(imgid)+"_"+str(r + 1)+"_"+str(c + 1))
            vals.append(preds_reshaped[r, c])

test_df = pd.DataFrame({"id": ids, "value": vals})
test_df.to_csv("submission.csv", index = False)
