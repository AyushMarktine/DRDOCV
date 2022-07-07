from imutils import paths
from tqdm import tqdm
import numpy as np
import cv2
import gc
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
# # Load the TensorBoard notebook extension
# from keras.callbacks import TensorBoard
# %load_ext tensorboard
import tensorflow as tf
import datetime, os
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
# Load the TensorBoard notebook extension
# from keras.callbacks import TensorBoard
# %load_ext tensorboard
import tensorflow as tf
import datetime, os
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = TensorBoard("logs", histogram_freq=1)



earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
mcp_save = ModelCheckpoint('/temp', save_best_only=True, monitor='val_loss', mode='min')

from imutils import paths
from tqdm import tqdm

import numpy as np

import cv2

from model import deeplabv3_plus



from keras.utils.np_utils import to_categorical
import numpy as np
colors = np.array([
    [0,1,0,0,0,0,0,0],      # Drivable
    [0,0,1,0,0,0,0,0],     # Non Drivable
    [0,0,0,1,0,0,0,0],      # Living Things
    [0,0,0,0,1,0,0,0],        # Vehicles
    [0,0,0,0,0,1,0,0],     # Road Side Objects
    [0,0,0,0,0,0,1,0],       # Far Objects
    [0,0,0,0,0,0,0,1],     # Sky
    [1,0,0,0,0,0,0,0]           # Misc
], dtype=np.int)

valimage_path  = r"E:\down\New folder\idd20kII\leftImg8bit\val"
vallabel_path = r"E:\down\New folder\idd20kII\vallabel"
trainimage_path = r"E:\down\New folder\idd20kII\leftImg8bit\train"
trainlabel_path = r"E:\down\New folder\idd20kII\trainlabel"


def IoU(y_val, y_pred):
    class_iou = []
    n_classes = 8
    
    y_predi = np.argmax(y_pred, axis=3)
    y_truei = np.argmax(y_val, axis=3)
    
    for c in range(n_classes):
        TP = np.sum((y_truei == c) & (y_predi == c))
        FP = np.sum((y_truei != c) & (y_predi == c))
        FN = np.sum((y_truei == c) & (y_predi != c)) 
        IoU = TP / float(TP + FP + FN)
        if(float(TP + FP + FN) == 0):
          IoU=TP/0.001
        class_iou.append(IoU)
    MIoU=sum(class_iou)/n_classes
    return MIoU
def miou( y_true, y_pred ) :
    score = tf.py_function( lambda y_true, y_pred : IoU( y_true, y_pred).astype('float32'),
                        [y_true, y_pred],
                        'float32')
    return score



def training():
    labels_path = paths.list_images(vallabel_path)
    labels_path = sorted(labels_path)
    image_y_cv = []
    for name in tqdm(labels_path):
        image = cv2.imread(name)
        image = cv2.resize(image, (512, 512))
        r,g,b =cv2.split(image)
        color_image = np.zeros(
                (r.shape[0], r.shape[1], 8), dtype=np.int)
        for i in range(8):
            color_image[r == i] = colors[i]
        image_y_cv.append(color_image)



    x_path = paths.list_images(valimage_path)
    x_path = sorted(x_path)


    image_cv = []
    for name in tqdm(x_path):
        image = cv2.imread(name)
        image_cv.append(cv2.resize(image, (256, 256))/255)


    image_cv = np.array(image_cv)
    image_y_cv = np.array(image_y_cv)
    image_y_cv.shape,image_cv.shape



    x_path = paths.list_images(trainimage_path)
    x_path = sorted(x_path)
    x_path[:5]

    labels_path = paths.list_images(trainlabel_path)
    labels_path = sorted(labels_path)


    image_y = []
    for name in tqdm(labels_path):
        image = cv2.imread(name)
        image = cv2.resize(image, (256, 256))
        r,g,b =cv2.split(image)
        color_image = np.zeros(
            (r.shape[0], r.shape[1], 8), dtype=np.int)
        for i in range(8):
            color_image[r == i] = colors[i]

        image_y.append(color_image)

    image_tr = []
    for name in tqdm(x_path):
        image = cv2.imread(name)
        img = cv2.resize(image, (256, 256))
        img = np.float32(img)  / 255 
        image_tr.append(img)

    image_tr = np.array(image_tr)
    image_y = np.array(image_y)

    model = deeplabv3_plus((256, 256, 3))
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'],run_eagerly=True)
    history = model.fit(image_tr,image_y,batch_size=64,epochs = 3 ,callbacks=[earlyStopping, mcp_save],validation_data=(image_cv,image_y_cv))
    model.save("deeplab_batch_norm.hdf5")


if __name__ == "__main__":
   training()
