from matplotlib import image
from matplotlib import pyplot
import numpy as np
from imutils import paths
import cv2
import tensorflow as tf

from keras.models import load_model

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

def detect_image():
    image_path = '/content/drive/MyDrive/idd-20k-II/idd20kII/leftImg8bit/train'
    model = tf.keras.models.load_model('model.h5',custom_objects={"miou": miou}) #model path

    x_path = paths.list_images(image_path)
    x_path = sorted(x_path)
    image_te = []
    image = cv2.imread(x_path[0])
    img = cv2.resize(image, (256, 256))
    img = np.float32(img)  / 255 
    image_te.append(img)

    image_te = np.array(image_te)
    result = model.predict(image_te)
    result = np.argmax(result, axis=3)
    colors = np.array([
        [255, 192 ,203	],      
        [255, 160, 122],     
        [255, 105, 180],      
        [205,  92,  92],        
        [255, 165,   0],    
        [255, 255,   0],      
        [165,  42,  42],     
        [0,   0, 255]           
    ], dtype=np.int)
    image_y_te = []

    color_image = np.zeros(
            (result.shape[1], result.shape[2], 3), dtype=np.int)
    for i in range(8):
        color_image[result[0] == i] = colors[i]


    image_y_te.append(color_image)
    image_y_te = np.array(image_y_te)

    pyplot.imshow(color_image)
    pyplot.show()



if __name__ == "__main__":
   detect_image()