from matplotlib import image
from matplotlib import pyplot
import numpy as np
from imutils import paths
import cv2


from keras.models import load_model

def detect_image():
    image_path = '/content/drive/MyDrive/idd-20k-II/idd20kII/leftImg8bit/train'
    model = load_model('model.h5')

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