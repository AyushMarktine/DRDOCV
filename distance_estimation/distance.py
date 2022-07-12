import utils
import numpy as np
import cv2


# step 1 - load the model
net = cv2.dnn.readNet(r'best02.onnx')
# step 2 - feed a 640x640 image to get predictions
def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


avg_heights = {'biker': 1, 
                'car': 1.6 , 
                'pedestrian':1.5,
                 'trafficLight':1.6 ,
                  'trafficLight-Green':1.6 ,
                   'trafficLight-GreenLeft':1.6 ,
                    'trafficLight-Red':1.6 ,
                     'trafficLight-RedLeft':1.6 ,
                      'trafficLight-Yellow':1.6 ,
                       'trafficLight-YellowLeft':1.6 ,
                        'truck': 1.6 }


def fn_show_bbox_and_distance(image : str, net, focal_length=0, find_focal = False, measured_distance = 0, confidence_thresh=.70):

    image = cv2.imread(image)
    input_image = format_yolov5(image) # making the image square
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()
    # step 3 - unwrap the predictions to get the object detections 
    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640


    class_list = ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck']

    for r in range(25200): # fixed
        row = output_data[r]
        confidence = row[4]
        if confidence >= confidence_thresh:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1] 
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
                # print(boxes)


    # class_list = []# with open("config_files/classes.txt", "r") as f:#     class_list = [cname.strip() for cname in f.readlines()]
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])


    """
    * Any unit of distances can be used *

    focal_length = (width_found_in_frame * Real_Distance)/(Real_Width) {unit: (pxls*inch)/(inch) = pxls}
    Distance = (Real_Width * Focal_Length)/(Width_in_Frame) {unit: (inch*pxls)/(pxls) = inch}

    """

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]
        height_in_rf_image = box[3] 
    
        cv2.rectangle(image, box, (0, 155, 255), 1)

        if find_focal==True:
            real_height = avg_heights[class_list[class_id]]
            focal_length = utils.FocalLength(measured_distance, real_height, height_in_rf_image)
            print("Focal Length:", focal_length)
            cv2.imwrite("Predictions/Focal_True01.jpg", image)
            return focal_length
            
        else:
            real_height = avg_heights[class_list[class_id]]
            distance = utils.Distance_finder(focal_length, real_height, height_in_rf_image)
            distance = round(distance, 2)
            cv2.putText(image, f"{distance} mts", (box[0]+int(box[2]/2), box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), thickness=2)
            # cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
            cv2.putText(image, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), thickness=2)
            
    temp_path = path.split("\\")[-1]
    cv2.imwrite(rf"TestImages\new\tuned_preds\{temp_path}", image)


"""
Reference image should contain only one object, whether a car, bus, etc.
And its average height should be taken as input
"""

f = fn_show_bbox_and_distance(image = r"TestImages\s4.jpg", 
                                net = net, 
                                find_focal=True, 
                                measured_distance = 4.26,
                                # real_height = 1.6,
                                confidence_thresh = .50) 

from glob import glob
paths = glob(r"TestImages\new\*.jpg")

for path in paths:
    fn_show_bbox_and_distance(image = path,
                                net = net,
                                focal_length=f,
                                # real_height=1.6,
                                confidence_thresh = .60)


"""
# Avg height of the objects should be slightly lower than their actual avgerage height
# Change focal length, for optimal results
"""

