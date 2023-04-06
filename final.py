# import the necessary packages
import numpy as np
import time
import cv2
import os


models='yolov4'

# load the COCO class labels our YOLO model was trained on
with open("coco.names", "r") as f:
  class_names = [cname.strip() for cname in f.readlines()]

confidence=0.6  #Setting confidence threshold for detection
Nms=0.3         #Non max suppression threshold
class_names = []  #To store class names 

arc = cv2.dnn.readNet('yolov4.weights','yolov4.cfg')
arc.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
arc.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(arc)
model.setInputParams(size=(640,640), scale=1/255,swapRB=True)




vid = cv2.VideoCapture('videos/conv_new.mp4')

def pil_to_cv(pil_image):
    return np.array(pil_image)[:, :, ::-1]

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
result = cv2.VideoWriter('final.mp4', fourcc,5, size)


while(True):

    ret,frame = vid.read() 
    if ret == False:
        print('video failed!!')

    x = time.time()
    classes, scores, boxes = model.detect(frame, confidence, Nms)
    y= time.time()
    fps=1/(y-x)
    
    # Initialize variables for person and baggage bounding boxes
    person_box = None
    baggage_box = None
    p_box = []
    b_box = []
    
    # Looping through all detected objects
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %.2f" % (class_names[classid],score)

        
        # Checking if the detected object is a person or baggage
        if class_names[classid] == 'person':
            person_box = box
            p_box.append(person_box)
        elif class_names[classid] == 'handbag' or class_names[classid] == 'suitcase' or class_names[classid] == 'backpack':
            baggage_box = box
            b_box.append(list(baggage_box))


        # Draw bounding box with different colors depending on whether the object is a person or baggage
        if class_names[classid] == 'person':
            cv2.rectangle(frame,box,color=(255, 0, 0),thickness=1)
            cv2.putText(frame,label, (box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(25,55,255),1)
        # elif class_names[classid] == 'handbag' or class_names[classid] == 'suitcase' or class_names[classid] == 'backpack':
        #     cv2.rectangle(frame,box,color=(0, 0, 255),thickness=1)
        #     cv2.putText(frame,label, (box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(25,55,255),1)

    
    not_aban = []   #to store bounding boxes of not abandoned baggage
    aban = []       #to store bounding boxes of abandoned baggage

    #Iterating through bounding boxes of all person wrt to baggages
    for i in range(len(p_box)):
        for j in range(len(b_box)):
            if b_box[j] not in not_aban:
            # Calculate the IOU between the two bounding boxes
                xA = max(p_box[i][0], b_box[j][0]) # x coord top left
                yA = max(p_box[i][1], b_box[j][1]) # y coord top left
                xB = min(p_box[i][0] + p_box[i][2], b_box[j][0] + b_box[j][2]) # bottom right x cord
                yB = min(p_box[i][1] + p_box[i][3], b_box[j][1] + b_box[j][3]) # bottom right y cord
                interArea = max(0,xB - xA) * max(0,yB - yA) # if dont intersect then area will be 0
                personArea = (p_box[i][2]) * (p_box[i][3])
                baggageArea = ((b_box[j][2]) * (b_box[j][3]))
                
                # Calculate the union area
                unionArea = float(personArea + baggageArea - interArea)

                # Calculate the IOU
                iou = interArea / unionArea


                iou_threshold = 0.01    #TRIAL AND ERROR
                euc_threshold = 100     #TRIAL AND ERROR
                distance = np.sqrt((int(p_box[i][0]+(p_box[i][2]//2)) - int(b_box[j][0]+(b_box[j][2]//2)))**2 + 
                                (int(p_box[i][1]+(p_box[i][3]//2)) - int(b_box[j][1]+(b_box[j][3]//2)))**2)
                
                if distance < euc_threshold:
                    if iou > iou_threshold:
                        cv2.putText(frame, "NOT ABANDONED", (b_box[j][0], b_box[j][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        cv2.rectangle(frame,(b_box[j][0],b_box[j][1]),(b_box[j][0] + b_box[j][2],b_box[j][1] + b_box[j][3]),color=(0, 255, 0),thickness=1)
                        not_aban.append(b_box[j])
                else:
                    continue

            else:
                continue

    res_set = set(map(tuple, b_box)) ^ set(map(tuple, not_aban))
    aban = list(map(list, res_set))
    for i in range(len(aban)):
        cv2.putText(frame, "ABANDONED", (aban[i][0], aban[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.rectangle(frame,(aban[i][0],aban[i][1]),(aban[i][0] + aban[i][2],aban[i][1] + aban[i][3]),color=(255, 0, 0),thickness=2)

    text =  "Abandoned baggages: {}".format(len(aban))
    cv2.putText(frame,text,(60,70),cv2.FONT_HERSHEY_PLAIN,fontScale=4,color=(255,255,255),thickness=2)
    cv2.putText(frame, "FPS:{0:.2f}".format(fps),(20, 25), cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255, 0, 0),thickness=2)
    
    
    #writing the output frame to disk
    result.write(pil_to_cv(frame))

    # show the output frame
    cv2.imshow("Frame", cv2.resize(frame,(800,500)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



vid.release()
result.release()
cv2.destroyAllWindows()