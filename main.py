# Import packages
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime
import os
from yolo import YOLO 
from PIL import Image, ImageDraw, ImageFont
from sort import *
import tensorflow as tf
from timeit import default_timer as timer
import math
import datetime


def main(yolo):

    # Put your video path here
    Root_Dir = '/content/drive/My Drive/counting_and_speed'
    vid_list = list(os.listdir(os.path.join(Root_Dir, "videos")))
    print(vid_list)
    
    # Set the font and size of output text on the video frame
    font = ImageFont.truetype(font='/content/drive/My Drive/counting_and_speed/font/FiraMono-Medium.otf', size=35)

    # Set x,y coordinates of the virtual lines; line1 for counting, line3 for speed detection
    line1 = [(130, 420), (410, 420)] 
    line3 = [(275, 238), (435, 238)]
    
    # Creation of text files for storing output
    text_file_counting = open('output/' + 'data_counting' + '.txt', 'w')
    text_file_speed = open('output/' + 'data_speed' + '.txt', 'w')
    header_counting = 'Class,Direction,Time\n'
    text_file_counting.write(header_counting)
    header_speed = 'Class,Direction,Speed,Time\n'
    text_file_speed.write(header_speed)
    
    # Outputs True if AB and CD lines intersect
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    # Loop over videos
    for vid in vid_list:
        frameIndex = 0
        speed_id = {}    
        tracker = Sort()
        memory = {}
        
        #Initialise the counter to zero for every new video
        car = 0
        bicycle = 0
        bus = 0
        truck = 0
        car2 = 0 
        bicycle2 = 0
        bus2 = 0
        truck2 = 0
        
        writer = None
        (W, H) = (None, None)
        print('\nVideo: {}...............'.format(vid))
        vs_path = Root_Dir + '/videos/' + vid
        vs = cv2.VideoCapture(vs_path)
        
        # Loop over frames from the video file
        while True:
            print('\nFrame No: {}'.format(frameIndex))
            # Read the next frame from the file
            (grabbed, frame) = vs.read()
            
            # Skip alternate frame; to reduce compute time
            if (frameIndex % 2) != 0:
                frameIndex += 1
                continue 
            
            # Break after reaching end of video file
            if not grabbed:
                break
    
            image = Image.fromarray(frame[...,::-1]) # Modify image format from OpenCV to Yolo
            boxes, out_class, confidences, midPoint = yolo.detect_image(image)  # Yolo detection in action

            # Non-Maximum Suppression to filter out low-confidence and overlapping BBoxes 
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)  #NMSBoxes(bboxes, scores, score_threshold, nms_threshold)
            
            dets = []
            if len(idxs) > 0:
                # Filter the indexes satisfying NMS criterion
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x+w, y+h, confidences[i]])
                    
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            dets = np.asarray(dets)
            
            # Update the accumulated tracking data based on current frame using SORT Algorithm
            tracks = tracker.update(dets)
    
            boxes = []
            indexIDs = []
            previous = memory.copy()
            memory = {}
            
            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                memory[indexIDs[-1]] = boxes[-1]
            
            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    # Bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
    
                    color = (0,0,255)
                    cv2.rectangle(frame, (x, y), (w, h), color, 1)
                    
                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        # Calculation of mid-points p0 & p1
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))

                        detected_class = yolo.counter(p0, out_class, midPoint)
                        test_class = ['car', 'bus', 'truck']
                        direction = None

                        # Conditional statement to check crossing of speed detection line by objects belonging to test class
                        if intersect(p0, p1, line3[0], line3[1]) and (detected_class in test_class):
                            if p0[1] > p1[1]:
                                speed_id[indexIDs[i]] = [frameIndex, 'in']
                            else:
                                speed_id[indexIDs[i]] = [frameIndex, 'out']
                                
                        # Conditional statement to check crossing of counting line by objects belonging to test class
                        if intersect(p0, p1, line1[0], line1[1]) and (detected_class in test_class ):
                            if p0[1] > p1[1]:
                                # Counter increment for specific vehicle class for left lane
                                if detected_class == 'car':
                                    car = car + 1
                                elif detected_class == 'bus':
                                    bus = bus + 1
                                elif detected_class == 'truck':
                                    truck = truck + 1
                                direction = 'in'
                                
                            else:
                                # Counter increment for specific vehicle class for right lane
                                if detected_class == 'car':
                                    car2 = car2 + 1
                                elif detected_class == 'bus':
                                    bus2 = bus2 + 1
                                elif detected_class == 'truck':
                                    truck2 = truck2 + 1
                                direction = 'out'  
                            
                            # Extraction of time stamp from the video filename and subsequent update of time using frame-rate
                            count_date = vid[0:8]
                            count_hour = vid[9:11]
                            count_min = vid[11:13]
                            count_sec = vid[13:15]
                            
                            if count_hour[0] == '0':
                                count_hour = int(count_hour[1])  
                            else:
                                count_hour = int(count_hour) 
                            
                            if count_min[0] == '0':
                                count_min = int(count_min[1])  
                            else:
                                count_min = int(count_min) 
                            
                            if count_sec[0] == '0':
                                total_sec = int((frameIndex+1)/25) + int(count_sec[1])
                            else:
                                total_sec = int((frameIndex+1)/25) + int(count_sec)
                            
                            if total_sec > 59:
                                count_sec = total_sec % 60
                                extra_min = int(total_sec/60)
                                total_min = extra_min + int(count_min) 
                                    
                                if total_min > 59:
                                    count_min = total_min % 60
                                    extra_hour = int(total_min/60)
                                    count_hour = extra_hour + int(count_hour) 
                                else:
                                    count_min = total_min
                            else:
                                count_sec = total_sec
                            
                            # Saving counting and detection data in text file
                            time_stamp = datetime.datetime(int(count_date[0:4]), int(count_date[5:6]), int(count_date[6:8]), int(count_hour), int(count_min), int(count_sec))
                            detection_data = '{},{},{}\n'.format(detected_class, direction, time_stamp) 
                            text_file_counting.write(detection_data)


                    # Cross-ratio and speed calculation at 10th frame after crossing the speed detection line; for left lane
                    if (indexIDs[i] in speed_id) and ((frameIndex - speed_id[indexIDs[i]][0])  == 10) and (speed_id[indexIDs[i]][1] == 'in') :
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        # Pre-defining the 3 points A,C,D
                        A = (192,478)
                        B = p0
                        C = (325,238)
                        D = (420,73)
                        
                        # Calculation of Euclidean distance between A,C,D
                        p_AC = math.sqrt((A[0] - C[0])**2 + (A[1] - C[1])**2)
                        p_AD = math.sqrt((A[0] - D[0])**2 + (A[1] - D[1])**2)
                        p_BD = math.sqrt((B[0] - D[0])**2 + (B[1] - D[1])**2)
                        p_BC = math.sqrt((B[0] - C[0])**2 + (B[1] - C[1])**2)
                        
                        c_ratio = (p_AC/ p_AD) * (p_BD/ p_BC) #Cross-ratio computation
                        # Real world distances in metres
                        dist_AC = 21.6
                        dist_AD = 84
                        dist_CD = 60

                        d = abs((dist_CD * dist_AC) / ((c_ratio * dist_AD) - dist_AC ))
                        travel_time = float(10/25) 
                        speed = (d/ travel_time) * (3.6)      #Final detected speed
                        
                        count_date = vid[0:8]
                        count_hour = vid[9:11]
                        count_min = vid[11:13]
                        count_sec = vid[13:15]
                        
                        if count_hour[0] == '0':
                            count_hour = int(count_hour[1])  
                        else:
                            count_hour = int(count_hour) 
                        
                        if count_min[0] == '0':
                            count_min = int(count_min[1])  
                        else:
                            count_min = int(count_min) 
                        
                        if count_sec[0] == '0':
                            total_sec = int((frameIndex+1)/25) + int(count_sec[1])
                        else:
                            total_sec = int((frameIndex+1)/25) + int(count_sec)
                        
                        if total_sec > 59:
                            count_sec = total_sec % 60
                            extra_min = int(total_sec/60)
                            total_min = extra_min + int(count_min) 
                                
                            if total_min > 59:
                                count_min = total_min % 60
                                extra_hour = int(total_min/60)
                                count_hour = extra_hour + int(count_hour) 
                            else:
                                count_min = total_min
                        else:
                            count_sec = total_sec
                        
                        # Saving speed and detection data in text file
                        detected_class = yolo.counter(p0, out_class, midPoint)
                        time_stamp = datetime.datetime(int(count_date[0:4]), int(count_date[5:6]), int(count_date[6:8]), int(count_hour), int(count_min), int(count_sec))
                        detection_data = '{},{},{},{}\n'.format(detected_class, speed_id[indexIDs[i]][1], int(speed), time_stamp) 
                        text_file_speed.write(detection_data)
                        
                        # Display speed on the frame
                        text = "{} Km/Hr".format(int(speed))
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        speed_id.pop(indexIDs[i])
                    
                    # Cross-ratio and speed calculation for right lane vehicles
                    if (indexIDs[i] in speed_id) and ((frameIndex - speed_id[indexIDs[i]][0])  == 10) and (speed_id[indexIDs[i]][1] == 'out'):
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        # Pre-defining the 3 points A,B,C
                        A = (345, 416)
                        B = (380, 263)
                        C = (391, 238)
                        D = p0
                        
                        # Calculation of Euclidean distance between A,B,C
                        p_AC = math.sqrt((A[0] - C[0])**2 + (A[1] - C[1])**2)
                        p_AD = math.sqrt((A[0] - D[0])**2 + (A[1] - D[1])**2)
                        p_BD = math.sqrt((B[0] - D[0])**2 + (B[1] - D[1])**2)
                        p_BC = math.sqrt((B[0] - C[0])**2 + (B[1] - C[1])**2)

                        c_ratio = (p_AC/ p_AD) * (p_BD/ p_BC) #Cross-ratio computation
                        # Real world distances in metres
                        dist_AC = 23.2
                        dist_BC = 4.3

                        d = abs(((1 - c_ratio) * (dist_BC) * dist_AC) / ((c_ratio * dist_BC) - dist_AC))
                        travel_time = float(10/25) 
                        speed = (d/ travel_time) * (3.6)     #Final detected speed
                        
                        count_date = vid[0:8]
                        count_hour = vid[9:11]
                        count_min = vid[11:13]
                        count_sec = vid[13:15]
                        
                        if count_hour[0] == '0':
                            count_hour = int(count_hour[1])  
                        else:
                            count_hour = int(count_hour) 
                        
                        if count_min[0] == '0':
                            count_min = int(count_min[1])  
                        else:
                            count_min = int(count_min) 
                        
                        if count_sec[0] == '0':
                            total_sec = int((frameIndex+1)/25) + int(count_sec[1])
                        else:
                            total_sec = int((frameIndex+1)/25) + int(count_sec)
                        
                        if total_sec > 59:
                            count_sec = total_sec % 60
                            extra_min = int(total_sec/60)
                            total_min = extra_min + int(count_min) 
                                
                            if total_min > 59:
                                count_min = total_min % 60
                                extra_hour = int(total_min/60)
                                count_hour = extra_hour + int(count_hour) 
                            else:
                                count_min = total_min
                        else:
                            count_sec = total_sec
                        
                        # Saving speed and detection data in text file
                        detected_class = yolo.counter(p0, out_class, midPoint)
                        time_stamp = datetime.datetime(int(count_date[0:4]), int(count_date[5:6]), int(count_date[6:8]), int(count_hour), int(count_min), int(count_sec))
                        detection_data = '{},{},{},{}\n'.format(detected_class, speed_id[indexIDs[i]][1], int(speed), time_stamp) 
                        text_file_speed.write(detection_data)
                        
                        # Display speed on the frame
                        text = "{} Km/Hr".format(int(speed))
                        cv2.putText(frame, text, (x , y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        speed_id.pop(indexIDs[i])

                    i += 1
                    
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            # Draw text on output frame
            draw.text((15,30), 'Incoming Traffic: \nCar %d \nBus %d \nTruck %d' %(car,bus,truck), fill=(255, 255, 255),font=font)
            draw.text((490,30), 'Outgoing Traffic: \nCar %d \nBus %d \nTruck %d' %(car2,bus2,truck2), fill=(255, 255, 255),font=font)
         
            frame = np.asarray(frame)
            # Draw lines
            cv2.line(frame, line1[0], line1[1], (0, 255, 255), 3)
            cv2.line(frame, line3[0], line3[1], (0, 255, 255), 3)
    
            if writer is None:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_fps = vs.get(cv2.CAP_PROP_FPS)
                vid_name = vid.split('.')[0] + '.avi'
                writer = cv2.VideoWriter('output/{}'.format(vid_name), fourcc, video_fps, (frame.shape[1], frame.shape[0]), True)
    
            # Save output frame
            writer.write(frame)
    
            # frameIndex increment
            frameIndex += 1
        
        writer.release()
        vs.release()

    print("Program End")
    

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    main(YOLO())
