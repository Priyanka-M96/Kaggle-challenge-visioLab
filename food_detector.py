# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python food_detector.py --video=".\videos\20200313-200924.avi"
#                 python food_detector.py --testpath=".\test\images"

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
import imutils
import glob
import pandas as pd

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

parser = argparse.ArgumentParser(description='Food Detection using YOLOv3')
parser.add_argument('--testpath', help='Path to test path.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "classes.txt"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3_custom_train.cfg"
modelWeights = "yolov3_custom_train_final.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (0, 0, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, frames_processed, flag, title):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    center_x = ""
    center_y = ""
    width = ""
    height = ""
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    if flag == 1 and frames_processed%5 == 0:
        dirName = args.video[:-4] + 'box_predictions'
        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        except FileExistsError:
            if frames_processed == 5:
                print("Directory ", dirName, " already exists")

        filename = str(dirName + "\\" + str(frames_processed) + ".txt")
        myfile = open(filename, 'w')
        class_predicted = ""
        file_string = ""

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        p1 = int((left + left + width) / 2.0)
        p2 = int((top + top + height) / 2.0)
        # left, top, right, bottom
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        if flag == 1 and frames_processed % 5 == 0:
            class_predicted = str(classIds[i])
            file_string = class_predicted + "," + str(float(p1 / frameWidth)) + "," + str(
                float(p2 / frameHeight)) + "," + str(
                float((width / frameHeight))) + "," + str(float(height / frameHeight))
            myfile.write(file_string)
            if len(indices) > 1:
                myfile.write("\n")
        elif flag == 0:
            id_cl = classIds[i]
            row = [title[4:], id_cl]
            return row

    if flag == 1 and frames_processed % 5 == 0:
        myfile.close()


# Process inputs
winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
image_list = []
i = 0
title = []
row_list = []
end_video = 0
total = 0
if args.testpath:
    for name in glob.iglob(os.path.join(args.testpath, "*.jpg")):
        title_each, ext = os.path.splitext(os.path.basename(name))
        title.append(title_each)
        image = args.testpath + "/" + title_each + '.jpg'
        image_list.append(image)

while True:
    if (args.testpath):
        # Open the image file
        if i < len(image_list):
            image = image_list[i]
            if not os.path.isfile(image):
                print("Input image file ", image, " doesn't exist")
                sys.exit(1)
            cap = cv.VideoCapture(image)
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4] + '_yolo_out_py.avi'

        try:
            prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv.CAP_PROP_FRAME_COUNT
            total = int(cap.get(prop))
            print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

    # Get the video writer initialized to save the output video
    if not args.testpath:
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    frames_processed = 0
    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            if args.video:
                print("Output file is stored as ", outputFile)
                end_video = 1
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        start = time.time()
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        end = time.time()

        frames_processed += 1
        # Remove the bounding boxes with low confidence
        if args.testpath:
            flag = 0
            row = postprocess(frame, outs, frames_processed, flag, title[i])
            row_list.append(row)
        elif (args.video):
            flag = 1
            postprocess(frame, outs, frames_processed, flag, None)

        # print(frames_processed)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # some information on processing single frame
        if (args.video):
            if total > 0:
                elap = (end - start)
                print("Frame {} is processed out of {} " .format(frames_processed,total) )
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # Write the frame with the detection boxes
        if (args.testpath):
            # cv.imwrite(outputFile, frame.astype(np.uint8))
            i = i + 1
            print("Processing image:", title[i - 1])
            if i == len(image_list):
                df = pd.DataFrame(row_list)
                df.columns = ["Id", "Category"]
                df.to_csv(r'results_food_detector.csv', index=False, header=True)
                df = pd.read_csv('results_food_detector.csv', header=0)
                df.columns = ["Id", "Category"]
                df.sort_values(by="Id", ascending=True, inplace=True)
                df.to_csv(r'results_food_detector.csv', index=False, header=True)
                break

        else:
            vid_writer.write(frame.astype(np.uint8))

        # cv.imshow(winName, frame)

    if (i >= len(image_list) and args.testpath) or (end_video == 1 and args.video) :
        break
