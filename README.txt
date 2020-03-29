This is a competition held by VisioLab Gmbh on Kaggle for the internship position "Deep Learning Intern". This task is based on Computer Vision Application of food detection in canteens of several universities. 

This folder is in the format of 

./test
  -> images
        -> img0.jpg...img49.jpg
./train
  -> images
        -> img50.jpg...img282.jpg
  -> labels
	-> img50.txt...img282.txt
./videos
   -> xxx.avi

The train dataset is used for training the darknet YOLOv3 which provided us with the weights "yolov3_custom_train_final.weights" and config file "yolov3_custom_train.cfg". These two files are used for further prediction and detection of objects in test data and videos.

Two ways to run detector.py
1)  python food_detector.py --testpath=".\test\images"

This reads all the images in test dataset and creates a csv file "results_food_detector.csv" which lists the image and the category it belongs.

2) python food_detector.py --video=".\videos\20200313-200924.avi"

This reads the video frames and writes a text file with its annotation in the file were the video is located. The annotation are created as in YOLO txt format and it saves after every 5 frames. After reading the complete video, a output video is also created in the same folder with respective detection.


Used reference of
- https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
- https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/
- https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
