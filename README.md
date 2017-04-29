# Real Time Face Detection using Viola-Jones and CAMSHIFT in Python

This repository mainly deals about real time Face Detection on a HD video (Last Week Tonight with John Oliver) using combined approach of Viola-Jones and CAMSHIFT. The prerequisites are a brief understanding about Viola-Jones face detection model using Haar features and CAMSHIFT algorithm for tracking object along with a fair amount of patience. If you are not interested in any explanation then here is the link to the [code](https://github.com/rahulptel/realtime-facedetection-hd-videos/tree/master/VJCMS.rar).

Starter codes for face detection in OpenCV using Haar features, along with a crisp background on them, are available at following links.<br>

[Face detection using Viola-Jones – Link 1](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)<br>
[Face detection using Viola-Jones – Link 2](http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)<br><br>

Starter codes for CAMSHIFT, along with a crisp explanation about it, are available at the following links.
<br><br>
[Track objects using CAMSHIFT – Link 1](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html)<br>
[Track objects using CAMSHIFT – Link 2](http://www.computervisiononline.com/blog/tutorial-using-camshift-track-objects-video)

# Part I

The task at hand is to detect multiple faces in a given video in real time. The naive approach would be to detect faces using Viola-Jones in each frame. The benefit of this approach is, we will be able to detect multiple faces in each frame with high accuracy. The drawback of this approach would be speed. Detecting faces using Viola-Jones is computationally expensive and it significantly reduces the speed of detecting faces.

On the other hand we have CAMSHIFT. CAMSHIFT is a tracking algorithm which tracks a particular objects whose coordinates needs to be provided once. The benefit of CAMSHIFT is its speed. The drawback of this approach is, once the object which is being tracked goes out of the frame, CAMSHIFT produces erroneous results i.e. it converges randomly to different object. Also if the foreground and background color are not well separated in the color space, then also it might fail.

If we try to focus on the benefits of both of these algorithms written in italics above, then it might click to few of the smarter ones that, that’s exactly how we want our desired algorithm to be! We want to detect multiple faces with high accuracy at great speed! I know you are very much tempted to try this out and check it’s validity after reading out this line. Hold your horses, we together will go through it.

But before we use the code available for CAMSHIFT at the above links, we need to equip it for tracking multiple faces. Viola-Jones will return the coordinates of the faces found in a given frame. But the current codes available for CAMSHIFT tracking can only track a single object. Hence first of all I wrote a code which can track multiple objects. After succeeding in doing so, I later decided to club them. Below mentioned is the flowchart for the proposed algorithm.

For a given video, first of all we try to read a frame and see if we are successful doing so. If we are, then we do further processing. This condition is checked right after the input. If the frame is read properly we then decide which function to call i.e. Viola-Jones or CAMSHIFT. But, as this is the first pass, the algorithm will learn from the counter i that we should apply Viola-Jones on the current frame as no faces are yet detected which can be tracked by CAMSHIFT.

Hence we re-size the frame by a certain RATIO (positive integer) and convert it to a gray-scale image. Later we apply Viola-Jones Haar features trained Cascade Classifier(consider it as a black box which detects faces for you!) to detect faces in the frame. If faces are found, we increment the counter i which signifies that we are successful in finding faces in the current frame, now lets track these faces in the subsequent frames. If no faces are found, then we just read subsequent SKIP (positive integer) number of frames and display them, as it is without any processing. We know that for high fps (frames per second) video it’s very unlikely to find the faces in the immediate next frame. And even if there are chances, the time quantum is too small for us to detect that a new face is going undetected. This heuristic helps us to improve the time performance which is very crucial in real time.

Later we try to track the face in the subsequent TRACK number of frames. The next frame in which you want to track the faces is converted to HSV color frame. The histogram of the pixels, lying inside the bounding box around faces, found in the previous frame by Viola-Jones is calculated. Based upon this histogram we try to find the probability of finding the face in the current frame and track them using CAMSHIFT.

# Part II

# Part III

