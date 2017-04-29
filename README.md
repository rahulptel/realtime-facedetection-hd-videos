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

First of all we would like to look at the necessary imports and few global variables initialization which will be used throughout the code. The first import is of the class which contains trained Cascade Classifier to detect frontal faces using Haar features. Next we import the OpenCV library which contains functions for reading video file, CAMSHIFT, back projection, etc and numpy packages for ease of computations.

The use of global variables like RATIO, TRACK, SKIP, cap, termination, I suppose should be clear to you based on the comments. Note that cap is like a handle to access the video.

After importing the packages, let’s understand the part where we try to detect the faces using Viola-Jones face detection model. Here we will use the Cascade Classifier for detecting frontol faces trained using Haar features to detect faces. I know it sounds daunting but OpenCV makes our life simpler. We already have this Classifier ready made for us. It’s waiting for people like us who can use it. Click on the image below to see it more clearly if not visible properly!

Here, we try to make a copy of the original frame we read. We then scale this frame based on the RATIO we provided earlier at the start of the code. This scaled down frame is then converted to gray-scale and then Cascade Classifier is applied on to it to detect faces. Applying Cascade Classifier on this scaled down gray-scale frame makes detection of faces very quick as compared to detecting faces on the original frame. For each face found in this modified frame, we have the value of its top left corner and the width and height for the same in a list name faceRects[]. These values of the coordinates and the width and height are w.r.t the modified frame. Hence they need to be scaled up so that we are able to map it with the face in the original frame. We append the scaled up coordinates and width and height in the list allRoiPts[]. After appending all these points in the list, we just show the current frame and then return the list allRoiPts[]. As simple as that!

I know you might have started wondering that why did I subtract 10 and 15 from the top left pixel values and width and height respectively. It should ideally be New_Pixel = RATIO x Old_Pixel, where New_Pixel is the scaled up pixel value corresponding to the Old_Pixel value and New_Width = RATIO x Old_Width where New_Width is the scaled up width corresponding to the Old_Width. For now, try to understand the reason from the comments. If you are not able to do so then just take it as it is for now. I will make sure that I justify this thing at the end of the tutorial series.

After mapping the pixel values of the top left corner and width and height of the modified frame to the original we need to find the histogram of the pixels lying inside those regions in the original frame. The main reason for doing so is these histograms will be used in the next step for back projection to track the faces. To describe back projection in one line, it is the process of generating the 2D probability matrix of the dimension same as the image/frame on which it is applied where each cell/pixel denotes the probability of a pixel of the desired frame lying inside the window region of some other image/frame. In our case, we generate a 2D matrix where each pixel shows the probability of lying inside the face window region found by Viola-Jones in a previous frame in the current frame. I know it’s heavy, read it twice or thrice you will get it. If not then revisit the links for CAMSHIFT in tutorial I. These histogram are normalized to counter the effects of various illumination conditions and are returned in the list allRoiHist[].

We found faces and their histograms! What next!

If you are thinking about tracking them then you really paying attention, if not then I must improve my explanation skill. In anycase you know now what to do. Below defined is the trackFace() function which helps us in tracking the faces in the subsequent TRACK  number of frames.

First of all we try to convert the current frame into HSV color frame and then try to back project the histogram of faces found in the previous frame on the current frame. As a result of this we will have a probabilistic matix which will help us to find out the region having maximum probability of finding the face detected by VJFindFace() in the previous frame in the current frame closest to the window in the previous frame. This we do for all the faces found by iterating over them and finding their new location in the current frame. This new locations will be used as a reference to track faces in the next frames and the process will continue until TRACK number of iterations. CAMSHIFT here will be useful for finding the new location of the face based on the results of back projection. CAMSHIFT also dynamically resizes the window along with tracking. A good explanation of working of CAMSHIFT as always is available at the link mentioned in tutorial I.

At last, we have one scenario left, the one in which no face is found in the current frame. In such cases we just try to skip SKIP number of frames. I know you don’t need an explanation for this part!

# Part III

