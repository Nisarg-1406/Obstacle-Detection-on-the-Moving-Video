# Obstacle-Detection-on-the-Moving-Video
Video is taken as the input and location coordinates along with the obstacle is being detected on the output video

## Table of Contents - 
* [About Project](#about-project)
* [Detailed Explanation about Project](#detailed-explanation-about-project)
* [Input Video Link](#input-video-link)
* [Output Video Link](#output-video-link)
* [About Me](#about-me)

## About Project
* This project is the computer vision part which is aiming for the Obstalce detection on the moving video. It takes the input as the any downloaded video and it gives the bounding box of the obstacle in front of the person either in front of him/her OR on the left OR on the Right. 
* When there is the space on the left then there is the indication to move Left, When there is the space on the right, then there is the indication that we can move to right, And when there is no space on Both of the side then to STOP!!

## Detailed Explanation about Project
1. First to import necessary libraries (It is being given in the code provided). 
2. Then we would be taking input as the video file 
    ```
    from google.colab import files
    def upload_files():
      from google.colab import files
      uploaded = files.upload()
      for k, v in uploaded.items():
        open(k, 'wb').write(v)
      return list(uploaded.keys())
    upload_files()
    ```
    **V. IMP Concept**- 
    * What this means is that with the help of Google Colab we would be first importing the necessary files which will help in opening and storing out the image with colab and then we would be defining the function named upload_files, and in the for loop we would be doing write in binary form (therefore 'wb' is done). So in general we would be taking the  key k and would be WRITING the value v in the binary form (in terms of 0s and 1s), therefore ```open(k, 'wb').write(v)``` is done at each and evey step and at last we would be returning list that would be storing out the key for the particular value. So in this form the input video is stored in the google colab for the further preprocessing.  
  
3. We would be using ```ssd_inception_v2_coco_2017_11_17``` as the model i.e **single shot multibox detector**. We would have faster RCNN, but Faster RCNN is accurate but its slow in compared to Single shot mulitbox detector. Then the model is stored as ``'.tar.gz'`` extension so that file is compressed and hence the file can be easily processed in ZIP format. Here we would be downloading the base tensorflow model for the further preprocessing. 

4. Then we would be saving all the required things, i.e weight and graph in a single file for the easy acess. This process is known as freezing ```PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'``` and this is the actual model that is used for the object detection. We would be defining the number of classes as 10. Also List of the strings that is used to add correct label for each box - ```PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')```.

5. Then we would be creating the detection graph for each and every location. A Graph instance supports an arbitrary number of "collections" that are identified by name. For convenience when building a large graph, collections can store groups of related objects. With the File we would be **reading in the binary form**.
    ```
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    ```
Finally we would be defining the **categories and the categories index** into consideration for object detection. 

6. Define input and output tensors (i.e. data) for the object detection classifier - Input tensor is the image and Output tensors are the detection boxes, scores, and classes. Each box represents a part of the image where a particular object was detected. Each score represents level of confidence for each of the objects. The score is shown on the result image, together with the class label. And at last we would be defining the number of objects detected. 
    ```
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') ----------------------> Input Image
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') ----------------> Detection Box
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')---------------> Detection Scores
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') ------------> Detection Class
    num_detections = detection_graph.get_tensor_by_name('num_detections:0') ------------------> Number of objects detected.
    ```

7. `img.shape` - It provides you the shape of img in all directions. ie number of rows, number of columns for a 2D array (grayscale image). For 3D array, it gives you number of channels also. So if `len(img.shape)` gives you two, it has a single channel (Gray Scale Image). If `len(img.shape)` gives you three, third element gives you number of channels (Color image). The channel is made up of one of the primary colors i.e - RED, GREEN, BLUE. 
    ```
    def region_of_interest(img, vertices):
    #Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black.
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:  --------------------------------------------------------------------------------> #This is for the Colored image
        channel_count = img.shape[2]  # i.e. 3 - RGB
        ignore_mask_color = (255,) * channel_count
    else: --------------------------------------------------------------------------------------------------> # For Grayscale Image. 
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask) # Doing Bitwise_and 
    return masked_image
    ```
    
 8. We would be outputting the video using `cv2.VideoWriter` function. The parameter while defining the output is - `frame_width, frame_height, rows, cols, left_boundary, left_boundary_top, right_boundary, right_boundary_top, bottom_left, top_left, bottom_right, top_right, vertices`. Then we would be forming line on the image with the syntax - `cv2.line(image, start_point, end_point, color, thickness)` 
     ```
      cv2.line(frame,tuple(bottom_left),tuple(bottom_right), (255, 0, 0), 5)
      cv2.line(frame,tuple(bottom_right),tuple(top_right), (255, 0, 0), 5)
      cv2.line(frame,tuple(top_left),tuple(bottom_left), (255, 0, 0), 5)
      cv2.line(frame,tuple(top_left),tuple(top_right), (255, 0, 0), 5)
     ```
     
 9. We would be defining the 4 parameters - `boxes, scores, classes, num` by ``(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: frame_expanded})``. We would be using np.squeeze - numpy.squeeze() function is used when we want to remove single-dimensional entries from the shape of an array in this line - `vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.78)`
 
 10. Then we would be finding the `ymin, ymax, xmin, xmax` that would be dimension of the bounding boxes in 3 dimension and in Y Coordinate we would be multiplying it with **frame width** and in X coordinate with **Frame Height** and all this value are in integer format.
    `ymin = int((boxes[0][0][0]*frame_width))
    xmin = int((boxes[0][0][1]*frame_height))
    ymax = int((boxes[0][0][2]*frame_width))
    xmax = int((boxes[0][0][3]*frame_height))
    Result = np.array(frame[ymin:ymax,xmin:xmax])`
    
11. Now to put Text around the bounding boxes - **cv2.putText() method is used to draw a text string on any image** -> Syntax - `cv2.putText(image, 'TEXT', org, font,  fontScale, color, thickness)` 
    ```
    ymin_str='y min  = %.2f '%(ymin)
    ymax_str='y max  = %.2f '%(ymax)
    xmin_str='x min  = %.2f '%(xmin)
    xmax_str='x max  = %.2f '%(xmax)
    cv2.putText(frame,ymin_str, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2) -----> ymin
    cv2.putText(frame,ymax_str, (50, 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2) -----> ymax
    cv2.putText(frame,xmin_str, (50, 90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2) -----> xmin
    cv2.putText(frame,xmax_str, (50, 110),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2) ----> xmax
    ```
    
12. Next is to simply print the output.
    ```
    if scores.max() > 0.78:
         print("inif")
    if(xmin >= left_boundary[0]):
      print("move LEFT - 1st !!!")
      cv2.putText(frame,'Move LEFT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
    elif(xmax <= right_boundary[0]):
      print("move Right - 2nd !!!")
      cv2.putText(frame,'Move RIGHT!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
    elif(xmin <= left_boundary[0] and xmax >= right_boundary[0]):
      print("STOPPPPPP !!!! - 3nd !!!")
      cv2.putText(frame,' STOPPPPPP!!!', (300, 100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
    cv2.line(frame,tuple(left_boundary),tuple(left_boundary_top), (255, 0, 0), 5)
    cv2.line(frame,tuple(right_boundary),tuple(right_boundary_top), (255, 0, 0), 5)
    out.write(frame)
    ```
    
## Input Video Link
The Link for Input Video is : https://drive.google.com/file/d/15pV7W33PFtivJ6zNn4vwvPgWeANGvNUM/view?usp=sharing

## Output Video Link
The Link for Output Video is : https://drive.google.com/file/d/14ci-lsFPUf1n6dNA8k9cYXKJGG-pWOd9/view?usp=sharing
 
## About Me
**IF YOU LIKED MY WORK, PLEASE HIT THE STAR BUTTON, AND IF POSSIBLE DO PLEASE SHARE, SO THAT COMMUNITY CAN GET BENIFIT OUT OF IT BEACUSE I AM EXLPANING EACH AND EVERY LINE OF CODE FOR EACH AND EVERY PROJECT OF MINE.**

Also I am Solving **Algorithms and Data Structure Problems from more than 230 Days (More than 32 Weeks) Without any off-Day and have solved more than 405 Questions on various topics and posting my solutions on Github Daily**. You can Visit my Profile of LeetCode here - **https://leetcode.com/Nisarg1406/**

I am good at Algorithms and Data structure and I have good Projects in Machine learning and Deep Learning (Computer Vision). **I am and would be posting the detialed explantion of each and every project working**. I am activily looking for an Internhip in **Software development enginering (SDE) Domain and Machine learning Domain**.

You can contact me on my mail ID - nisarg.mehta18@vit.edu OR nisargmehta2000@gmail.com and even Contact me on LinkedIn - https://www.linkedin.com/in/nisarg-mehta-4a378a185/
