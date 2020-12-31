# Obstacle-Detection-on-the-Moving-Video
Video is taken as the input and location coordinates along with the obstacle is being detected on the output video

## Table of Contents - 
* [About Project](#about-project)
* [About Working](#about-working)
* [About Me](#about-me)

## About Project
* This project is the computer vision part which is aiming for the Obstalce detection on the moving video. It takes the input as the any downloaded video and it gives the bounding box of the obstacle in front of the person either in front of him/her OR on the left OR on the Right. 
* When there is the space on the left then there is the indication to move Left, When there is the space on the right, then there is the indication that we can move to right, And when there is no space on Both of the side then to STOP!!

## About Working
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
    
 8. TO BE CONTINUE...............:)
