# Obstacle-Detection-on-the-Moving-Video
Video is taken as the input and location coordinates along with the obstacle is being detected on the output video

## Table of Contents - 
* [About Project](#about-project)
* [About Working](#about-working)
* [About Me](#about-me)

## About Project
* This project is the computer vision part which is aiming for the Obstalce detection on the moving video. It takes the input as the any downloaded video and it gives the bounding box of the obstacle in front of the person either in front of him/her OR on the left OR on the Right. 
* When there is the space on the left then there is the indication to move Left, When there is the space on the right, then there is the indication that we can move to right, And when there is no space on Both of the side then to STOP!!

## About Working - 
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
  
3. TO BE CONTINUE..........:)
