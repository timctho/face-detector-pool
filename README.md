# Face Detector Pool
* Implement common face detection algorithms.
<p align="center">
    <img src="https://github.com/timctho/dlib-face-detector/raw/master/twice.jpg", width="480">
</p>

<p align="center">
    <img src="https://github.com/timctho/dlib-face-detector/raw/master/sample.jpg", width="320">
    <img src="https://github.com/timctho/dlib-face-detector/raw/master/Yuniko/yuniko_0.jpg", width="240">
</p>

## Requirements
 - Python 3.6
 - OpenCV 3
 - Dlib
 - Tensorflow 1.4
 
## Current methods
 - Dlib
 - MTCNN

## Usage
* Use scripts

 `python run_face_analyze.py --detector <method> --input <image file or folder> --viz <visualize or not>`

 For example, `python run_face_analyze.py --detector dlib --input Yuniko --viz True`

* Use high level class

  - Create **FaceAnalyzer()** instance with a detector.
 
  - Run **full_analyze()** will return detected **faces** and related **facial landmarks**.
 
* Use detector api
  - **detect_faces()** return face bounding boxes.
  - **detect_facial_landmarks()** return facial landmarks.
 

## Outputs
* Detected face will be cropped and saved into `output` folder.
* Related landmarks will be saved into `face_metadata.json` with `<key, value> = <img_path, landmarks>`.
