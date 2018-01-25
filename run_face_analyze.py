from face_analyzer import FaceAnalyzer

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, default='mtcnn')
parser.add_argument('--input', type=str, default='Yuniko/yuniko_0.jpg')
parser.add_argument('--viz', type=bool, default=True)

args = parser.parse_args()

if __name__ == '__main__':

    if args.detector == 'dlib':
        from face_detector_pool.dlib_face import DlibFaceDetector as detector
    elif args.detector == 'mtcnn':
        from face_detector_pool.mtcnn_face import MTCNNFaceDetector as detector


    analyzer = FaceAnalyzer(detector())
    faces, landmarks = analyzer.full_analyze(args.input)

    if args.viz:
        for face, landmark in zip(faces, landmarks):
            for idx in range(len(landmark)):
                cv2.circle(face, (int(landmark[idx][0]),
                                  int(landmark[idx][1])), radius=2, color=(0,255,0), thickness=-1)
            cv2.imshow('', face)
            cv2.waitKey(0)
