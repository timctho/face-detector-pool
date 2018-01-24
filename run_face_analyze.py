from face_analyzer import FaceAnalyzer

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, default='dlib')
parser.add_argument('--input', type=str, default='1.jpg')
parser.add_argument('--viz', type=bool, default=True)

args = parser.parse_args()

if __name__ == '__main__':

    if args.detector == 'dlib':
        from face_detector import DlibFaceDetector as detector

    analyzer = FaceAnalyzer(detector())
    faces, landmarks = analyzer.full_analyze(args.input)

    if args.viz:
        for face, landmark in zip(faces, landmarks):
            for idx in range(len(landmark)):
                cv2.circle(face, (landmark[idx][0], landmark[idx][1]), radius=2, color=(255,0,0), thickness=-1)
            cv2.imshow('', face)
            cv2.waitKey(0)
