import os, sys
from face_detector import DlibFaceDetector
import cv2
import json


class FaceAnalyzer(object):
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self._bbox_expand_ratio = 0.25

    @property
    def bbox_expand_ratio(self):
        return self._bbox_expand_ratio

    @bbox_expand_ratio.setter
    def bbox_expand_ratio(self, val):
        self._bbox_expand_ratio = val

    def full_analyze(self, input):
        self.root_dir = input + '_analyze'
        return self._full_analyze(self.root_dir, input)

    def _full_analyze(self, root_dir, input):
        res_faces = []
        res_landmarks = []

        if input.endswith(('.jpg', '.png')):
            return self._analyze_single_image('/'.join(root_dir.split('/')[:-1]), input)

        if os.path.isdir(input):
            self._create_dir(root_dir)
            self._img_cnt_in_dir = 0
            self.json_log = {}
            for sub_input in os.listdir(input):
                tmp_faces, tmp_landmarks = self._full_analyze(os.path.join(root_dir, sub_input),
                                                              os.path.join(input, sub_input))
                res_faces += tmp_faces
                res_landmarks += tmp_landmarks
                self.json_log['{}.jpg'.format(self._img_cnt_in_dir-1)] = tmp_landmarks
            with open(os.path.join(root_dir, 'facial_landmarks.json'), 'w') as f:
                json.dump(self.json_log, f)
            return res_faces, res_landmarks
        else:
            return [], []

    def _analyze_single_image(self, root_dir, image):
        res_faces = []
        res_landmarks = []
        image = cv2.imread(image)
        bboxes = self.face_detector.detect_faces(image)
        for bbox in bboxes:
            landmarks = self.face_detector.detect_facial_landmarks(image, bbox).parts()

            img_h, img_w, _ = image.shape
            expand_top, expand_left, expand_bot, expand_right = self._expand_bbox(bbox, img_h, img_w)
            cropped_face = image[expand_top:expand_bot, expand_left:expand_right, :]
            landmarks = [[landmarks[i].x - expand_left, landmarks[i].y - expand_top]
                         for i in range(len(landmarks))]
            res_faces.append(cropped_face)
            res_landmarks.append(landmarks)
            cv2.imwrite(os.path.join(root_dir, '{}.jpg'.format(self._img_cnt_in_dir)), cropped_face)
            self._img_cnt_in_dir += 1
        return res_faces, res_landmarks

    def _expand_bbox(self, bbox, img_h, img_w):
        h = bbox.bottom() - bbox.top()
        w = bbox.right() - bbox.left()
        return [int(max(0, bbox.top() - self._bbox_expand_ratio * h)),
                int(max(0, bbox.left() - self._bbox_expand_ratio * w)),
                int(min(img_h, bbox.bottom() + self._bbox_expand_ratio * h)),
                int(min(img_w, bbox.right() + self._bbox_expand_ratio * w))]

    def _create_dir(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)


if __name__ == '__main__':
    face_analyzer = FaceAnalyzer(DlibFaceDetector())
    faces, landmarks = face_analyzer.full_analyze('tmp0')
    print(len(faces))

    for i in range(len(faces)):
        for j in range(len(landmarks[i])):
            cv2.circle(faces[i], center=(landmarks[i][j][0], landmarks[i][j][1]), radius=2, color=(255, 0, 0),
                       thickness=-1)
            cv2.imshow('', faces[i])
            cv2.waitKey(0)
