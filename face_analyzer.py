import os
import cv2
import json


class FaceAnalyzer(object):
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self._bbox_expand_ratio = 0.0

    @property
    def bbox_expand_ratio(self):
        return self._bbox_expand_ratio

    @bbox_expand_ratio.setter
    def bbox_expand_ratio(self, val):
        self._bbox_expand_ratio = val

    def full_analyze(self, input):
        if os.path.isdir(input):
            self.input_root = input
        else:
            self.input_root = './'
        output_root = 'output/'
        self.res_faces = []
        self.res_landmarks = []
        self.json_log = {}
        self._img_cnt_in_dir = 0
        self._full_analyze(self.input_root, input, output_root)
        with open('face_metadata.json', 'w') as f:
            json.dump(self.json_log, f)
        return self.res_faces, self.res_landmarks

    def _full_analyze(self, root_dir, input, output_root):

        if os.path.isdir(root_dir):
            self._create_dir(output_root)
        if input.endswith(('.jpg', '.png')):
            self._analyze_single_image(
                input,
                '/'.join(output_root.split('/')[:-1]))
            return
        if os.path.isdir(input):
            for sub_input in os.listdir(input):
                self._full_analyze(os.path.join(root_dir, sub_input),
                                   os.path.join(input, sub_input),
                                   os.path.join(output_root, sub_input))
        return

    def _analyze_single_image(self, image_path, output_root):
        image = cv2.imread(image_path)
        bboxes = self.face_detector.detect_faces(image)
        print('Detect {} faces in [{}]'.format(len(bboxes), image_path))
        print(bboxes)

        if self.face_detector.name == 'dlib':
            for bbox in bboxes:
                landmarks = self.face_detector.detect_facial_landmarks(image, bbox)
                img_h, img_w, _ = image.shape
                expand_top, expand_left, expand_bot, expand_right = self._expand_bbox(bbox, img_h, img_w)
                cropped_face = image[expand_top:expand_bot, expand_left:expand_right, :]
                landmarks = [[landmarks[i].x - expand_left, landmarks[i].y - expand_top]
                             for i in range(len(landmarks))]
                img_path = os.path.join(output_root, '{}.jpg'.format(self._img_cnt_in_dir))
                self.res_faces.append(cropped_face)
                self.res_landmarks.append(landmarks)
                self.json_log[img_path] = landmarks
                cv2.imwrite(img_path, cropped_face)
                self._img_cnt_in_dir += 1

        elif self.face_detector.name == 'mtcnn':
            landmarks = self.face_detector.points
            for i in range(0, len(bboxes)):
                cropped_face = image[int(bboxes[i][1]):int(bboxes[i][3]),
                               int(bboxes[i][0]):int(bboxes[i][2]), :]
                _landmark = landmarks[:, i]
                landmark = [[int(_landmark[2*j])-bboxes[i][0], int(_landmark[2*j+1])-bboxes[i][1]] for j in range(5)]
                img_path = os.path.join(output_root, '{}.jpg'.format(self._img_cnt_in_dir))
                self.res_faces.append(cropped_face)
                self.res_landmarks.append(landmark)
                self.json_log[img_path] = landmark
                cv2.imwrite(img_path, cropped_face)
                self._img_cnt_in_dir += 1




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
