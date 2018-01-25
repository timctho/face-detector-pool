import dlib
from face_detector_pool.base_face_detector import BaseFaceDetector

class DlibFaceDetector(BaseFaceDetector):
    def __init__(self, facial_model_path='model_weights/shape_predictor_68_face_landmarks.dat'):
        self.face_model = dlib.get_frontal_face_detector()
        self.facial_landmark_model = dlib.shape_predictor(facial_model_path)
        self._threshold = 0.4


    @property
    def name(self):
        return 'dlib'


    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        self._threshold = val

    def create_window(self):
        self.window = dlib.image_window()

    def draw_bbox(self, image, bbox):
        try:
            self.window.set_image(image)
            self.window.add_overlay(bbox)
        except:
            print('Can\'t draw bbox on image.')

    def draw_landmark(self, image, landmark):
        try:
            self.window.set_image(image)
            self.window.add_overlay(landmark)
        except:
            print('Can\'t draw landmarks on image.')

    def detect_faces(self, image, scale=1):
        bboxes, scores, _ = self.face_model.run(image, scale)
        bboxes = [bboxes[i] for i in range(len(bboxes)) if scores[i] > self.threshold]
        return bboxes

    def detect_facial_landmarks(self, image, bbox):
        return self.facial_landmark_model(image, bbox).parts()




