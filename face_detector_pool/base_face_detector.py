import abc


class BaseFaceDetector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def detect_faces(self, image):
        """
        Detect faces of an single image
        """
        return NotImplementedError

    @abc.abstractmethod
    def detect_facial_landmarks(self, image):
        """
        Detect facial landmarks of an single image
        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        return NotImplementedError