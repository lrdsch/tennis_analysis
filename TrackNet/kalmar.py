import numpy as np
import cv2

class KalmanFilter2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # [x, y, dx, dy]
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

    def predict(self):
        prediction = self.kalman.predict()
        return prediction[0, 0], prediction[1, 0]
