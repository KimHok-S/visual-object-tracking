import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u = np.array([[u_x], [u_y]])
        self.xk = np.array([[0], [0], [0], [0]])
        self.xk_ = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.array([[(dt ** 2) / 2, 0],
                            [0, (dt ** 2) / 2],
                            [dt, 0],
                            [0, dt]])
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.array([[(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                            [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                            [(dt ** 3) / 2, 0, dt ** 2, 0],
                            [0, (dt ** 3) / 2, 0, dt ** 2]]) * std_acc ** 2
        self.R = np.array([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.xk_ = np.dot(self.A, self.xk) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, zk):
        Sk = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        Kk = np.dot(np.dot(self.P, self.H.T), inv(Sk))

        self.xk = self.xk_ + np.dot(Kk, zk - np.dot(self.H, self.xk_))
        self.P = np.dot(np.eye(self.A.shape[1]) - np.dot(Kk, self.H), self.P)