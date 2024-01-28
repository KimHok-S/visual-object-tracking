from KalmanFilter import KalmanFilter
from Detector import detect
import cv2


def main():
    dt = 0.1
    u_x, u_y, std_acc = 1, 1, 1
    x_std_meas = 0.1
    y_std_meas = 0.1
    karlman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    cap = cv2.VideoCapture('../data/randomball.avi')
    centers = []
    prev_predicted = []
    prev_center = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            centers = detect(frame)
            for center in centers:
                karlman_filter.predict()
                predicted = karlman_filter.xk_
                karlman_filter.update(center)
                # Centre de la boule en vert
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 255, 0), -1)
                # Rectangle bleu pour la prédiction de la bounding box
                cv2.rectangle(frame, (int(predicted[0]) - 15, int(predicted[1]) - 15), (int(predicted[0]) + 15,
                              int(predicted[1]) + 15), (255, 0, 0), 2)
                # Rectangle rouge pour l'estimation de la bounding box
                cv2.rectangle(frame, (int(karlman_filter.xk[0]) - 15, int(karlman_filter.xk[1]) - 15),
                              (int(karlman_filter.xk[0]) + 15, int(karlman_filter.xk[1]) + 15), (0, 0, 255), 2)
                # Vraie trajectoire du centre de la boule en vert
                if len(prev_center) != 0:
                    for i in range(1, len(prev_center)):
                        cv2.line(frame, (int(prev_center[i-1][0]), int(prev_center[i-1][1])),
                                 (int(prev_center[i][0]), int(prev_center[i][1])), (0, 255, 0), 2)
                prev_center.append(center)
                # Trajectoire prédite du centre de la boule en rouge
                if len(prev_predicted) != 0:
                    for i in range(2, len(prev_predicted)):
                        cv2.line(frame, (int(prev_predicted[i-1][0]), int(prev_predicted[i-1][1])),
                                 (int(prev_predicted[i][0]), int(prev_predicted[i][1])), (0, 0, 255), 2)
                prev_predicted.append(predicted)
                cv2.imshow('frame', frame)
                cv2.waitKey(50)
        else:
            break


if __name__ == '__main__':
    main()
