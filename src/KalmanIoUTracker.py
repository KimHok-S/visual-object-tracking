import sys
import os
import numpy as np
import pandas as pd
import cv2
from KalmanFilter import KalmanFilter
from IoUTracker import iou, compute_similarity_matrix, hungarian_assignment, update_tracks
from scipy.optimize import linear_sum_assignment


def compute_centroid(bbox):
    return bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2)


# convert centroid and bounding box dimensions into a bounding box
def centroid_to_box(centroid, width, height):
    return (centroid[0] - (width / 2), centroid[1] - (height / 2)), width, height


# convert centroids into a dataframe with bounding boxes informations
def original_to_frame(original_boxes, centroids):
    frame = pd.DataFrame(columns=['bb_left', 'bb_top', 'bb_width', 'bb_height'])
    for id in original_boxes:
        bbox = centroid_to_box(centroids[id], original_boxes[id][0], original_boxes[id][1])
        frame = frame._append({'bb_left': bbox[0][0],
                              'bb_top': bbox[0][1],
                              'bb_width': bbox[1],
                              'bb_height': bbox[2]}, ignore_index=True)
    return frame


def setup_dict_id(kalman_boxes, original_boxes, centroids, kalman_filters, frame_nb):
    original_boxes[frame_nb] = {}
    centroids[frame_nb] = {}
    kalman_filters[frame_nb] = {}


def main():
    # if the results file already exists, delete it
    try:
        os.remove('../results/results_Kalman.txt')
    except OSError:
        pass

    det = pd.read_csv('../data/det/det.txt', sep=',', index_col=0)
    frames = det.index.unique()
    sigma_iou = 0.2

    # centroids       =  {numero de la frame: {id du tracker: [x, y]}}
    # original_boxes  =  {numero de la frame: {id du tracker: [w, h]}}
    # kalman_filters  =  {numero de la frame: {id du tracker: KalmanFilter}}
    kalman_filters = {}
    original_boxes = {}
    centroids = {}
    setup_dict_id(kalman_filters, original_boxes, centroids, kalman_filters, 1)

    dt = 0.1
    u_x, u_y, std_acc = 1, 1, 1
    x_std_meas = 0.1
    y_std_meas = 0.1

    # init first frame
    tracks = update_tracks(0, [], len(det.loc[1]))
    for i in range(1, len(tracks) + 1):
        kalman_filters[1][i] = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        kalman_filters[1][i].predict()
        original_boxes[1][i] = (det.loc[1].iloc[i - 1]['bb_width'], det.loc[1].iloc[i - 1]['bb_height'])
        centroids[1][i] = compute_centroid((det.loc[1].iloc[i - 1]['bb_left'], det.loc[1].iloc[i - 1]['bb_top'],
                                            det.loc[1].iloc[i - 1]['bb_width'], det.loc[1].iloc[i - 1]['bb_height']))
        kalman_filters[1][i].update(centroids[1][i])

    # main loop
    for i in range(2, len(frames)):
        frame1 = original_to_frame(original_boxes[i - 1], centroids[i - 1])
        frame2 = det.loc[i]
        similarity_matrix = compute_similarity_matrix(frame1, frame2)
        assignments = hungarian_assignment(similarity_matrix, tracks, sigma_iou)
        tracks = update_tracks(max(tracks), assignments, len(frame2))

        setup_dict_id(kalman_filters, original_boxes, centroids, kalman_filters, i)
        for j in range(len(tracks)):
            if tracks[j] in kalman_filters[i - 1].keys():
                kalman_filters[i][tracks[j]] = kalman_filters[i - 1][tracks[j]]
                kalman_filters[i][tracks[j]].predict()
                original_boxes[i][tracks[j]] = (frame2.iloc[j]['bb_width'], frame2.iloc[j]['bb_height'])
                centroids[i][tracks[j]] = (kalman_filters[i][tracks[j]].xk[0][0], kalman_filters[i][tracks[j]].xk[0][1])
                kalman_filters[i][tracks[j]].update(centroids[i][tracks[j]])
            else:
                kalman_filters[i][tracks[j]] = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
                kalman_filters[i][tracks[j]].predict()
                original_boxes[i][tracks[j]] = (frame2.iloc[j]['bb_width'], frame2.iloc[j]['bb_height'])
                centroids[i][tracks[j]] = compute_centroid((frame2.iloc[j]['bb_left'], frame2.iloc[j]['bb_top'],
                                                            frame2.iloc[j]['bb_width'], frame2.iloc[j]['bb_height']))
                kalman_filters[i][tracks[j]].update(centroids[i][tracks[j]])

        # display
        image = "../data/img1/" + str(int(frames[i])).zfill(6) + ".jpg"
        image = cv2.imread(image)
        boxes = []
        for id in kalman_filters[i]:
            boxes.append(centroid_to_box(centroids[i][id], original_boxes[i][id][0], original_boxes[i][id][1]))
            cv2.rectangle(image, (int(boxes[-1][0][0]), int(boxes[-1][0][1])), (int(boxes[-1][0][0] + boxes[-1][1]), int(boxes[-1][0][1] + boxes[-1][2])), (255, 0, 0), 2)
            cv2.putText(image, str(id), (int(boxes[-1][0][0]), int(boxes[-1][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(10)

        # write results
        with open('../results/results_Kalman.txt', 'a') as f:
            for j in range(len(boxes)):
                f.write(str(i-1) + ',' + str(tracks[j]) + ',' + str(boxes[j][0][0]) + ',' +
                        str(boxes[j][0][1]) + ',' + str(boxes[j][1]) + ',' +
                        str(boxes[j][2]) + ',' + str(frame2.iloc[j]['conf']) + ',' +
                        str(frame2.iloc[j]['x']) + ',' + str(frame2.iloc[j]['y']) + ',' + str(frame2.iloc[j]['z']) + '\n')


if __name__ == '__main__':
    main()