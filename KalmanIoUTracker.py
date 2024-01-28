import sys

import numpy as np
import pandas as pd
import cv2
from KalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


def compute_centroid(bbox):
    return (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))


# bbox = (bb_left , bb_top), width, height
def iou(bbox1, bbox2):
    xA = max(bbox1[0][0], bbox2[0][0])
    yA = max(bbox1[0][1], bbox2[0][1])
    xB = min(bbox1[0][0] + bbox1[1], bbox2[0][0] + bbox2[1])
    yB = min(bbox1[0][1] + bbox1[2], bbox2[0][1] + bbox2[2])

    intersection = max(0, (xB - xA) * (yB - yA))

    return intersection / ((bbox1[1] * bbox1[2]) + (bbox2[1] * bbox2[2]) - intersection)


def compute_similarity_matrix(frame1, frame2):
    similarity_matrix = np.zeros((len(frame1), len(frame2)))
    for i in range(len(frame1)):
        for j in range(len(frame2)):
            bbox1 = ((frame1.iloc[i]['bb_left'], frame1.iloc[i]['bb_top']), frame1.iloc[i]['bb_width'], frame1.iloc[i]['bb_height'])
            bbox2 = ((frame2.iloc[j]['bb_left'], frame2.iloc[j]['bb_top']), frame2.iloc[j]['bb_width'], frame2.iloc[j]['bb_height'])
            similarity_matrix[i][j] = iou(bbox1, bbox2)
    return similarity_matrix


def hungarian_assignment(similarity_matrix, previous_tracks, sigma_iou):
    clear_sim_matrix = similarity_matrix.copy()
    for i in range(len(clear_sim_matrix)):
        for j in range(len(clear_sim_matrix[i])):
            if clear_sim_matrix[i][j] < sigma_iou:
                clear_sim_matrix[i][j] = 0
    rows, cols = linear_sum_assignment(-clear_sim_matrix)
    assignments = []
    for i in range(len(rows)):
        if clear_sim_matrix[rows[i]][cols[i]] != 0:
            assignments.append((cols[i], previous_tracks[rows[i]]))
    return assignments


def update_tracks(max_id, assignments, nb_obj):
    updated_tracks = []
    for i in range(nb_obj):
        is_new = True
        for col, id in assignments:
            if i == col:
                updated_tracks.append(id)
                is_new = False
        if is_new:
            max_id += 1
            updated_tracks.append(max_id)
    return updated_tracks


def centroid_to_box(centroid, width, height):
    return (centroid[0] - (width / 2), centroid[1] - (height / 2)), width, height


# original_boxes = {frame_nb: {id: (bb_width, bb_height)}}
def original_to_frame(original_boxes, centroids):
    print("original_boxes", original_boxes)
    print("centroids", centroids)
    frame = pd.DataFrame(columns=['bb_left', 'bb_top', 'bb_width', 'bb_height'])
    for id in original_boxes:
        bbox = centroid_to_box(centroids[id], original_boxes[id][0], original_boxes[id][1])
        frame = frame._append({'bb_left': bbox[0][0],
                              'bb_top': bbox[0][1],
                              'bb_width': bbox[1],
                              'bb_height': bbox[2]}, ignore_index=True)
    return frame


def setup_dict_id(kalman_boxes, original_boxes, centroids, kalman_filters, frame_nb):
    kalman_boxes[frame_nb] = {}
    original_boxes[frame_nb] = {}
    centroids[frame_nb] = {}
    kalman_filters[frame_nb] = {}


def main():
    det = pd.read_csv('ADL-Rundle-6/det/det.txt', sep=',', index_col=0)
    frames = det.index.unique()
    sigma_iou = 0.2

    kalman_filters = {}
    original_boxes = {}
    centroids = {}
    setup_dict_id(kalman_filters, original_boxes, centroids, kalman_filters, 1)

    dt = 0.1
    u_x, u_y, std_acc = 1, 1, 1
    x_std_meas = 0.1
    y_std_meas = 0.1

    tracks = update_tracks(0, [], len(det.loc[1]))
    for i in range(1, len(tracks) + 1):
        kalman_filters[1][i] = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        kalman_filters[1][i].predict()
        original_boxes[1][i] = (det.loc[1].iloc[i - 1]['bb_width'], det.loc[1].iloc[i - 1]['bb_height'])
        centroids[1][i] = compute_centroid((det.loc[1].iloc[i - 1]['bb_left'], det.loc[1].iloc[i - 1]['bb_top'],
                                            det.loc[1].iloc[i - 1]['bb_width'], det.loc[1].iloc[i - 1]['bb_height']))
        kalman_filters[1][i].update(centroids[1][i])

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


        image = "ADL-Rundle-6/img1/" + str(int(frames[i])).zfill(6) + ".jpg"
        image = cv2.imread(image)
        boxes = []
        for id in kalman_filters[i]:
            boxes.append(centroid_to_box(centroids[i][id], original_boxes[i][id][0], original_boxes[i][id][1]))
            cv2.rectangle(image, (int(boxes[-1][0][0]), int(boxes[-1][0][1])), (int(boxes[-1][0][0] + boxes[-1][1]), int(boxes[-1][0][1] + boxes[-1][2])), (255, 0, 0), 2)
            cv2.putText(image, str(id), (int(boxes[-1][0][0]), int(boxes[-1][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()