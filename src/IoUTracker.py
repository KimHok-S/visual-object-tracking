import sys

import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment


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


def greedy_assignment(similarity_matrix, previous_tracks, sigma_iou):
    assignments = []
    used_cols = []
    for i in range(similarity_matrix.shape[1]):
        max_col = np.argsort(-(similarity_matrix[:, i]))
        for col in max_col:
            if similarity_matrix[col][i] < sigma_iou:
                break
            if col not in used_cols:
                assignments.append((i, previous_tracks[col]))
                used_cols.append(col)
                break
    return assignments


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


def display_boxes(boxes, labels, tracks):
    image = "../data/img1/" + str(int(labels)).zfill(6) + ".jpg"
    image = cv2.imread(image)
    for i in range(len(boxes)):
        cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3])), (0, 255, 0), 2)
        cv2.putText(image, str(tracks[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', image)


def frame_to_boxes(frame):
    boxes = np.zeros((len(frame), 4))
    for i in range(len(frame)):
        boxes[i][0] = frame.iloc[i]['bb_left']
        boxes[i][1] = frame.iloc[i]['bb_top']
        boxes[i][2] = frame.iloc[i]['bb_width']
        boxes[i][3] = frame.iloc[i]['bb_height']
    return boxes


def main():
    tracking_method = sys.argv[1]

    det = pd.read_csv('../data/det/det.txt', sep=',', index_col=0)
    frames = det.index.unique()
    sigma_iou = 0.2
    tracks = update_tracks(0, [], len(det.loc[1]))

    for i in range(1, len(frames)-1):
        frame1 = det.loc[i]
        frame2 = det.loc[i + 1]
        similarity_matrix = compute_similarity_matrix(frame1, frame2)
        if tracking_method == 'greedy':
            assignments = greedy_assignment(similarity_matrix, tracks, sigma_iou)
        elif tracking_method == 'hungarian':
            assignments = hungarian_assignment(similarity_matrix, tracks, sigma_iou)
        else:
            print('Invalid tracking method')
            return
        tracks = update_tracks(max(tracks), assignments, len(frame2))
        display_boxes(frame_to_boxes(frame2), frames[i + 1], tracks)
        cv2.waitKey(10)

        with open('../results/results_TP3.txt', 'a') as f:
            for col, id in assignments:
                f.write(str(i) + ',' + str(id) + ',' + str(frame2.iloc[col]['bb_left']) + ',' +
                        str(frame2.iloc[col]['bb_top']) + ',' + str(frame2.iloc[col]['bb_width']) + ',' +
                        str(frame2.iloc[col]['bb_height']) + ',' + str(frame2.iloc[col]['conf']) + ',' +
                        str(frame2.iloc[col]['x']) + ',' + str(frame2.iloc[col]['y']) + ',' + str(frame2.iloc[col]['z']) + '\n')


if __name__ == '__main__':
    main()