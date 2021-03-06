import json
import numpy as np

dist_list = []
for i in range(2735):
    total_dist = 0
    if i == 67: continue # only for ours/obama1 case
    with open('./json_gt_obama1/'+'obama1_cut%04d_keypoints.json' % i, 'r') as f:
        gt = json.load(f)
        gt = gt['people'][0]['face_keypoints_2d']
    with open('./json_ours_obama1/'+'%04d_keypoints.json' % i, 'r') as f:
        ours = json.load(f)
        ours = ours['people'][0]['face_keypoints_2d']

    for j in range(70):
        gt_coord = np.array(gt[0+3*j:2+3*j])
        ours_coord = np.array(ours[0+3*j:2+3*j])
        dist = np.linalg.norm(gt_coord-ours_coord)
        total_dist = total_dist + dist
        if j == 69:
            total_dist = total_dist / 70

    dist_list.append(total_dist)

lmk = sum(dist_list) / len(dist_list)
print('Landmark Difference:', lmk)