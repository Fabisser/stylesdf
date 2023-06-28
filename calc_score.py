import numpy as np

def calc_score(inputs):
    i, j, images, extrinsic, points3d, args = inputs
    id_i = images[i+1].point3D_ids
    id_j = images[j+1].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]
    cam_center_i = -np.matmul(extrinsic[i+1][:3, :3].transpose(), extrinsic[i+1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j+1][:3, :3].transpose(), extrinsic[j+1][:3, 3:4])[:, 0]
    score = 0
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(cam_center_j - p))
        score += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
    return i, j, score