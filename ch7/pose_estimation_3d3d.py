#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.14'


import time
import argparse

import g2o
import cv2
import numpy as np
import levmar
import scipy
import scipy.linalg


def homo(points_dxn):
    return np.vstack((points_dxn, np.ones((1, points_dxn.shape[1]))))


def unHomo(points_dxn):
    un_homo_points = points_dxn[:-1].copy()
    un_homo_points /= points_dxn[-1]
    return un_homo_points


def findFeatureMatches(img1, img2, minDistThresh=30):
    orb = cv2.ORB_create()
    key_points1 = orb.detect(img1)
    key_points1, descriptors1 = orb.compute(img1, key_points1)
    key_points2 = orb.detect(img2)
    key_points2, descriptors2 = orb.compute(img2, key_points2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(descriptors1, descriptors2)
    min_dist = min(matches, key=lambda i: i.distance).distance
    dist_thresh = max(2 * min_dist, minDistThresh)
    good_matches = list(filter(lambda m: m.distance <= dist_thresh, matches))
    return key_points1, key_points2, good_matches


def get3DPts(depth_n, mask, pts2D, cameraMatrix, depthScale):
    valid_depth_n = depth_n[mask]
    valid_pts2D_nx2 = pts2D[mask]
    valid_normed_pts_nx3 = np.linalg.inv(cameraMatrix).dot(homo(valid_pts2D_nx2.T)).T
    pts3d_nx3 = valid_normed_pts_nx3 * valid_depth_n.reshape(-1, 1) * depthScale
    return pts3d_nx3


def estimatePose3D3D(pts1, pts2):
    p1 = pts1.mean(axis=0)
    p2 = pts2.mean(axis=0)
    q1 = pts1 - p1
    q2 = pts2 - p2
    W = sum([v1.reshape(3, 1).dot(v2.reshape(1, 3)) for v1, v2 in zip(q1, q2)])
    U, _, VT = np.linalg.svd(W)
    if np.linalg.det(U) * np.linalg.det(VT) < 0:
        U[:, 2] *= -1
    R = U.dot(VT)
    t = p1 - R.dot(p2)
    return R, t


def skewSymmetric(vec):
    vec_array = np.array(vec).ravel()
    return np.array([
        [0, -vec_array[2], vec_array[1]],
        [vec_array[2], 0, -vec_array[0]],
        [-vec_array[1], vec_array[0], 0]
    ])


def X(twist):
    x = np.zeros((4, 4))
    x[:3, :3] = skewSymmetric(twist[:3])
    x[:3, -1] = twist[-3:].ravel()
    return x


def X_inv(x):
    w = np.array([x[2, 1], x[0, 2], x[1, 0]])
    v = x[:3, -1]
    t = np.zeros((6, 1))
    t[:3] = w.reshape(3, 1)
    t[-3:] = v.reshape(3, 1)
    return t


def se3ToSE3(twist):
    return scipy.linalg.expm(X(twist))


def SE3Tose3(T):
    return X_inv(scipy.linalg.logm(T))


def bundleAdjustment_LM(pts1_3xn, pts2_3xn):
    def errorFunc(twist, pts1, pts2):
        return (pts1 - unHomo(se3ToSE3(twist).dot(homo(pts2)))).T.ravel()
    def calJacobian(twist, pts1, pts2):
        trans_pts2 = unHomo(se3ToSE3(twist).dot(homo(pts2)))
        Jacobian = []
        for i in range(trans_pts2.shape[1]):
            x, y, z = trans_pts2[:, i]
            J = np.array([
                [0, -z, y, -1, 0, 0],
                [z, 0, -x, 0, -1, 0],
                [-y, x, 0, 0, 0, -1],
            ])
            Jacobian.append(J)
        return np.vstack(Jacobian)
    t0 = time.time()
    opt_twist1, _, info1 = levmar.levmar(errorFunc, np.zeros(6), [0] * pts1_3xn.size, (pts1_3xn, pts2_3xn))
    t1 = time.time()
    opt_twist2, _, info2 = levmar.levmar(errorFunc, np.zeros(6), [0] * pts1_3xn.size, (pts1_3xn, pts2_3xn), calJacobian)
    t2 = time.time()
    print("numerical levmar used time: {}".format(t1 - t0))
    print("analystic levmar used time: {}".format(t2 - t1))
    return se3ToSE3(opt_twist2)


def main():
    parser = argparse.ArgumentParser(description='Pose estimation / PNP')
    parser.add_argument('img1', type=str, help="first image")
    parser.add_argument('img2', type=str, help="second image (moved from the first image)")
    parser.add_argument('depth_img1', type=str, help="first depth image")
    parser.add_argument('depth_img2', type=str, help="second depth image (moved from the first image)")
    args = parser.parse_args()
    img1 = cv2.imread(args.img1)
    if img1 is None:
        raise OSError("reading first image error, given path: {}".format(args.img1))
    img2 = cv2.imread(args.img2)
    if img2 is None:
        raise OSError("reading second image error, given path: {}".format(args.img2))
    depth_img1 = cv2.imread(args.depth_img1, cv2.IMREAD_UNCHANGED)
    if depth_img1 is None:
        raise OSError("reading first depth image error, given path: {}".format(args.depth_img1))
    depth_img2 = cv2.imread(args.depth_img2, cv2.IMREAD_UNCHANGED)
    if depth_img2 is None:
        raise OSError("reading second depth image error, given path: {}".format(args.depth_img2))

    key_points1, key_points2, good_matches = findFeatureMatches(img1, img2)
    matched_points1_nx2 = np.array([key_points1[m.queryIdx].pt for m in good_matches])
    matched_points2_nx2 = np.array([key_points2[m.trainIdx].pt for m in good_matches])

    camera_matrix = np.array([
        [520.9, 0, 325.1],
        [0, 521.0, 249.7],
        [0, 0, 1]
    ])
    depth1_n = np.array([depth_img1[p[1], p[0]]for p in matched_points1_nx2.astype(int)])
    depth2_n = np.array([depth_img2[p[1], p[0]]for p in matched_points2_nx2.astype(int)])
    valid_depth_mask = np.logical_and(depth1_n != 0, depth2_n != 0)
    points3d_cam1_nx3 = get3DPts(depth1_n, valid_depth_mask, matched_points1_nx2, camera_matrix, 1 / 5000.)
    points3d_cam2_nx3 = get3DPts(depth2_n, valid_depth_mask, matched_points2_nx2, camera_matrix, 1 / 5000.)
    print(points3d_cam1_nx3.shape)
    print(points3d_cam2_nx3.shape)

    R, t = estimatePose3D3D(points3d_cam1_nx3, points3d_cam2_nx3)
    print("R = \n{}".format(R))
    print("t = \n{}".format(t))

    T = bundleAdjustment_LM(points3d_cam1_nx3.T, points3d_cam2_nx3.T)
    print("T = \n{}".format(T))



if __name__ == '__main__':
    main()