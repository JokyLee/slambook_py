#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.14'


import argparse

import cv2
import numpy as np


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


def estimatePose2D2D(keyPoints1, keyPoints2, cameraMatrix):
    essential_matrix, _  = cv2.findEssentialMat(keyPoints1, keyPoints2, cameraMatrix, cv2.FM_8POINT)
    _, R, t, mask = cv2.recoverPose(essential_matrix, keyPoints1, keyPoints2, cameraMatrix)
    return R, t


def triangulation(keyPoints1, keyPoints2, cameraMatrix, R, t):
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, :3] = R
    T2[:3, -1] = t.ravel()
    homo_matched_points1_3xn = homo(keyPoints1.T)
    homo_matched_points2_3xn = homo(keyPoints2.T)
    matched_points_cam1_3xn = np.linalg.inv(cameraMatrix).dot(homo_matched_points1_3xn)
    matched_points_cam2_3xn = np.linalg.inv(cameraMatrix).dot(homo_matched_points2_3xn)
    points_4xn = cv2.triangulatePoints(T1[:3, :], T2[:3, :], matched_points_cam1_3xn[:2], matched_points_cam2_3xn[:2])
    return unHomo(points_4xn)


def main():
    parser = argparse.ArgumentParser(description='triangulation points')
    parser.add_argument('img1', type=str, help="first image")
    parser.add_argument('img2', type=str, help="second image (moved from the first image)")
    args = parser.parse_args()
    img1 = cv2.imread(args.img1)
    if img1 is None:
        raise OSError("reading first image error, given path: {}".format(args.img1))
    img2 = cv2.imread(args.img2)
    if img2 is None:
        raise OSError("reading second image error, given path: {}".format(args.img2))

    key_points1, key_points2, good_matches = findFeatureMatches(img1, img2)
    matched_points1 = np.array([key_points1[m.queryIdx].pt for m in good_matches])
    matched_points2 = np.array([key_points2[m.trainIdx].pt for m in good_matches])

    camera_matrix = np.array([
        [520.9, 0, 325.1],
        [0, 521.0, 249.7],
        [0, 0, 1]
    ])
    R, t = estimatePose2D2D(matched_points1, matched_points2, camera_matrix)
    print("R:\n{}".format(R))
    print("t:\n{}".format(t))

    points_cam1_3xn = triangulation(matched_points1, matched_points1, camera_matrix, R, t)

    matched_points_cam1_3xn = np.linalg.inv(camera_matrix).dot(homo(matched_points1.T))
    normed_matched_points_cam1_2xn = unHomo(points_cam1_3xn)
    print("points errors in the first camera frame:")
    print(matched_points_cam1_3xn[:2, :] - normed_matched_points_cam1_2xn)
    img_points_cam1_2xn = unHomo(camera_matrix.dot(points_cam1_3xn))
    img_points_cam1_3xn_ = camera_matrix.dot(homo(unHomo(points_cam1_3xn)))
    print("image points errors in the first camera:")
    print(img_points_cam1_2xn - matched_points1.T)

    points_cam2_3xn = R.dot(points_cam1_3xn) + t
    matched_points_cam2_3xn = np.linalg.inv(camera_matrix).dot(homo(matched_points2.T))
    normed_matched_points_cam2_2xn = unHomo(points_cam2_3xn)
    print("points errors in the second camera frame:")
    print(matched_points_cam2_3xn[:2, :] - normed_matched_points_cam2_2xn)
    img_points_cam2_2xn = unHomo(camera_matrix.dot(points_cam2_3xn))
    print("image points errors in the second camera:")
    print(img_points_cam2_2xn - matched_points2.T)


if __name__ == '__main__':
    main()