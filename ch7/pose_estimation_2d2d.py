#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.14'


import argparse

import cv2
import numpy as np


def homo(points_dxn):
    return np.vstack((points_dxn, np.ones((1, points_dxn.shape[1]))))


def skewSymmetric(vec):
    vec_array = np.array(vec).ravel()
    return np.array([
        [0, -vec_array[2], vec_array[1]],
        [vec_array[2], 0, -vec_array[0]],
        [-vec_array[1], vec_array[0], 0]
    ])


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


def main():
    parser = argparse.ArgumentParser(description='Pose estimation from two 2D images')
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

    essential_matrix = skewSymmetric(t).dot(R)
    homo_matched_points1 = homo(matched_points1.T)
    homo_matched_points2 = homo(matched_points2.T)
    matched_points_cam1 = np.linalg.inv(camera_matrix).dot(homo_matched_points1)
    matched_points_cam2 = np.linalg.inv(camera_matrix).dot(homo_matched_points2)
    errors = [p2.dot(essential_matrix).dot(p1.T) for p1, p2 in zip(matched_points_cam1.T, matched_points_cam2.T)]
    print(errors)


if __name__ == '__main__':
    main()