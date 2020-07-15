#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.14'


import argparse

import g2o
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


def bundleAdjustment(pts3D, pts2D, cameraMatrix, R, t):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3()))
    optimizer.set_algorithm(solver)

    pose = g2o.VertexSE3Expmap()
    pose.set_estimate(g2o.SE3Quat(R, t.reshape(3, )))
    pose.set_id(0)
    optimizer.add_vertex(pose)

    index = 1
    for p_3d in pts3D:
        point = g2o.VertexSBAPointXYZ()
        point.set_id(index)
        point.set_estimate(p_3d)
        point.set_marginalized(True)
        optimizer.add_vertex(point)
        index += 1

    camera = g2o.CameraParameters(cameraMatrix[0, 0], cameraMatrix[:2, 2], 0)
    camera.set_id(0)
    optimizer.add_parameter(camera)

    index = 1
    for p_2d in pts2D:
        edge = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
        edge.set_id(index)
        edge.set_vertex(0, optimizer.vertex(index))
        edge.set_vertex(1, pose)
        edge.set_measurement(p_2d)
        edge.set_parameter_id(0, 0)
        edge.set_information(np.eye(2))
        optimizer.add_edge(edge)
        index += 1

    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(100)
    return pose.estimate().matrix()


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
    depth_n = np.array([depth_img1[p[1], p[0]]for p in matched_points1_nx2.astype(int)])
    valid_depth_n = depth_n[depth_n != 0]
    valid_matched_points1_nx2 = matched_points1_nx2[depth_n != 0]
    valid_matched_points2_nx2 = matched_points2_nx2[depth_n != 0]
    valid_matched_points1_nx3 = np.linalg.inv(camera_matrix).dot(homo(valid_matched_points1_nx2.T)).T
    points3d_cam1_nx3 = valid_matched_points1_nx3 * valid_depth_n.reshape(-1, 1) / 5000.0

    _, r, t = cv2.solvePnP(points3d_cam1_nx3, valid_matched_points2_nx2, camera_matrix, ())
    R, _ = cv2.Rodrigues(r)
    print("R:\n{}".format(R))
    print("t:\n{}".format(t))

    T = bundleAdjustment(points3d_cam1_nx3, valid_matched_points1_nx2, camera_matrix, R, t)
    print('T =\n{}'.format(T))


if __name__ == '__main__':
    main()