#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.15'


import time
import random
import argparse

import cv2
import levmar
import scipy
import scipy.linalg
import numpy as np


def homo(points_dxn):
    return np.vstack((points_dxn, np.ones((1, points_dxn.shape[1]))))


def unHomo(points_dxn):
    un_homo_points = points_dxn[:-1].copy()
    un_homo_points /= points_dxn[-1]
    return un_homo_points


def filterPts(pts_nx2, xBoardWidth, yBoardWidth, imgSize_wh):
    x_mask = np.logical_and(pts_nx2[:, 0] >= xBoardWidth, pts_nx2[:, 0] < imgSize_wh[0] - xBoardWidth)
    y_mask = np.logical_and(pts_nx2[:, 1] >= yBoardWidth, pts_nx2[:, 1] < imgSize_wh[1] - yBoardWidth)
    mask = np.logical_and(x_mask, y_mask)
    return pts_nx2[mask], mask


def get3DPts(depth_n, mask, pts2D_nx2, cameraMatrix, depthScale):
    valid_depth_n = depth_n[mask]
    valid_pts2D_nx2 = pts2D_nx2[mask]
    valid_normed_pts_nx3 = np.linalg.inv(cameraMatrix).dot(homo(valid_pts2D_nx2.T)).T
    pts3d_nx3 = valid_normed_pts_nx3 * valid_depth_n.reshape(-1, 1) * depthScale
    return pts3d_nx3


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


def getPixelValue(img, x, y):
    int_x, int_y = int(x), int(y)
    xx = x - int_x
    yy = y - int_y
    bilinear_interpolated_value = \
        (1 - xx) * (1 - yy) * img[int_y, int_x] \
        + xx * (1 - yy) * img[int_y, int_x + 1] \
        + (1 - xx) * yy * img[int_y + 1, int_x] \
        + xx * yy * img[int_y + 1, int_x + 1]
    return bilinear_interpolated_value


def estimatePoseDirect(pts3D_nx3, grayScales, grayImg, cameraMatrix):
    def errorFunc(twist, pts3D_3xn, grayScales, grayImg, cameraMatrix, mask):
        local_pts3d_3xn = unHomo(se3ToSE3(twist).dot(homo(pts3D_3xn)))
        local_pts2d_2xn = unHomo(cameraMatrix.dot(local_pts3d_3xn))[:2]
        local_pts2d_nx2 = local_pts2d_2xn.T
        valid_pts2d_nx2, mask[:] = filterPts(local_pts2d_nx2, 4, 4, grayImg.shape[::-1])
        cur_gray = np.array([getPixelValue(grayImg, p[0], p[1]) for p in valid_pts2d_nx2])
        errors = np.zeros_like(grayScales, dtype=np.float)
        errors[mask] = cur_gray - grayScales[mask]
        # errors[np.logical_not(mask)] = np.nan
        return errors

    def calJacobian(twist, pts3D_3xn, grayScales, grayImg, cameraMatrix, mask):
        fx, fy, cx, cy = cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2]

        trans_pts2 = unHomo(se3ToSE3(twist).dot(homo(pts3D_3xn)))
        Jacobian = []
        for i, m in zip(range(trans_pts2.shape[1]), mask):
            if not m:
                Jacobian.append(np.zeros((1, 6)))
                continue
            x, y, z = trans_pts2[:, i]
            invz = 1.0 / z
            invz_2 = invz * invz
            u = x * fx * invz + cx
            v = y * fy * invz + cy
            J1 = np.array([
                [-x*y*invz_2*fx, (1+(x*x*invz_2))*fx, -y*invz*fx, invz*fx, 0, -x*invz_2*fx],
                [-(1+y*y*invz_2)*fy, x*y*invz_2*fy, x*invz*fy, 0, invz * fy, -y*invz_2*fy]
            ])
            J2 = np.array([
                [getPixelValue(grayImg, u + 1, v) - getPixelValue(grayImg, u - 1, v),
                getPixelValue(grayImg, u, v + 1) - getPixelValue(grayImg, u, v - 1)]
            ]) / 2
            Jacobian.append(J2.dot(J1))
        return np.vstack(Jacobian)
    # t0 = time.time()
    # opt_twist1, _, info1 = levmar.levmar(
    #     errorFunc, np.zeros(6), [0] * len(grayScales), (pts3D_nx3.T, grayScales, grayImg, cameraMatrix)
    # )
    # print(info1)
    t1 = time.time()
    mask = np.zeros(pts3D_nx3.shape[0], dtype=np.bool)
    opt_twist2, _, info2 = levmar.levmar(
        errorFunc, np.zeros(6), [0] * len(grayScales), (pts3D_nx3.T, grayScales, grayImg, cameraMatrix, mask), calJacobian
    )
    t2 = time.time()
    # print("numerical levmar used time: {}".format(t1 - t0))
    print("analystic levmar used time: {}".format(t2 - t1))
    return se3ToSE3(opt_twist2)


def main():
    parser = argparse.ArgumentParser(description='direct sparse method')
    parser.add_argument('dataPath', type=str, help="path to data")
    args = parser.parse_args()
    with open(args.dataPath + '/associate.txt', 'r') as f:
        associate = f.readlines()

    _, color_path, _, depth_path = associate[0].split()
    first_color_img = cv2.imread("{}/{}".format(args.dataPath, color_path))
    first_grad_img = cv2.cvtColor(first_color_img, cv2.COLOR_BGR2GRAY)
    first_depth_img = cv2.imread("{}/{}".format(args.dataPath, depth_path), cv2.IMREAD_UNCHANGED)
    if first_color_img is None or first_depth_img is None:
        raise OSError("cannot read image: {}, {}".format(color_path, depth_path))

    fast_detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    key_points = fast_detector.detect(first_color_img)
    tracking_pts_nx2 = np.array([p.pt for p in key_points])
    print(len(key_points))

    valid_tracking_pts_nx2, mask = filterPts(tracking_pts_nx2, 20, 20, first_color_img.shape[:2][::-1])
    depth_n = np.array([first_depth_img[p[1], p[0]] for p in valid_tracking_pts_nx2.astype(int)])
    camera_matrix = np.array([
        [518.0, 0, 325.5],
        [0, 519.0, 253.5],
        [0, 0, 1],
    ])
    first_pts_3d_nx3 = get3DPts(depth_n, depth_n != 0, valid_tracking_pts_nx2, camera_matrix, 1000.0)
    valid_tracking_pts_nx2 = valid_tracking_pts_nx2[depth_n != 0]
    first_gray_scales_n = np.array([first_grad_img[p[1], p[0]] for p in valid_tracking_pts_nx2.astype(np.int)])
    assert len(first_gray_scales_n) == len(first_pts_3d_nx3)

    for info in associate[1:]:
        _, color_path, _, depth_path = info.split()
        color_img = cv2.imread("{}/{}".format(args.dataPath, color_path))
        depth_img = cv2.imread("{}/{}".format(args.dataPath, depth_path), cv2.IMREAD_UNCHANGED)
        if color_img is None or depth_img is None:
            break
        grad_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        Tcw = estimatePoseDirect(first_pts_3d_nx3, first_gray_scales_n, grad_img, camera_matrix)
        print("Tcw = \n{}".format(Tcw))

        pixel_now = unHomo(camera_matrix.dot(unHomo(Tcw.dot(homo(first_pts_3d_nx3.T))))).T
        valid_pixel_now, mask = filterPts(pixel_now, 0, 0, grad_img.shape[::-1])
        draw_first_img = first_color_img.copy()
        draw_now_img = color_img.copy()
        img_show = np.vstack((draw_first_img, draw_now_img))

        sampled = random.sample(range(len(valid_pixel_now)), len(valid_pixel_now) // 5)
        sampled_first_pts = valid_tracking_pts_nx2[mask].astype(int)[sampled]
        sampled_now_pts = valid_pixel_now.astype(int)[sampled]
        for p1, p2 in zip(sampled_first_pts, sampled_now_pts):
            color = np.random.randint(0, 256, 3, dtype=np.uint8).tolist()
            cv2.circle(img_show, tuple(p1), 8, color, 2)
            cv2.circle(img_show, tuple([p2[0], p2[1] + grad_img.shape[0]]), 8, color, 2)
            cv2.line(img_show, tuple(p1), tuple([p2[0], p2[1] + grad_img.shape[0]]), color, 1)

        cv2.imshow("result", img_show)
        cv2.waitKey()


if __name__ == '__main__':
    main()
