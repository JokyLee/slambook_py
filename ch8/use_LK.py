#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.15'


import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='LK optical flow')
    parser.add_argument('dataPath', type=str, help="path to data")
    args = parser.parse_args()
    with open(args.dataPath + '/associate.txt', 'r') as f:
        associate = f.readlines()

    _, color_path, _, depth_path = associate[0].split()
    color_img = cv2.imread("{}/{}".format(args.dataPath, color_path))
    if color_img is None:
        raise OSError("cannot read image: {}".format(color_path))
    fast_detector = cv2.FastFeatureDetector_create()
    key_points = fast_detector.detect(color_img)
    tracking_pts = np.array([p.pt for p in key_points], dtype=np.float32)
    prev_color_img = color_img
    print("nnumber of tracking points: {}".format(len(tracking_pts)))
    for pt in tracking_pts:
        cv2.circle(color_img, tuple(pt), 10, (0, 240, 0), 1)
    cv2.imshow("color", color_img)
    cv2.waitKey()

    for info in associate[1:]:
        _, color_path, _, depth_path = info.split()
        color_img = cv2.imread("{}/{}".format(args.dataPath, color_path))
        if color_img is None:
            break

        next_keypoints, status, error = cv2.calcOpticalFlowPyrLK(prev_color_img, color_img, tracking_pts, None)
        tracking_pts = next_keypoints[(status == 1).ravel()]
        prev_color_img = color_img
        print("nnumber of tracking points: {}".format(len(tracking_pts)))
        for pt in tracking_pts:
            cv2.circle(color_img, tuple(pt), 10, (0, 240, 0), 1)
        cv2.imshow("color", color_img)
        cv2.waitKey()


if __name__ == '__main__':
    main()
