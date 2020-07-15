#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.13'


import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('img1', type=str, help="first image")
    parser.add_argument('img2', type=str, help="second image (moved from the first image)")
    args = parser.parse_args()
    img1 = cv2.imread(args.img1)
    if img1 is None:
        raise OSError("reading first image error, given path: {}".format(args.img1))
    img2 = cv2.imread(args.img2)
    if img2 is None:
        raise OSError("reading second image error, given path: {}".format(args.img2))
    orb = cv2.ORB_create()
    key_points1 = orb.detect(img1)
    key_points1, descriptors1 = orb.compute(img1, key_points1)
    key_points2 = orb.detect(img2)
    key_points2, descriptors2 = orb.compute(img2, key_points2)

    draw_img1 = cv2.drawKeypoints(img1, key_points1, None)
    draw_img2 = cv2.drawKeypoints(img2, key_points2, None)
    cv2.imshow("img1", draw_img1)
    cv2.imshow("img2", draw_img2)
    cv2.waitKey()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(descriptors1, descriptors2)
    min_dist = min(matches, key=lambda i: i.distance).distance
    max_dist = max(matches, key=lambda i: i.distance).distance
    print("min dist: {}".format(min_dist))
    print("max dist: {}".format(max_dist))

    dist_thresh = max(2 * min_dist, 30)
    good_matches = list(filter(lambda m: m.distance <= dist_thresh, matches))
    draw_match_img = cv2.drawMatches(img1, key_points1, img2, key_points2, matches, None)
    draw_good_match_img = cv2.drawMatches(img1, key_points1, img2, key_points2, good_matches, None)
    cv2.imshow("match img", draw_match_img)
    cv2.imshow("good match img", draw_good_match_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
