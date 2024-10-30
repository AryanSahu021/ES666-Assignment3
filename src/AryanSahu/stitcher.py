import pdb
import glob
import cv2
import os
import numpy as np
import random
class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        # Read images
        images = [cv2.imread(im) for im in all_images]
        images = [cv2.resize(im, (800,400)) for im in images]
        if len(images) < 2:
            print("Need at least 2 images to stitch a panorama.")
            return None, []

        # Homography matrices will be stored here
        homography_matrix_list = []
        # middle_index = len(images) // 2
        # stitched_image = images[middle_index]  # Start with the middle image as the base

        # # Stitch images from middle to the left
        # left_stitched_image = stitched_image
        # for i in range(middle_index - 1, -1, -1):
        #     H = self.compute_homography(images[i], left_stitched_image)
        #     if H is not None:
        #         homography_matrix_list.append(H)
        #         left_stitched_image = self.warp_and_stitch(left_stitched_image, images[i], H)
        #     else:
        #         print(f"Failed to compute homography for images {i} and {i+1}")

        # # Stitch images from middle to the right
        # right_stitched_image = stitched_image
        # for i in range(middle_index + 1, len(images)):
        #     H = self.compute_homography(images[i], right_stitched_image)
        #     if H is not None:
        #         homography_matrix_list.append(H)
        #         right_stitched_image = self.warp_and_stitch(right_stitched_image, images[i], H)
        #     else:
        #         print(f"Failed to compute homography for images {i-1} and {i}")
        # H = self.compute_homography(right_stitched_image, left_stitched_image)
        # stitched_image = self.warp_and_stitch(left_stitched_image, right_stitched_image, H)

        # Initialize stitched image (base as the first image)

        # Compute and apply homography for each image pair
        while (len(images)>1):
            stitched_image = images[0]
            images.pop(0)
            n = len(images)
            best_H = None
            best_in = 0
            best_img = 0
            for j in range(n):
                img = images[j]
                H, inliers = self.compute_homography(stitched_image, img)
                if inliers > best_in:
                    best_H = H
                    best_in = inliers
                    best_img = j
            homography_matrix_list.append(best_H)
            stitched_image = self.warp_and_stitch(images[best_img], stitched_image, best_H)
            images.pop(best_img)
            images.append(stitched_image)
                # if H is not None:
                    # homography_matrix_list.append(H)
                # Warp the current image to the panorama space using the homography
                # stitched_image = self.warp_and_stitch(images[i], stitched_image, H)
            # stitched_image, H = solution(stitched_image, images[i])
            # cv2.imwrite("./results/I4/aryansahu.png", stitched_image)
            # homography_matrix_list.append(H)
            # else:
            #     print(f"Failed to compute homography for images {i-1} and {i}")
        


        return stitched_image, homography_matrix_list

    def warp_to_reference(self, image, H):
        # Get image dimensions
        height, width = image.shape[:2]

        # Apply the alignment homography to the stitched image
        warped_image = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)

        return warped_image
        
    def compute_homography(self, img1, img2):
        """ Compute the homography between two images using feature matching """
        orb = cv2.ORB_create()

        # Detect ORB keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Match the descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort the matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Compute homography using DLT and RANSAC
        H, inliers = self.estimate_homography_ransac(src_pts, dst_pts)
        # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 12.0)
        return H, inliers

    def estimate_homography_ransac(self, src_pts, dst_pts, threshold=5.0, iterations=1000):
        """ RANSAC-based homography estimation from point correspondences """

        max_inliers = 0
        best_H = None

        for _ in range(iterations):
            # Randomly select 4 point pairs
            idxs = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[idxs]
            dst_sample = dst_pts[idxs]

            # Estimate homography using DLT
            H = self.estimate_homography_dlt(src_sample, dst_sample)

            # Count inliers
            if H is not None:
                inliers = self.count_inliers(H, src_pts, dst_pts, threshold)
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_H = H

        return best_H, max_inliers

    def estimate_homography_dlt(self, src_pts, dst_pts):
        """ Estimate homography using Direct Linear Transform (DLT) """

        A = []
        for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
            A.append([-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime])
            A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])

        A = np.array(A)

        # Solve Ah = 0 using SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :] / Vt[-1, -1]  # Normalize

        H = h.reshape(3, 3)
        return H

    def count_inliers(self, H, src_pts, dst_pts, threshold):
        """ Count inliers given a homography and point correspondences """
        inliers = 0
        for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
            projected_pt = H @ np.array([x, y, 1])
            projected_pt /= projected_pt[2]
            dist = np.linalg.norm([x_prime, y_prime] - projected_pt[:2])
            if dist < threshold:
                inliers += 1
        return inliers

    def warp_and_stitch(self, right_img, left_img, final_H):
        """ Warp img2 into img1's panorama space and stitch them """
        # Warp the second image to the first image's plane
        # img2 = cv2.resize(img2, img1.shape[:2])
        # img2_warped = cv2.warpPerspective(img2, H, (img2.shape[1] + img1.shape[1] , img1.shape[0]))
        
        # # Stitch by combining img1 and img2_warped
        # img2_warped[0:img1.shape[0], 0:img1.shape[1]] = img1
        # return img2_warped
        rows1, cols1 = left_img.shape[:2]
        rows2, cols2 = right_img.shape[:2]

        # Define the corner points of both images
        # Define the corner points of both images
        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)  # corners of left_img
        points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)   # corners of right_img

        # Transform the corner points of left_img using the homography matrix
        points2 = cv2.perspectiveTransform(points1, final_H)

        # Combine all corner points to find the overall bounds of the panorama
        list_of_points = np.concatenate((points, points2), axis=0)
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        # Compute translation matrix to move the panorama within the positive coordinate space
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Warp left_img onto the larger canvas with the adjusted homography matrix
        warped_left_img = cv2.warpPerspective(left_img, H_translation.dot(final_H), (x_max - x_min, y_max - y_min))

        # Create an output canvas that can hold both images
        output_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        # Place right_img onto the canvas without transformation
        output_img[-y_min:-y_min + rows2, -x_min:-x_min + cols2] = right_img

        # Add the warped left image onto the output canvas
        nonzero_mask = np.where(warped_left_img > 0)  # Only take non-zero pixels to avoid overwriting with black
        output_img[nonzero_mask[0], nonzero_mask[1]] = warped_left_img[nonzero_mask[0], nonzero_mask[1]]
        
        # Save and return the final panorama result
        cv2.imwrite("./results/I4/aryansahu.png", output_img)
        result_img = output_img
        return result_img
