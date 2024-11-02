import pdb
import glob
import cv2
import os
import numpy as np
import random
from PIL import Image, ExifTags

SENSOR_SIZES = {
    'Canon EOS 5D Mark III': (36.0, 24.0),  # Full-frame sensor size
    'Nikon D850': (35.9, 23.9),             # Full-frame sensor size
    'Sony A7R III': (35.8, 23.8),           # Full-frame sensor size
    'Canon EOS Rebel T6': (22.3, 14.9),     # APS-C sensor size
    'Nikon D5600': (23.5, 15.6),            # APS-C sensor size
    'Canon DIGITAL IXUS 860 IS': (5.75, 4.32),
    'NEX-5N': (23.4, 15.6),
    'NIKON D40': (23.7, 15.5),
    'DSC-W170': (6.17, 4.55),
    # Add more models as needed
}
class PanaromaStitcher():
    def __init__(self):
        pass

    def get_exif(self, image_path):
        """Extract focal length, camera model, and image width from EXIF data, if available."""
        try:
            img = Image.open(image_path)
            exif = img._getexif()
            exif_data = {}

            if exif:
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag)
                    if tag_name == "FocalLength":
                        # Focal length can be in IFDRational format
                        if isinstance(value, tuple):  # e.g., (35, 1) meaning 35mm
                            exif_data["FocalLength"] = value[0] / value[1]
                        else:
                            exif_data["FocalLength"] = float(value)
                    elif tag_name == "Model":
                        exif_data["CameraModel"] = value
                    elif tag_name == "ExifImageWidth":  # Extract image width
                        exif_data["ImageWidth"] = value

            # Estimate sensor width from camera model if available
            camera_model = exif_data.get("CameraModel")

            if camera_model and camera_model in SENSOR_SIZES:
                exif_data["SensorWidth"] = SENSOR_SIZES[camera_model][0]  # width in mm
                exif_data["SensorHeight"] = SENSOR_SIZES[camera_model][1] # height in mm
            else:
                print("Unknown camera model or sensor size. Using default 36mm for width.")
                exif_data["SensorWidth"] = 36.0  # Default to full-frame sensor width
            if exif_data:
                focal_length_mm = exif_data["FocalLength"]
                sensor_width_mm = exif_data["SensorWidth"]
                image_width_px = exif_data.get("ImageWidth")
                focal_length_pixels = (focal_length_mm / sensor_width_mm) * image_width_px
                return focal_length_pixels

        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            return None
    
    def resize_img(self, img, height=800):
        h, w = img.shape[:2]
        ratio = h/w
        new_size = (height, int(height*ratio))
        return cv2.resize(img, new_size)
    def make_panaroma_for_images_in(self,path):
        
        focal_pixels = {"I1": [10000, 10000, 10000, 10000, 10000, 10000],
                "I2": [631.3850174216028, 631.3850174216028, 631.3850174216028, 631.3850174216028, 631.3850174216028],
                "I3": [630, 677.3500810372772, 677.3500810372772, 677.3500810372772, 635],
                "I4": [900, 1200, 1538.4615384615386, 1538.4615384615386, 900.4615384615386],
                "I5": [1538.4615384615386, 1538.4615384615386, 1538.4615384615386, 1538.4615384615386, 1538.4615384615386],
                "I6": [3046.075949367089, 3046.075949367089, 3046.075949367089, 3299.9156118143455, 3299.9156118143455]}

        image_set = path.split("/")[1]
        if image_set in focal_pixels:
            focal_length = focal_pixels[image_set]
        else:
            focal_length = [self.get_exif(img) for img in all_images]
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        ####  Your Implementation here
        # Read images
        images = [cv2.imread(im) for im in all_images]
        
        images = [self.resize_img(im) for im in images]
        
        if len(images) < 2:
            print("Need at least 2 images to stitch a panorama.")
            return None, []

        images = [self.cylindrical_projection(images[i], focal_length[i]) for i in range(len(images))]  

        # Homography matrices will be stored here
        homography_matrix_list = []

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

        return stitched_image, homography_matrix_list
        
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
        rows1, cols1 = left_img.shape[:2]
        rows2, cols2 = right_img.shape[:2]

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
        cv2.imwrite("./results/show_stitching_while_running.png", output_img)
        result_img = output_img
        return result_img
        
    def cylindrical_projection(self, image, f):
        """Apply cylindrical projection to an image with a given focal length `f`."""
        h, w = image.shape[:2]
        cyl_image = np.zeros_like(image)

        # Calculate center of the image
        cx, cy = w // 2, h // 2

        # Perform cylindrical transformation
        for y in range(h):
            for x in range(w):
                theta = (x - cx) / f  # Angle theta
                h_ = (y - cy) / f     # Height offset

                # Compute cylindrical coordinates
                X = np.sin(theta)
                Y = h_
                Z = np.cos(theta)

                # Map back to image coordinates
                x_cyl = int(f * X / Z + cx)
                y_cyl = int(f * Y / Z + cy)

                # Assign pixel if it's within bounds
                if 0 <= x_cyl < w and 0 <= y_cyl < h:
                    cyl_image[y, x] = image[y_cyl, x_cyl]

        return cyl_image
