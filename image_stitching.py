import cv2
import numpy as np

def match_sift_features(image_path1, image_path2, max_features):
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Check if images were loaded successfully
    if img1 is None or img2 is None:
        print(f"Error: Could not read one or both images")
        return None, None, None, None
    
    # Convert to grayscale (SIFT works on grayscale images)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with the requested number of features
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    print(f"Number of keypoints in image1: {len(keypoints1)}")
    print(f"Number of keypoints in image2: {len(keypoints2)}")
    
    # Create a BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher()
    
    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Visualize matches
    visualize_matches(img1, img2, keypoints1, keypoints2, good_matches)
    
    return img1, img2, keypoints1, keypoints2, good_matches

def visualize_matches(img1, img2, keypoints1, keypoints2, good_matches):
    """Visualize the matching features between two images"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a blank image that can fit both images side by side
    max_height = max(h1, h2)
    combined_img = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
    
    # Place the images side by side
    combined_img[0:h1, 0:w1] = img1
    combined_img[0:h2, w1:w1+w2] = img2
    
    # Draw circles at keypoints for both images
    for kp in keypoints1:
        x, y = map(int, kp.pt)
        cv2.circle(combined_img, (x, y), 7, (0, 0, 255), 2)
    
    for kp in keypoints2:
        x, y = map(int, kp.pt)
        x += w1  # Offset for the second image
        cv2.circle(combined_img, (x, y), 10, (0, 0, 255), 3)
    
    # Draw lines between matching keypoints
    for match in good_matches:
        x1, y1 = map(int, keypoints1[match.queryIdx].pt)
        x2, y2 = map(int, keypoints2[match.trainIdx].pt)
        x2 += w1  # Offset for the second image
        cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the combined image with keypoints and matches
    cv2.imshow('SIFT Matches', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def estimate_homography(keypoints1, keypoints2, good_matches):
    """Estimate homography matrix using RANSAC"""
    if len(good_matches) < 4:
        print("Error: Not enough matches to compute homography (minimum 4 required)")
        return None
    
    # Extract matching points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Count inliers
    inliers = np.sum(mask)
    print(f"Number of inliers: {inliers} out of {len(good_matches)} matches")
    
    return homography

def warp_and_blend_images(img1, img2, homography):
    """Warp the first image and blend it with the second image"""
    h1, w1 = img1.shape[:2] # Get dimensions of the first image
    h2, w2 = img2.shape[:2] # Get dimensions of the second image
    
    # Get corners of the first image
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    
    # Transform corners using homography
    transformed_corners = cv2.perspectiveTransform(corners1, homography)
    
    # Get corners of the second image
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    
    # Combine all corners to find the bounding box of the panorama
    all_corners = np.concatenate((transformed_corners, corners2), axis=0)
    
    # Find the bounding box using the max and min of the corner coordinates
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    
    # Calculate translation to ensure all coordinates are positive
    translation_x = -x_min if x_min < 0 else 0
    translation_y = -y_min if y_min < 0 else 0
    
    # Create translation matrix
    translation_matrix = np.array([[1, 0, translation_x],
                                   [0, 1, translation_y],
                                   [0, 0, 1]], dtype=np.float32)
    
    # Combine homography with translation
    combined_homography = translation_matrix @ homography
    
    # Calculate output size
    output_width = x_max - x_min
    output_height = y_max - y_min
    
    print(f"Output panorama size: {output_width} x {output_height}")
    
    # Warp the first image
    warped_img1 = cv2.warpPerspective(img1, combined_homography, (output_width, output_height))
    
    # Create a canvas for the second image with translation
    canvas_img2 = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Place the second image on the canvas
    start_x = translation_x
    start_y = translation_y
    end_x = start_x + w2
    end_y = start_y + h2
    
    # Ensure we don't exceed canvas boundaries
    end_x = min(end_x, output_width)
    end_y = min(end_y, output_height)
    
    # Place the second image on the canvas
    canvas_img2[start_y:end_y, start_x:end_x] = img2[:end_y-start_y, :end_x-start_x]
    
    # # For debugging: show image2 on canvas
    # cv2.imshow('Debugging: image2', canvas_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Create masks for blending (1s where images are present and 0s where they are not)
    mask1 = (warped_img1 > 0).astype(np.float32)
    mask2 = (canvas_img2 > 0).astype(np.float32)
    
    # Find overlap region
    overlap_mask = (mask1 * mask2).astype(np.float32)
    
    # Create blending weights using distance transform
    # For smoother blending, we'll use a feathering approach (applying weights based on distance between each pixel and the nearest edge)
    mask1_dist = cv2.distanceTransform((mask1[:,:,0] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    mask2_dist = cv2.distanceTransform((mask2[:,:,0] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    
    # Normalize distance transforms
    total_dist = mask1_dist + mask2_dist
    total_dist[total_dist == 0] = 1  # Avoid division by zero
    
    weight1 = mask1_dist / total_dist
    weight2 = mask2_dist / total_dist
    
    # Expand weights to 3 channels since images are in RGB color
    weight1 = np.stack([weight1] * 3, axis=-1)
    weight2 = np.stack([weight2] * 3, axis=-1)
    
    # Apply weights only in overlap regions
    overlap_3d = np.stack([overlap_mask[:,:,0]] * 3, axis=-1)
    weight1 = weight1 * overlap_3d + mask1 * (1 - overlap_3d)
    weight2 = weight2 * overlap_3d + mask2 * (1 - overlap_3d)
    # For pixels inside the overlap area (where overlap_3d = 1): use weight1 or weight2 (smooth blending weights)
    # For pixels outside the overlap area (where overlap_3d = 0): use mask1 or mask2 (which is basically 1 for that image area and 0 elsewhere)

    # Blend the images
    blended_img = (warped_img1.astype(np.float32) * weight1 + 
                   canvas_img2.astype(np.float32) * weight2)
    
    # Normalize and convert back to uint8 for display
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    # Blending with weights could produce values outside the valid image range of 0 to 255
    # Normalizing ensures that the final blended image is within valid range

    return blended_img

def stitch_images(image_path1, image_path2, max_features=500):
    """Complete image stitching pipeline"""
    print("Starting image stitching process...")
    
    # Step 1 & 2: Feature detection and matching
    print("\n1. Detecting and matching features...")
    result = match_sift_features(image_path1, image_path2, max_features)
    if result[0] is None:
        return None
    
    img1, img2, keypoints1, keypoints2, good_matches = result   # Unpack the result tuple
    
    # Step 3: Transformation estimation (Homography)
    print("\n2. Estimating homography...")
    homography = estimate_homography(keypoints1, keypoints2, good_matches)
    if homography is None:
        return None
    
    print("Homography matrix:")
    print(homography)
    
    # Step 4 & 5: Image warping (applying Homography transformation) and blending
    print("\n3. Warping and blending images...")
    panorama = warp_and_blend_images(img1, img2, homography)
    
    print("Image stitching completed successfully!")
    return panorama

if __name__ == "__main__":
    # Number of features to detect
    max_features = 500
    
    # Path to the images
    image_path1 = "image11.png"
    image_path2 = "image12.png"
    
    # Perform image stitching
    result = stitch_images(image_path1, image_path2, max_features)
    
    if result is not None:
        # Display the result
        cv2.imshow('Panorama', result)
        
        # # Save the result
        # cv2.imwrite('panorama_result.jpg', result)
        # print("Panorama saved as 'panorama_result.jpg'")
        
        # Wait for key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image stitching failed!")
