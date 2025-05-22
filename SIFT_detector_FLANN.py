import cv2
import numpy as np

def match_sift_features(image_path1, image_path2, max_features):
    # Read the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Check if images were loaded successfully
    if img1 is None or img2 is None:
        print(f"Error: Could not read one or both images")
        return
    
    # Convert to grayscale (SIFT works on grayscale images)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    # sift = cv2.SIFT_create()

    # Initialize SIFT detector with the requested number of features
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Detect and compute keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    print(f"Number of keypoints in image1: {len(keypoints1)}")
    print(f"Number of keypoints in image2: {len(keypoints2)}")

    # Create a BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher()

    # Euclidean distance measures how "far apart" two descriptors are in 128D space
    # meaning how different / similar the two descriptors are

    # Match descriptors
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # k = 2 means we get the two best matches for each descriptor
    # This is used later to compare the Euclidean distances of the two best matches

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # 5 KD trees
    search_params = dict(checks=50) # do 50 checks on leaf nodes of the KD trees
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # insert parameters into the FLANN matcher

    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches: # m is the best match, n is the second best match
        if m.distance < 0.75 * n.distance: # if the Euclidean distance of the best match is sufficiently smaller than the second best match
            # Append the best match to the good matches list
            good_matches.append(m)
        # if the Euclidean distance of the best match is not sufficiently smaller than the second best match, we discard it (they are too similar, could create noises, filter them out)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    # Create a blank image that can fit both images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate the height of the combined image (maximum height of the two images)
    max_height = max(h1, h2)
    
    # Create a blank image with width = width of both images combined and height = max height
    combined_img = np.zeros((max_height, w1 + w2, 3), dtype=np.uint8)
    
    # Place the first image on the left side
    combined_img[0:h1, 0:w1] = img1
    
    # Place the second image on the right side
    combined_img[0:h2, w1:w1+w2] = img2
    
    # Draw circles at keypoints for both images
    for kp in keypoints1:
        x, y = map(int, kp.pt)
        cv2.circle(combined_img, (x, y), 7, (0, 0, 255), 2)
    
    for kp in keypoints2:
        x, y = map(int, kp.pt)
        # Add the width of the first image to x-coordinate
        x += w1
        cv2.circle(combined_img, (x, y), 10, (0, 0, 255), 3)
    
    # Draw lines between matching keypoints
    for match in good_matches:  # only get keypoints that are good matches to draw lines
        # Get the coordinates of matching keypoints
        x1, y1 = map(int, keypoints1[match.queryIdx].pt)
        x2, y2 = map(int, keypoints2[match.trainIdx].pt)
        
        # Adjust x2 coordinate to account for the offset in the combined image
        x2 += w1
        
        # Draw a line between matching points (green color)
        cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Display the combined image with keypoints and matches
    cv2.imshow('SIFT Matches', combined_img)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Number of features to detect
    max_features = 100
    
    # Path to the images
    image_path1 = "image9.png"
    image_path2 = "image10.png"
    
    # Match SIFT features between the two images
    match_sift_features(image_path1, image_path2, max_features)
