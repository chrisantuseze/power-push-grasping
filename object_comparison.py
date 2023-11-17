import cv2
import math

def get_object_centroid(segmentation_mask):
    # Find the contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the center of mass and maximum contour area
    max_contour_area = 0
    center_of_mass = None

    # Iterate through all the contours to find the largest one and its center of mass
    for contour in contours:
        # Calculate the area of the current contour
        contour_area = cv2.contourArea(contour)

        # Update the maximum contour area and center of mass if a larger contour is found
        if contour_area > max_contour_area:
            max_contour_area = contour_area
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    center_of_mass = list(center_of_mass)
    return center_of_mass

def get_distance(point1, point2):
    # Calculate the differences

    '''
    d = √((x2 - x1)^2 + (y2 - y1)^2)
    '''

    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calculate the squared differences
    delta_x_squared = delta_x ** 2
    delta_y_squared = delta_y ** 2

    # Sum of squared differences
    sum_of_squared_diff = delta_x_squared + delta_y_squared

    # Calculate the distance
    distance = math.sqrt(sum_of_squared_diff)

    # print("Distance between the two points:", distance)
    return distance


def get_grasped_object(processed_masks, action):

    for id, mask in enumerate(processed_masks):
        dist = get_distance(get_object_centroid(mask), (action[0], action[1]))
        print(dist)
        if dist < 250:
            return id, mask

    return -1, None
