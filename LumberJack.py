# Import the required libraries
import numpy as np
import cv2
import time
start_time = time.time()

# Constants
ref_height_cm = 80
ref_width_cm = 40
log_length = 300
cm_cb_m_cb = 1e-6

# Removes isolated pixels from an image
def remove_isolated_pixels(image, connectivity = 8):
    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    new_image = image.copy()
    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0
    return new_image

# Find the height of an object in pixels
def find_obj_pixel_info(image):
    l_row = 9999999
    h_row = 0
    l_pix = 9999999
    h_pix = 0
    for r_index, rows in enumerate(image):
        for p_index, p in enumerate(rows):
            if p == 255:
                if r_index < l_row:
                    l_row = r_index
                elif r_index > h_row:
                    h_row = r_index
                if p_index < l_pix:
                    l_pix = p_index
                elif p_index > h_pix:
                    h_pix = p_index
    return (h_row - l_row, h_pix - l_pix)

def px_to_cm(image, actual_height_in_cm):
    # Find the ref point and find it's height
    # to find the actual height per pixel ratio
    
    # Set hue upper and lower bounds for green
    lower_bound = np.array([0, 255, 255])
    upper_bound = np.array([255, 255, 255])

    # Find the size of the reference point
    ref_mask = cv2.inRange(image, lower_bound, upper_bound)

    ref_mask = remove_isolated_pixels(ref_mask)
    ref_info = find_obj_pixel_info(ref_mask)
    #cv2.imshow('reference', ref_mask)
    ref_height = ref_info[0]
    ref_width = ref_info[1]

    ref_act_height = ref_height_cm / ref_height
    ref_act_width = ref_width_cm / ref_width
    
    return ref_act_height * ref_act_width

# Import the image
image = cv2.imread('test/image4.png')

# Display the normal image
#cv2.imshow('the image', image)

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Calculate the cm per pixel
cm_per_pixel = px_to_cm(hsv, ref_height_cm)

#print("1 pixel = " + str(cm_per_pixel) + " cm2!")

# Set hue upper and lower bounds for brown 'logs'
#lower_bound = np.array([0, 47, 24])
#upper_bound = np.array([21, 250, 255])
lower_bound = np.array([0, 63, 180])
upper_bound = np.array([255, 255, 255])

logs = cv2.inRange(hsv, lower_bound, upper_bound)
logs = remove_isolated_pixels(logs)

#cv2.imshow('logs', logs)

#print(p_count)
# amount of set pixels
amount = np.sum(logs)/ 255

cm_sqrt = amount * cm_per_pixel
#print("Area in cm2: " + str(cm_sqrt))
cm_cubed = cm_sqrt * log_length
#print("Area in cm3: " + str(cm_cubed))
#print("Actual area in m3: " + str(cm_cubed * cm_cb_m_cb))

print("--- %s seconds ---" % (time.time() - start_time))
