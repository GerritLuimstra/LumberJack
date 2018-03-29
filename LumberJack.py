# Import the required libraries
import numpy as np
import cv2

class LumberJack:

    ref_image = np.array([])
    image = np.array([])

    def __init__(self, ref_height = 80, ref_width = 40, log_length = 300):
        self.ref_height = ref_height
        self.ref_width = ref_width
        self.log_length = log_length

    def _remove_isolated_pixels(self, image):
        return image

    """ Extracts the reference points from the image """
    def _extract_ref(self):
        
        # Set hue upper and lower bounds for red
        lower_bound = np.array([0, 255, 255])
        upper_bound = np.array([255, 255, 255])

        ref_image = cv2.inRange(self.hsv, lower_bound, upper_bound)
        ref_image = self._remove_isolated_pixels(ref_image)
        self.ref_image = ref_image
        

    """ Finds the dimensions of the reference point """
    def _ref_dimensions(self):
        l_row = 9999999
        h_row = 0
        l_pix = 9999999
        h_pix = 0
        for r_index, rows in enumerate(self.ref_image):
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

    """ Finds the pixel/cm ratio by using the reference point """
    def _px_to_cm(self):
        # Find the ref point and find it's height
        # to find the actual height per pixel ratio
        self._extract_ref()
        ref_dimensions = self._ref_dimensions()

        ref_height = ref_dimensions[0]
        ref_width = ref_dimensions[1]

        ref_act_height = self.ref_height / ref_height
        ref_act_width = self.ref_width / ref_width
    
        return ref_act_height * ref_act_width

    def estimate(self, image):
        self.image = image
        
        # Convert the image to HSV
        self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate the cm per pixel
        cm_per_pixel = self._px_to_cm()

        # Set hue upper and lower bounds for brown 'logs'
        lower_bound = np.array([0, 47, 24])
        upper_bound = np.array([21, 250, 255])

        logs = cv2.inRange(self.hsv, lower_bound, upper_bound)
        logs = self._remove_isolated_pixels(logs)

        amount = np.sum(logs)/ 255

        cm_sqrt = amount * cm_per_pixel
        cm_cubed = cm_sqrt * self.log_length
        m_cubed = cm_cubed * 1e-6

        return m_cubed


image = cv2.imread('test/anothertest.png')
lumberjack = LumberJack()
estimate = lumberjack.estimate(image)
print(estimate)














