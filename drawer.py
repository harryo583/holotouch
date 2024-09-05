"""
This script contains utility functions for drawing graphical elements on image frames using OpenCV.

Dependencies:
    - OpenCV (cv2)
    - Math (math)
"""

import math
import cv2

def draw_dashed_line(frame, start, end, color, thickness=3, dash_length=5, gap_length=10):
    """
    Draws a dashed line on the given frame.
    
    Arguments:
        frame (numpy.ndarray): The image frame to draw on.
        start (tuple): The (x, y) coordinates of the start point of the line.
        end (tuple): The (x, y) coordinates of the end point of the line.
        color (tuple): The color of the line in BGR format.
        thickness (int, optional): The thickness of the dashed line. Default is 3.
        dash_length (int, optional): The length of each dash. Default is 5.
        gap_length (int, optional): The length of the gap between dashes. Default is 10.
    """
    
    if not (isinstance(start, tuple) and len(start) == 2):
        raise ValueError("start must be a 2-element tuple")
    if not (isinstance(end, tuple) and len(end) == 2):
        raise ValueError("end must be a 2-element tuple")
    
    x1, y1 = start
    x2, y2 = end
    
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if length == 0: return
    direction = ((x2 - x1) / length, (y2 - y1) / length)
    num_dashes = int(length / (dash_length + gap_length))
    
    for i in range(num_dashes):
        dash_start = (int(x1 + i * (dash_length + gap_length) * direction[0]),
                      int(y1 + i * (dash_length + gap_length) * direction[1]))
        dash_end = (int(dash_start[0] + dash_length * direction[0]),
                    int(dash_start[1] + dash_length * direction[1]))
        cv2.line(frame, dash_start, dash_end, color, thickness)


def draw_slider(frame, volume_level):
    """
    Draws a slider on the given frame to represent the volume level.

    Arguments:
        frame (numpy.ndarray): The image frame to draw on.
        volume_level (int): The current volume level, expected to be between 0 and 100.
    """
    height, width, _ = frame.shape
    slider_width = 500
    slider_height = 40
    slider_x = 50
    slider_y = height - 150
    
    cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_width, slider_y + slider_height), (200, 200, 200), -1)
    
    filled_width = int((volume_level / 100) * slider_width)
    cv2.rectangle(frame, (slider_x, slider_y), (slider_x + filled_width, slider_y + slider_height), (0, 255, 0), -1)
    
    cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_width, slider_y + slider_height), (0, 0, 0), 2)
