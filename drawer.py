"""
This script contains utility functions for drawing graphical elements on image frames using OpenCV.
"""

import math
import cv2
from typing import Tuple, Generator

def dashed_line_segments(start: Tuple[int, int], end: Tuple[int, int],
                           dash_length: int, gap_length: int) -> Generator[Tuple[Tuple[int, int], Tuple[int, int]], None, None]:
    """
    Generator yielding segments for a dashed line.
    """
    x1, y1 = start
    x2, y2 = end
    length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if length == 0:
        return
    direction = ((x2 - x1) / length, (y2 - y1) / length)
    num_dashes = int(length / (dash_length + gap_length))
    for i in range(num_dashes):
        dash_start = (int(x1 + i * (dash_length + gap_length) * direction[0]),
                      int(y1 + i * (dash_length + gap_length) * direction[1]))
        dash_end = (int(dash_start[0] + dash_length * direction[0]),
                    int(dash_start[1] + dash_length * direction[1]))
        yield dash_start, dash_end

def draw_dashed_line(frame, start: Tuple[int, int], end: Tuple[int, int],
                     color: Tuple[int, int, int], thickness: int = 3,
                     dash_length: int = 5, gap_length: int = 10) -> None:
    """
    Draws a dashed line on the given frame.
    """
    if not (isinstance(start, tuple) and len(start) == 2):
        raise ValueError("start must be a 2-element tuple")
    if not (isinstance(end, tuple) and len(end) == 2):
        raise ValueError("end must be a 2-element tuple")
    
    for dash_start, dash_end in dashed_line_segments(start, end, dash_length, gap_length):
        cv2.line(frame, dash_start, dash_end, color, thickness)

def draw_slider(frame, volume_level: int) -> None:
    """
    Draws a slider on the given frame to represent the volume level.
    """
    height, width, _ = frame.shape
    slider_width = 500
    slider_height = 40
    slider_x = 50
    slider_y = height - 150
    
    cv2.rectangle(frame, (slider_x, slider_y),
                  (slider_x + slider_width, slider_y + slider_height), (200, 200, 200), -1)
    
    filled_width = int((volume_level / 100) * slider_width)
    cv2.rectangle(frame, (slider_x, slider_y),
                  (slider_x + filled_width, slider_y + slider_height), (0, 255, 0), -1)
    
    cv2.rectangle(frame, (slider_x, slider_y),
                  (slider_x + slider_width, slider_y + slider_height), (0, 0, 0), 2)
