"""
This script contains a utility class called Timer.

Dependencies:
    - Time (time)
"""

import time

class Timer:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def get_elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time
    
    def reset(self):
        self.start_time = time.time()