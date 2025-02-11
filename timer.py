"""
This script contains a utility class called Timer.
"""

import time

class Timer:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time
    
    def reset(self):
        self.start_time = time.time()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optionally, cleanup or reset
        self.reset()
