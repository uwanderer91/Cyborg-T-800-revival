from collections import deque
import numpy as np

class FrameStacker:
    def __init__(self, frame_stack=2, frame_shape=(64, 64)):
        self.frame_stack = frame_stack
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=frame_stack)
        
    def reset(self, initial_frame):
        self.frames.clear()
        zero_frame = np.zeros_like(initial_frame)
        for _ in range(self.frame_stack - 1):
            self.frames.append(zero_frame)
        self.frames.append(initial_frame)
        return self.get_stacked_frames()
    
    def append(self, frame):
        self.frames.append(frame)
        return self.get_stacked_frames()
    
    def get_stacked_frames(self):
        stacked = np.array(self.frames)
        return stacked