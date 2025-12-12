import enum
import numpy as np
import pickle


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

# where this methd going to be implemented 
def load_motion(file):
    with open(file, "rb") as filestream:
        # read the pickle files
        in_dict = pickle.load(filestream)

        loop_mode_val = in_dict["loop_mode"]
        fps = in_dict["fps"]
        frames = in_dict["frames"]

        loop_mode = LoopMode(loop_mode_val)
        frames = np.array(frames, dtype=np.float32)

        motion_data = Motion(loop_mode=loop_mode, fps=fps, frames=frames)

    return motion_data

class Motion():
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
        return

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            loop_mode = self.loop_mode.value
            frames = self.frames.tolist()

            out_dict = {
                "loop_mode": loop_mode,
                "fps": self.fps,
                "frames": frames,
            }
            pickle.dump(out_dict, out_f)
        return
    
    # returns the total number of frames?
    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len

    # it seems that is a class that is meant to store a single motion file and getting it ready for 
    # the preparation part.