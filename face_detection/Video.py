import cv2
import threading
import numpy as np

class VideoLoader(object):
    """ Load the video frame and return the paired consecutive frames for Face Detection DSFD Model"""

    def __init__(self, path, batch_size, frame_cache_size):
        self.video = cv2.VideoCapture(str(path))
        self.cache_size = frame_cache_size
        self.batch_size = batch_size

        #self.transform = TestBaseTransform((103, 117, 123))
        self.factor = 2

        self.cache = None
        self.num_frame_loaded = None
        self.num_frame_processed = None
        self.loaded = True

        self.process = threading.Thread(target=self._preload_frames)
        self.cache_write_lock = threading.Lock()

    def _preload_frames(self):
        while not self.loaded:
            if self.num_frame_loaded - self.num_frame_processed < self.cache_size:
                self._load_next_frame()

    def _load_next_frame(self):
        with self.cache_write_lock:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.num_frame_loaded)
            ret, frame = self.video.read()

            if ret:
                self.cache[self.num_frame_loaded+1] =frame  
                self.num_frame_loaded += 1
            else:
                self.loaded = True

    def reset(self):
        with self.cache_write_lock:
            self.loaded = True  # Attempt to complete the preloading
            if self.process.is_alive():
                del self.process  # Force to finish the thread

            self.cache = dict()
            self.num_frame_loaded = 0
            self.num_frame_processed = 0
            self.loaded = False
           
            self.max_im_shrink = None
            self.shrink1 = None
            self.shrink2 = None
            self.shrink3 = None

            self.process = threading.Thread(target=self._preload_frames)

            self.process.start()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.loaded and self.num_frame_processed == self.num_frame_loaded:
            raise StopIteration
        else:
            while self.num_frame_processed >= self.num_frame_loaded:
                pass  # wait for new frame to be loaded

            rst = self.cache[self.num_frame_processed+1] 
            self.cache.pop(self.num_frame_processed+1)
            self.num_frame_processed += 1 
            return np.array(rst)


if __name__ == '__main__':
    import time
    loader = VideoLoader('/home/charley/Research/Yudong/Videos/ADOS1047_E2_ADOS_091319.mp4', 16, 1000)
    st = time.time() 
    for all_frames in loader:
        end = time.time()
        print([frames.shape for frames in all_frames], len(loader.cache), end-st)
        st = time.time()

