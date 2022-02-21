import cv2
import threading
import numpy as np


class VideoLoader(object):
    """ Generator of face images in the video

    Invoke an additional thread to load the video frames in the file and generate the cropped faces of the same size

    Attributes
    ----------
    path : Path
        Path to the video file
    video : cv2.VideoCapture
        Video file loader
    fd_rst : list of tuple of (int, dict of {str : float})
        Face detection results, each item in the list is a tuple of two elements. The first element is the frame ID
        and the second element is face detection results :class:`dict` with min_x, min_y, width, height, and confidence
    target_size : tuple of (int, int)
        The tuple has two element: (height, width), which represents the size of output face images
    cache_size : int
        Size of the frame pre-loading cache, unit: number of frames
    batch_size : int
        Batch size of the output of each iteration, i.e., number of faces in the batch
    cache : None or dict of {int : np.ndarray}
        Pre-loaded video frames mapping frame ID to the frames. None if has not :func:`reset`
    last_face_loaded : None or int
        Index of :attr:`fd_rst` corresponds to the last pre-loaded frame. None if has not :func:`reset`
    num_face_generated : None or int
        Index of :attr:`fd_rst`, number of faces has been generated. None if has not :func:`reset`
    all_cached : bool
        Whether all the frames needed to generate face images are loaded in memory
    process : threading.Thread
        The thread to pre-load frames from video file
    cache_write_lock : threading.Lock
        The lock preventing concurrent write attempt to the :attr:`cache`
    """

    def __init__(self, video_path, fd_rst, age_rst, target_size, batch_size, frame_cache_size):
        self.path = video_path
        self.video = cv2.VideoCapture(str(video_path))
        self.cache_size = frame_cache_size
        self.batch_size = batch_size

        self.fd_rst = list()
        for frame_id, faces in fd_rst.items():
            for face in faces:
                self.fd_rst.append((frame_id, face))
        self.age_rst = age_rst
        self.target_size = target_size

        self.cache = None
        self.num_face_generated = None
        self.last_face_loaded = None
        self.all_cached = True

        self.process = threading.Thread(target=self._preload_frames)
        self.cache_write_lock = threading.Lock()

    def __iter__(self):
        self.reset()
        return self

    def __len__(self):
        return np.ceil(len(self.fd_rst) / self.batch_size)

    def __next__(self):
        if self.num_face_generated == len(self.fd_rst):
            raise StopIteration
        else:
            # Generate the next batch of face images
            img_batch = list()
            video_frame_batch = list()
            meta_batch = list()
            while len(img_batch) != self.batch_size and self.num_face_generated != len(self.fd_rst):
                # Wait for new frame to be loaded
                face_meta = self.fd_rst[self.num_face_generated]
                frame_id = face_meta[0]
                while not self.all_cached and frame_id not in self.cache.keys():
                    pass
         
                # Filter non child faces
                if int(frame_id) not in self.age_rst:
                    self.num_face_generated += 1
                    if self.num_face_generated == len(self.fd_rst) or self.fd_rst[self.num_face_generated][0] != frame_id:
                        self.cache.pop(frame_id)
                    continue

                # Load the next image
                frame = self.cache[frame_id]
                min_x = max(0, int(round(face_meta[1]['min_x'], 0)))
                min_y = max(0, int(round(face_meta[1]['min_y'], 0)))
                width = min(frame.shape[1]-min_x, int(round(face_meta[1]['width'], 0)))
                height = min(frame.shape[0]-min_y, int(round(face_meta[1]['height'], 0)))
                face = frame[min_y:min_y+height, min_x:min_x+width, :]
                face = self._resize_face(face)
                img_batch.append(face)
                meta_batch.append(face_meta)

                # Zoom out the face for iMotion Expression detection
                center_x = min_x + width / 2
                center_y = min_y + height / 2
                half_target_size = max(width, height)
                space_x_left = center_x - half_target_size
                space_x_right = frame.shape[1] - center_x - half_target_size
                space_y_top = center_y - half_target_size
                space_y_bot = frame.shape[0] - center_y - half_target_size
                if space_x_left + space_x_right >= 0:
                    if space_x_left < 0:
                        space_x_right += space_x_left
                        space_x_left = 0
                    if space_x_right < 0:
                        space_x_left += space_x_right
                        space_x_right = 0
                else:
                    diff = abs(space_x_left + space_x_right)
                    space_y_top += diff / 2
                    space_y_bot += diff / 2
                    space_x_left = 0
                    space_x_right = 0
                if space_y_top + space_y_bot >= 0:
                    if space_y_top < 0:
                        space_y_bot += space_y_top
                        space_y_top = 0
                    if space_y_bot < 0:
                        space_y_top += space_y_bot
                        space_y_bot = 0
                else:
                    diff = abs(space_y_top + space_y_bot)
                    space_x_left += diff / 2
                    space_x_right += diff / 2
                    space_y_top = 0
                    space_y_bot = 0
                space_x_left = int(round(space_x_left, 0))
                space_x_right = int(round(space_x_right, 0))
                space_y_top = int(round(space_y_top, 0))
                space_y_bot = int(round(space_y_bot, 0))
                #print(space_x_left, space_x_right, space_y_top, space_y_bot, frame.shape[1], frame.shape[0])
                #print(space_x_left, frame.shape[1]-space_x_right, space_y_top, frame.shape[0]-space_y_bot)
                video_frame = frame[space_y_top:(frame.shape[0]-space_y_bot), space_x_left:(frame.shape[1]-space_x_right), :]
                #print(frame.shape, video_frame.shape)
                video_frame = self._resize_face(video_frame)
                video_frame_batch.append(frame)#video_frame)

                # Update status
                self.num_face_generated += 1
                if self.num_face_generated == len(self.fd_rst) or self.fd_rst[self.num_face_generated][0] != frame_id:
                    self.cache.pop(frame_id)

            print('{}: {} faces have been generated'.format(self.path.stem, self.num_face_generated))
            return np.array(img_batch), meta_batch, np.array(video_frame_batch)

    @property
    def num_frame_in_cache(self):
        return len(self.cache)

    def _preload_frames(self):
        """ Pre-load video frames

        Load frames from :attr:`self.video` and store into :attr:`self.cache`.
        This function will be executed by :attr:`self.process`

        Raises
        ------
        IOError
            Cannot retrieve a needed frame from the video file
        """

        while not self.all_cached:
            if self.num_frame_in_cache < self.cache_size:
                if self._load_next_frame():
                    self.all_cached = True

    def _load_next_frame(self):
        """ Load a single frame

        Load the next unloaded frame needed for face image generation from :attr:`self.video`
        and store into :attr:`self.cache`

        Returns
        -------
        hitting_end : bool
            Whether all the required video frames are loaded

        Raises
        ------
        IOError
            Cannot retrieve a needed frame from the video file
        """

        # Determine which frame to load
        face_to_load = self.last_face_loaded + 1
        if self.last_face_loaded != -1:
            if face_to_load == len(self.fd_rst):
                return True
            while self.fd_rst[face_to_load][0] == self.fd_rst[self.last_face_loaded][0]:
                face_to_load += 1
                if face_to_load == len(self.fd_rst):
                    return True

        # Load the frame
        frame_to_load = self.fd_rst[face_to_load][0]
        with self.cache_write_lock:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_to_load))
            ret, frame = self.video.read()

            if ret:
                self.cache[frame_to_load] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_face_loaded = face_to_load
                return False
            else:
                # TODO: Handle the error
                #raise IOError('Fail to load the frame {} in the video'.format(face_to_load))
                self.last_face_loaded = face_to_load
                print(IOError('Fail to load the frame {} in the video'.format(face_to_load)))

    def _resize_face(self, face_img):
        """ Resize the face image to the target size

        Parameters
        ----------
        face_img: np.ndarray
            Face image to be resized

        Returns
        -------
        resized_img : np.ndarray
            Resized face image
        """
        return cv2.resize(face_img, self.target_size)

    def reset(self):
        """ Reset the face image generator and ready to generate images based on the current configuration """

        with self.cache_write_lock:
            # Attempt to terminate the previous pre-loading process gracefully
            self.all_cached = True
            if self.process.is_alive():
                del self.process

            # Re-initiate the generator
            self.cache = dict()
            self.last_face_loaded = -1  # Indicate to load the first required frame
            self.num_face_generated = 0
            self.all_cached = False

            # Restart the pre-loading
            self.process = threading.Thread(target=self._preload_frames)
            self.process.start()
