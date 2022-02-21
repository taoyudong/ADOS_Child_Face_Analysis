import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import sys


class FaceGenWithPositionInVideo(object):
    """ Generator of face images based on positions of faces in video """

    VIDEO_SUFFIX = ['.mp4', '.MP4']
    COLUMN_FILENAME = 'Name'
    COLUMN_FACE_X = 'FaceRectX'
    COLUMN_FACE_Y = 'FaceRectY'
    COLUMN_FACE_WIDTH = 'FaceRectWidth'
    COLUMN_FACE_HEIGHT = 'FaceRectHeight'

    def __init__(self, mode, video_dir, meta_path, batch_size=32, face_size=(224, 224)):
        """
        Initialize the face image generator

        :param mode: str,
            the format of metadata file
        :param video_dir: str,
            the directories of the videos
        :param meta_path: str,
            the path to the metadata file with face detection results
        """

        self.meta = None
        self.video_files = None
        self.batch_size = batch_size
        self.face_size = face_size

        if mode == 'iMotion':
            # load metadata from iMotion results
            self.COLUMN_FACE_DETECTED = 'NoOfFaces_FACET'
            self._load_iMotion_meta(meta_path)
            print('{} faces from iMotion results have been loaded'.format(len(self.meta)))

            # check video availability
            self._check_video_availability(video_dir)
            print('All required video files are found')

            self.video_dir = video_dir
        elif mode == 'Affectiva':
            # load metadata from Affectiva results
            self.COLUMN_FACE_DETECTED = 'Numberoffaces'
            self.COLUMN_FACE_FEATURE_X = 'feature-x.{featureid}'
            self.COLUMN_FACE_FEATURE_Y = 'feature-y.{featureid}'
            self.NUM_OF_FEATURES = 34
            self._load_Affectiva_meta(meta_path)
            print('{} faces from Affectiva results have been loaded'.format(len(self.meta)))

            # check video availability
            self._check_video_availability(video_dir)
            print('All required video files are found')

            self.video_dir = video_dir
        else:
            raise NotImplementedError('The mode \'{}\' of metadata format has not been implemented'.format(mode))

    def __iter__(self):
        current_video = None
        current_video_capture = None

        faces = list()
        idxs = list()
        for idx, row in self.meta.iterrows():
            if current_video != row[self.COLUMN_FILENAME]:
                current_video = row[self.COLUMN_FILENAME]
                path = self._find_video_path(current_video)
                current_video_capture = cv2.VideoCapture(path)

            # To load the frame with the specified frameNo
            ret = current_video_capture.set(cv2.CAP_PROP_POS_FRAMES, row['FrameNo'])
            if not ret:
                sys.stderr.write('WARNING: Cannot set to load '
                                 'frame {} in file {}\n'.format(row['FrameNo'], current_video))
                continue

            # Read the next frame
            ret, frame = current_video_capture.read()
            if not ret:
                sys.stderr.write('WARNING: Cannot load frame {} in file {}\n'.format(row['FrameNo'], current_video))
                continue

            # Locate the face and preprocessing
            x = int(row[self.COLUMN_FACE_X])
            y = int(row[self.COLUMN_FACE_Y])
            w = int(row[self.COLUMN_FACE_WIDTH])
            h = int(row[self.COLUMN_FACE_HEIGHT])
            if x < 0:
                w -= x
                x = 0
            if y < 0:
                h -= y
                y = 0
            face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
            faces.append(cv2.resize(face, self.face_size))
            idxs.append(idx)

            # Generate one batch
            if len(faces) == self.batch_size:
                yield np.array(idxs), np.array(faces)
                faces = list()
                idxs = list()

        # Generate the remaining
        if len(faces) != 0:
            yield np.array(idxs), np.array(faces)

    def _load_iMotion_meta(self, meta_path):
        """
        Load metadata from iMotion result file (.csv converted from .sav)

        :param meta_path: str,
            the path to the iMotion result file
        :return: list,
            locations of all the faces
        """

        self.meta = pd.read_csv(meta_path)

        # filter out rows without detected face
        self.meta = self.meta[self.meta[self.COLUMN_FACE_DETECTED] != 0]

        # obtain required video files
        self.video_files = set(self.meta[self.COLUMN_FILENAME])

        return self.meta

    def _load_Affectiva_meta(self, meta_path):
        """
        Load metadata from iMotion result file (.txt)

        :param meta_path: str,
            the path to the Affectiva result file
        :return: list,
            locations of all the faces
        """

        self.meta = pd.read_csv(meta_path, sep='\t')

        # filter out rows without detected face
        self.meta = self.meta[self.meta[self.COLUMN_FACE_DETECTED] != 0]

        # Rename column for easier processing
        self.meta = self.meta.rename(index=str, columns={self.COLUMN_FACE_FEATURE_X.format(featureid="")[:-1]:
                                                             self.COLUMN_FACE_FEATURE_X.format(featureid=0),
                                                         self.COLUMN_FACE_FEATURE_Y.format(featureid="")[:-1]:
                                                             self.COLUMN_FACE_FEATURE_Y.format(featureid=0)})

        # Get Face Locations
        self.meta[self.COLUMN_FACE_X] = self.meta[[self.COLUMN_FACE_FEATURE_X.format(featureid=i)
                                                   for i in range(self.NUM_OF_FEATURES)]].min(axis=1)
        self.meta[self.COLUMN_FACE_Y] = self.meta[[self.COLUMN_FACE_FEATURE_Y.format(featureid=i)
                                                   for i in range(self.NUM_OF_FEATURES)]].min(axis=1)
        self.meta[self.COLUMN_FACE_HEIGHT] = self.meta[[self.COLUMN_FACE_FEATURE_Y.format(featureid=i)
                                                        for i in range(self.NUM_OF_FEATURES)]].max(axis=1) - \
                                             self.meta[self.COLUMN_FACE_Y]
        self.meta[self.COLUMN_FACE_WIDTH] = self.meta[[self.COLUMN_FACE_FEATURE_X.format(featureid=i)
                                                       for i in range(self.NUM_OF_FEATURES)]].max(axis=1) - \
                                            self.meta[self.COLUMN_FACE_X]

        # obtain required video files
        self.video_files = set(self.meta[self.COLUMN_FILENAME])

        return self.meta

    def _check_video_availability(self, video_dir):
        """
        Check whether the required videos for analysis are available

        :param video_dir: str,
            the directories of the videos
        :return: bool,
            TRUE if all the required videos are available
        """

        # Ensure metadata has been loaded
        assert self.video_files is not None

        for file in self.video_files:
            found = Path(video_dir).glob('{}.*'.format(file))
            found_suffix = [p.suffix for p in found]
            # Ensure find only one video with recognized suffix
            if len(found_suffix) != 1 or all([suffix not in found_suffix for suffix in self.VIDEO_SUFFIX]):
                raise FileNotFoundError('File \'{}\' does not exist'.format(file))

    def _find_video_path(self, video_name):
        """
        Locate the video file with the given name

        :param video_name: str,
            name of the video file
        :return: str,
            absolute path to the video file
        """
        found = Path(self.video_dir).glob('{}.*'.format(video_name))
        for p in found:
            return str(p)
