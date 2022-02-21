import os
from pathlib import Path
import numpy as np
import cv2
import data_loader
import pandas as pd
from skvideo.io import FFmpegWriter
from model import get_DEX_model
import sys


def load_face_detection_result(rst_path):
    """ Load the face detection results of an first-person ADOS video

    Load the face detection results of a first-person ADOS video from a plain text file into a :class:`dict`.
    The results of each frame are attached successively in the file.

    Notes
    -----
    For each frame, the result has *2+<number_of_face_detected>* lines and is organized in the following way::

        <video_file_name>/<frame_id>
        <number_of_face_detected>
        <face_1_location_x> <face_1_location_y> <face_1_width> <face_1_height> <face_1_confidence_score>
        ...
        <face_k_location_x> <face_k_location_y> <face_k_width> <face_k_height> <face_k_confidence_score>

    where each field is specified in the followings:

    * **video_file_name** (*string*): filename of the video
    * **frame_id** (*int*): zero-based frame index
    * **number_of_face_detected** (*int*): number of faces detected in the frame, no less than 0
    * **face_k_location_x, face_k_location_y** (*float*): location of left-top corner of the k-th face in pixel
    * **face_k_width, face_k_height** (*float*): the size of the k-th face in pixel
    * **face_k_confidence_score** (*float*): the confidence in the k-th detected region being a face

    Parameters
    ----------
    rst_path : Path
        The input file of face detection result

    Returns
    -------
    face_detection_results : dict of {int : list of {str : float}}
        Mapping *<frame_id>* to a :class:`list` of face detection results. Each item in the list is a :class:`dict`
        with fields min_x, min_y, width, height, and confidence.

    Raises
    ------
    FileNotFoundError
        The face detection result file cannot be located
    IOError
        The face detection result file cannot be accessed or the format of the file is not as expected
    """

    STATES = {'FRAME_ID': 0, 'NUM_FACES': 1, 'FACE_META': 2}

    if not rst_path.exists():
        raise FileNotFoundError('Cannot find face detection result at \'{}\''.format(rst_path))
    elif not rst_path.is_file():
        raise IOError('Face detection result at \'{}\' is not a file'.format(rst_path))

    state = STATES['FRAME_ID']
    filename_check = None
    frame_id = None
    num_face = None
    num_face_loaded = None
    fd_rst = dict()
    for line in open(str(rst_path), 'r'):
        line = line.strip()

        if state == STATES['FRAME_ID']:
            # load the first line of each frame's result
            filename, frame_id = line.split('/')
            if filename_check is None:
                filename_check = filename
            elif filename != filename_check:
                raise IOError('Multiple filenames (\'{}\' and \'{}\') are detected '
                              'in face detection result at \'{}\''.format(filename, filename_check, rst_path))
            if frame_id not in fd_rst.keys():
                fd_rst[frame_id] = list()
            else:
                raise IOError('Multiple entries are found for frameID \'{}\''
                              'in the face detection result at \'{}\''.format(frame_id, rst_path))
            state = STATES['NUM_FACES']
        elif state == STATES['NUM_FACES']:
            # load the second line of each frame's result
            num_face = int(line)
            if num_face == 0:
                state = STATES['FRAME_ID']
            else:
                num_face_loaded = 0
                state = STATES['FACE_META']
        elif state == STATES['FACE_META']:
            # load the rest of each frame's result
            data = list(map(float, line.split(' ')))
            fd_rst[frame_id].append({
                'min_x': data[0], 'min_y': data[1],
                'width': data[2], 'height': data[3],
                'confidence': data[4]})
            num_face_loaded += 1
            if num_face_loaded == num_face:
                state = STATES['FRAME_ID']
        else:
            raise ValueError('Unexpected state \'{}\''.format(state))

    return fd_rst


def load_age_estimation_result(rst_path):
    if not rst_path.exists():
        raise FileNotFoundError('Cannot find face detection result at \'{}\''.format(rst_path))
    elif not rst_path.is_file():
        raise IOError('Face detection result at \'{}\' is not a file'.format(rst_path))
    data = pd.read_csv(rst_path)
    age_rst = dict()
    for idx, row in data.iterrows():
        if int(row['FrameID']) in age_rst:
            age_rst[int(row['FrameID'])].append(row.to_dict())
        else:
            age_rst[int(row['FrameID'])] = [row.to_dict()]
    return age_rst


def infer_target_face_size(fd_rst, max_shrink_ratio=1.5):
    """ Adaptively infer the target size of the output face video

    Parameters
    ----------
    fd_rst : dict of {int : list of {str : float}}
        Face detection results mapping *<frame_id>* to a :class:`list` of face detection results.
        Each item in the list is a :class:`dict` with min_x, min_y, width, height, and confidence
    max_shrink_ratio : float
        Maximal allowed original to target figure ratio of the shape

    Returns
    -------
    face_height : int
        the target height of the output face video
    face_width : int
        the target width of the output face video
    """

    ave_width = 0.0
    ave_height = 0.0
    max_width = 0.0
    max_height = 0.0
    num_faces = 0
    for frame_id, faces in fd_rst.items():
        for face in faces:
            ave_height += face['height']
            ave_width += face['width']
            if face['height'] > max_height:
                max_height = face['height']
            if face['width'] > max_width:
                max_width = face['width']
            num_faces += 1
    ave_width /= num_faces
    ave_height /= num_faces

    # force the shrink ratio no larger than the the given threshold
    target_width = max_width / max_shrink_ratio if max_width / max_shrink_ratio < ave_width else ave_width
    target_height = max_height / max_shrink_ratio if max_height < ave_height * max_shrink_ratio else ave_height
    return int(round(target_height, 0)), int(round(target_width, 0))


def generate_child_faces(video_path, fd_rst, age_rst, threshold=15.0, batch_size=32, cache_size=320):
    """ Generate all child faces in the original video reshaped to the same size

    Parameters
    ----------
    video_path : Path
        Path to the first-person ADOS video file
    fd_rst : dict of {int : list of {str : float}}
        Face detection results mapping *<frame_id>* to a :class:`list` of face detection results.
        Each item in the list is a :class:`dict` with min_x, min_y, width, height, and confidence
    threshold : float
        A face with estimated age smaller than the :attr:`threshold` is regarded as a child's face, default 15.0
    batch_size : int
        Batch size of the age estimation, default 32
    cache_size : int
        Maximum number of frames cached in the memory, default 320

    Returns
    -------
    child_faces : ny.ndarray
        A three-dimensional array of (num_faces, face_height, face_width)
    face_meta : dict of {int : dict of {str: float}}}
        Mapping *<face_id>* to a :class:`dict`, which contains the metadata and face detection and
        age estimation results of the output faces.
        Each result is a :class:`dict` with fields frame_id, min_x, min_y, width, height, confidence and estimated_age
    """

    weight_path = Path(os.path.abspath(__file__)).parent.parent / \
        'model_weights/age_only_resnet50_weights.061-3.300-4.410.hdf5'
    model = get_DEX_model(weight_path)
    loader = data_loader.AgeVideoLoader(video_path, fd_rst, age_rst, model.input_shape[1:3], batch_size, cache_size)

    expected_ages = np.arange(0, 101).reshape(101, 1)
    child_faces = list()
    face_meta = dict()
    try:
        for idx, (img_batch, meta_batch, video_frame_batch) in enumerate(loader):
            for face, meta in zip(img_batch, meta_batch):
                if int(meta[0]) in age_rst:
                    found = False
                    for r in age_rst[int(meta[0])]:
                        check = (abs(float(meta[1]['min_x']) - r['Min_X']) < 1e-4) & (abs(float(meta[1]['min_y']) - r['Min_Y']) < 1e-4) & (abs(float(meta[1]['height']) - r['Height']) < 1e-4) & (abs(float(meta[1]['width']) - r['Width']) < 1e-4)
                        if check:
                            found = True
                    if found:
                        child_faces.append(face)
    except KeyError as e:
        print('Err:', e)
        
    return child_faces


def output_video(child_faces, output_path, overwrite=True):
    """ Merging all child faces into a .mpg video file

    Parameters
    ----------
    child_faces : np.ndarray
        A numpy array of (num_faces, face_height, face_width, 3)
    output_path : Path
        Path to the output video file
    overwrite : bool
        True if overwriting the existing file at :attr:`output_path`, default True

    Returns
    -------
    success : bool
        Whether the video file is generated

    Raises
    ------

    """

    if output_path.exists() and not overwrite:
        return False

    # Assume all the faces with the same size
    out = FFmpegWriter(str(output_path), inputdict={'-r': '20'}, outputdict={'-r': '20'})

    for face in child_faces:
        out.writeFrame(face)

    out.close()
    return True


def output_meta(face_meta, output_path, overwrite=True):
    """ Output face metadata into a .csv file

    Parameters
    ----------
    face_meta : dict of {int : dict of {str: float}}}
        Mapping *<face_id>* to a :class:`dict`, which contains the metadata and face detection and
        age estimation results of the output faces.
        Each result is a :class:`dict` with fields frame_id, min_x, min_y, width, height, confidence and estimated_age
    output_path : Path
        Path to the output face metadata file
    overwrite : bool
        True if overwriting the existing file at :attr:`output_path`, default True

    Returns
    -------
    success : bool
        Whether the video file is generated
    """

    if output_path.exists() and not overwrite:
        return False

    with open(str(output_path), 'w') as fout:
        fout.write('FaceID,FrameID,Min_X,Min_Y,Width,Height,Confidence,EstimatedAge\n')
        for face_id, info in face_meta.items():
            fout.write('{},{},{},{},{},{},{},{}\n'.format(face_id, info['frame_id'],
                                                          info['min_x'], info['min_y'],
                                                          info['width'], info['height'],
                                                          info['confidence'], info['estimated_age']))
    return True


if __name__ == '__main__':
    # Path to the Face Detection results
    # result_dir = Path('/Users/yudongtao/Research/ADOS_Analysis/data/face_detection_results')
    # result_dir = Path('/home/charley/Research/Yudong/FaceDetectionRST_MTCNN')
    
    # Path to the Age Estimation results
    age_result_dir = Path('/home/mitch-b/dmis-research/Yudong/2019/Psychology/Expression/ADOS_Analysis/data/wo_distort/')

    # Path to the ADOS first-person videos
    # video_dir = Path('/Users/yudongtao/Research/ADOS_Analysis/data/videos')
    video_dir = Path('/home/charley/Research/Yudong/Videos/')
    video_list = [video_dir / a for a in sys.argv[1:]]
    print(video_list)

    # Path to the outputs
    # output_dir = Path('/Users/yudongtao/Research/ADOS_Analysis/data/child_face_videos')
    output_dir = Path('/home/mitch-b/dmis-research/Yudong/2019/Psychology/Expression/ADOS_Analysis/data/wo_distort')
    weight_path = Path(os.path.abspath(__file__)).parent.parent / \
        'model_weights/age_only_resnet50_weights.061-3.300-4.410.hdf5'
    model = get_DEX_model(weight_path)

    for file in sorted(video_list):
        print('Processing {} ...'.format(file.stem))
        if file.suffix in ['.mp4', '.MP4']:
            output_video_path = output_dir / '{}.avi'.format(file.stem)
            if output_video_path.exists():
                print('{} Skipped'.format(file.stem))
                continue
            # locate face detection result
            # fd_result_path = result_dir / '{}_results.txt'.format(file.stem)
            # if not fd_result_path.exists():
            #     fd_result_path = result_dir / '{}_results.txt'.format(file.stem.replace('_ADOS',''))
            # if not fd_result_path.exists():
            #     print('{} Not Found'.format(fd_result_path))
            # else:
            #     fd_result = load_face_detection_result(fd_result_path)
            age_result_path = age_result_dir / '{}.csv'.format(file.stem)
            if not age_result_path.exists():
                age_result_path = age_result_dir / '{}.csv'.format(file.stem.replace('_ADOS',''))
            if not age_result_path.exists():    
                print('{} Not Found'.format(age_result_path))
            else:
                # fd_result = load_face_detection_result(fd_result_path)
                age_result = load_age_estimation_result(age_result_path)
                loader = cv2.VideoCapture(str(file))
                child_faces = list()
                for fid, meta in age_result.items():
                    print('Processing {} ...'.format(fid))
                    loader.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ret, frame = loader.read()
                    if not ret:
                        print(IOError('Fail to load the frame {} in the video'.format(face_to_load)))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for m in meta:
                        min_x = max(0, int(round(m['Min_X'], 0)))
                        min_y = max(0, int(round(m['Min_Y'], 0)))
                        width = min(frame.shape[1]-min_x, int(round(m['Width'], 0)))
                        height = min(frame.shape[0]-min_y, int(round(m['Height'], 0)))
                        #face = frame[min_y:min_y+height, min_x:min_x+width, :]
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
                        face = frame[space_y_top:(frame.shape[0]-space_y_bot), space_x_left:(frame.shape[1]-space_x_right), :]
                        face = cv2.resize(face, model.input_shape[1:3])
                        child_faces.append(face)

                # filter out adult faces with age estimation on face
                #child_faces = generate_child_faces(file, fd_result, age_result)

                # generate outputs
                if len(child_faces) > 0:
                    output_video_path = output_dir / '{}.avi'.format(file.stem)
                    output_video(child_faces, output_video_path)
                    print('Child face \'{}\' processed'.format(file.stem))
                else:
                    print('No child face `{}` detected'.format(file.stem))
