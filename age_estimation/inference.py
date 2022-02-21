import argparse
from utils.data_generator import FaceGenWithPositionInVideo
import numpy as np
from models.DEX_model import get_DEX_model


def parse_args():
    parser = argparse.ArgumentParser(description='This script infer age of faces images based on various input formats',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str,
                        help='The mode of input format: \'iMotion-Video\'')
    parser.add_argument('data_dir', type=str,
                        help='The directory of face images or videos')
    parser.add_argument('weight_path', type=str,
                        help='The path to the pre-trained model file with weights')
    parser.add_argument('-m', '--metadata', type=str,
                        help='The path to the metadata file with face detection results')
    parser.add_argument('--model', type=str, default='ResNet50',
                        help='The type of model: \'Resnet50\'')
    args = parser.parse_args()
    return args


def main_video(args, meta_mode):
    # create generator of faces in videos based on face detection results of meta_mode
    face_gen = FaceGenWithPositionInVideo(meta_mode, args.data_dir, args.metadata)

    # create age estimation model
    model = get_DEX_model(args.weight_path)

    # estimate age for all the faces
    expected_ages = np.arange(0, 101).reshape(101, 1)
    for face_idxs, face_imgs in face_gen:
        probs = model.predict(face_imgs)
        predicted_ages = probs.dot(expected_ages).flatten()
        for idx, predicted_age in zip(face_idxs, predicted_ages):
            print('{},{}'.format(idx, predicted_age))


if __name__ == '__main__':
    args = parse_args()
    if args.mode in ['iMotion-Video', 'Affectiva-Video']:
        main_video(args, args.mode.split('-')[0])
    else:
        raise NotImplementedError('The mode \'{}\' of input format has not been implemented'.format(args.mode))
