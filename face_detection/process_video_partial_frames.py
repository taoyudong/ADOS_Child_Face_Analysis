import numpy as np
import argparse
import time
from pathlib import Path
from Video import VideoLoader
from mtcnn.mtcnn import MTCNN

parser = argparse.ArgumentParser(description='Face detection for videos with Dual Shot Face Detector')
parser.add_argument('video_path', type=str, help='Location of test video')
parser.add_argument('start_frame', type=int)
parser.add_argument('end_frame', type=int)
args = parser.parse_args()

def write_to_txt(f, det , event , im_name):
    f.write('{:s}\n'.format(event + '/' + im_name))
    f.write('{:d}\n'.format(len(det)))
    for i in range(len(det)):
        xmin = det[i]['box'][0]
        ymin = det[i]['box'][1]
        width = det[i]['box'][2] 
        height = det[i]['box'][3]
        score = det[i]['confidence'] 
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, width, height, score))

def test_video():
    # load net
    net = MTCNN()

    # load data
    loader = VideoLoader(args.video_path, 1, 100)
    video_name = Path(args.video_path).stem
    frame_id = 0

    st = time.time()
    for idx, img in enumerate(loader): #, img1_batch, img2_batch, img3_batch in loader:
        if idx % 1000 == 0:
            print(idx)
        if idx < args.start_frame:
            continue
        if idx > args.end_frame:
            break
        loading_time = time.time() - st

        det = net.detect_faces(img)   
        processing_time = time.time() - st - loading_time

        with open('/home/mitch-b/dmis-research/Yudong/2019/Psychology/FaceDetection/MTCNN/results/{}_results_{}_{}.txt'.format(video_name, args.start_frame, args.end_frame), 'a') as f:
            write_to_txt(f, det, video_name, str(frame_id))
        writing_time = time.time() - st - processing_time
        print('Time: L={}, P={}, W={}, A={}'.format(loading_time, processing_time, writing_time, loading_time + processing_time + writing_time))
        frame_id += 1 
        st = time.time()
            

if __name__ == '__main__':
    test_video()
