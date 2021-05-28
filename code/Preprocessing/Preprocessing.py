import json
import math
import librosa
import numpy as np
import os


sample_rate = 22050
dataset_path = "sounds"
json_path = "data1.json"
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = sample_rate * TRACK_DURATION

def preprocess (dataset_path, json_path, n_fft=2048, hop_length=512, num_mfcc=13, num_segments=5):
    data_dict = {
        "mappings":[],
        "labels":[],
        "mfcc":[]
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


    for subdir, dirs, files in os.walk(dataset_path):
        for filename in files:
            filepath = subdir + os.sep + filename
            label = filepath[7:-len('.00000.wav')]
            #label = filepath[10:-len('.00000.wav')]
            if label not in data_dict["mappings"]:
                data_dict["mappings"].append(label)

            # load audio file
            signal, sr = librosa.load(filepath, sr=sample_rate)

            # process all segments of audio file
            for d in range(num_segments):

                # calculate start and finish sample for current segment
                start = samples_per_segment * d
                finish = start + samples_per_segment

                # extract mfcc
                mfcc = librosa.feature.mfcc(signal[start:finish], sr, n_mfcc=num_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T

                # store only mfcc feature with expected number of vectors
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    data_dict["mfcc"].append(mfcc.tolist())
                    data_dict["labels"].append(data_dict["mappings"].index(label))
#                    print("{}, segment:{}".format(filepath, d + 1))


            # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data_dict, fp, indent=4)

    xt = np.array(data_dict['mfcc'])
    # print(xt[0])
    print(xt[0].shape)
    print(xt[1].shape)
    print(xt[2].shape)
    print(xt[3].shape)
    print(xt[4].shape)

preprocess(dataset_path, json_path)

