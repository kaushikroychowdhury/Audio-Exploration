from tensorflow.keras.models import load_model
import numpy as np
import os
import librosa
import math

sample_rate = 22050
json_path = "data1.json"
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = sample_rate * TRACK_DURATION

def predict(model, X):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    """

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Predicted label: {}".format(predicted_index))


def preprocess (n_fft=2048, hop_length=512, num_mfcc=13, num_segments=5):
    data_dict = {
        "mfcc":[]
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for subdir, dirs, files in os.walk(r'sounds'):
        for filename in files:
            file = subdir + os.sep + filename
            print(file)

            # load audio file
            signal, sr = librosa.load(file, sr=sample_rate)

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

        X = np.array(data_dict["mfcc"])
        return X


if __name__  == "__main__":

    X = preprocess()

    model = load_model('RNN_model.h5')

    predict(model, X)
