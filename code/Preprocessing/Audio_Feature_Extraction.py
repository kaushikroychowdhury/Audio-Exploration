import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os
import pandas as pd
sns.set()

file = "sounds/blues.00000.wav"
signal, sr = librosa.load(file)


### Basics of Audio Feature Transformations .. ( ADC Analog Digital Conversion ) .......................................
# Waveform
def Waveform_gen (signal) :
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(b=None)
    print(plt.show())

# Fast Fourier Transform
def fourierTransform (signal) :
    ft = np.fft.fft(signal)
    magnitude = np.abs(ft)
    frequency = np.linspace(0,sr,len(magnitude))
    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(magnitude)/2)]
    plt.figure(figsize=(12, 6))
    plt.plot(left_frequency, left_magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid(b = None)
    print(plt.show())

# Short Time Fourier Transform
def STFT (signal, n_fft=2048, hop_length=512):
    stft = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=n_fft))
    log_stft = librosa.amplitude_to_db(stft)
    img = librosa.display.specshow(log_stft)
    plt.title("Power Spectrum")
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.colorbar(img, format="%+2.0f dB")
    plt.grid(b=None)
    print(plt.show())

# Mel Frequency Cepstral Coefficient
def MFCCs (signal, sr, n_fft=2048, hop_length=512, n_mfcc = 13) :
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc =n_mfcc)
    img = librosa.display.specshow(mfcc)
    plt.title("MFCC")
    plt.xlabel("Time")
    plt.ylabel("MFCCs")
    plt.colorbar(img)
    plt.grid(b=None)
    print(plt.show())

Waveform_gen(signal)
fourierTransform(signal)
STFT(signal)
MFCCs(signal,sr)

###.....................................................................................................................
### FEATURE EXTRACTION ..........

#spectral features ..

def spectral_features (signal, sr, n_fft=2048, hop_length=512, n_mfcc=13) :
    # Chromagram
    chroma_stft = librosa.feature.chroma_stft(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # Constant-Q Transform
    chroma_cqt = librosa.feature.chroma_cqt(signal, sr=sr, hop_length=hop_length)
    chroma_cqt_mean = np.mean(chroma_cqt)
    chroma_cqt_var = np.var(chroma_cqt)

    # Chroma Energy Normalized
    chroma_cens = librosa.feature.chroma_cens(signal, sr=sr, hop_length=hop_length)
    chroma_cens_mean = np.mean(chroma_cens)
    chroma_cens_var = np.var(chroma_cens)

    # mel-scaled spectrogram
    melspectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft= n_fft, hop_length=hop_length)
    melspectrogram_mean = np.mean(melspectrogram)
    melspectrogram_var = np.var(melspectrogram)

    # MFCCs
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc)
    mfcc_var = np.var(mfcc)

    # root-mean-square (RMS) value for each frame
    rms = librosa.feature.rms(signal, hop_length=hop_length)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Compute spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_centroid_mean = np.mean(spec_centroid)
    spec_centroid_var = np.var(spec_centroid)

    # Compute spectral bandwidth
    spec_bandwith = librosa.feature.spectral_bandwidth(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_bandwith_mean = np.mean(spec_bandwith)
    spec_bandwith_var = np.var(spec_bandwith)

    # Compute spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_contrast_mean = np.mean(spec_contrast)
    spec_contrast_var = np.var(spec_contrast)

    # Compute spectral flatness
    spec_flatness = librosa.feature.spectral_flatness(signal, n_fft=n_fft, hop_length=hop_length)
    spec_flatness_mean = np.mean(spec_flatness)
    spec_flatness_var = np.var(spec_flatness)

    # Compute spectral roll-off
    spec_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_rolloff_mean = np.mean(spec_rolloff)
    spec_rolloff_var = np.var(spec_rolloff)

    # Compute tonal centroid features
    tonnetz = librosa.feature.tonnetz(signal, sr=sr)
    tonnetz_mean = np.mean(tonnetz)
    tonnetz_var = np.var(tonnetz)

    # Compute Zero-Crossing-Rate
    crossing_rate = librosa.feature.zero_crossing_rate(signal, hop_length= hop_length)
    crossing_rate_mean = np.mean(crossing_rate)
    crossing_rate_var = np.var(crossing_rate)

    spec_features = [chroma_stft_mean, chroma_stft_var, chroma_cens_mean, chroma_cens_var, chroma_cqt_mean,
                     chroma_cqt_var, melspectrogram_mean, melspectrogram_var, mfcc_mean, mfcc_var, rms_mean,
                     rms_var, spec_bandwith_mean, spec_bandwith_var, spec_centroid_mean, spec_centroid_var,
                     spec_contrast_mean, spec_contrast_var, spec_flatness_mean, spec_flatness_var, spec_rolloff_mean,
                     spec_rolloff_var, tonnetz_mean, tonnetz_var, crossing_rate_mean, crossing_rate_var]

    return spec_features

# print(spectral_features(signal, sr))

### Creating the Dataset  ..............................................................................................
Start = time.time()
Features = []
# for subdir, dirs, files in os.walk(r'sounds'):
#     for filename in files:
#         filepath = subdir + os.sep + filename
#         label = filepath[12:-len('.00000.wav')]
#         print(label)
#         signal, sr = librosa.load(filepath)
#         l = spectral_features(signal, sr)
#         l.append(label)
#         Features.append(l)
#
# df = pd.DataFrame(columns=['chroma_stft_mean', 'chroma_stft_var', 'chroma_cens_mean', 'chroma_cens_var', 'chroma_cqt_mean',
#                      'chroma_cqt_var', 'melspectrogram_mean', 'melspectrogram_var', 'mfcc_mean', 'mfcc_var', 'rms_mean',
#                      'rms_var', 'spec_bandwith_mean', 'spec_bandwith_var', 'spec_centroid_mean', 'spec_centroid_var',
#                      'spec_contrast_mean', 'spec_contrast_var', 'spec_flatness_mean', 'spec_flatness_var', 'spec_rolloff_mean',
#                      'spec_rolloff_var', 'tonnetz_mean', 'tonnetz_var', 'crossing_rate_mean', 'crossing_rate_var', 'labels'],
#                       data=Features)
#
# ### Conversion of Dataframe into csv file ..
#
# df.to_csv("Audio_Features.csv")

#
for subdir, dirs, files in os.walk(r'sounds'):
    for filename in files:
        filepath = subdir + os.sep + filename
        print(filepath)
        label = filepath[7:-len('.00000.wav')]
        signal, sr = librosa.load(filepath)
        l = spectral_features(signal, sr)
        print('done 1')
        l.append(label)
        Features.append(l)
        print('done 2')

df = pd.DataFrame(columns=['chroma_stft_mean', 'chroma_stft_var', 'chroma_cens_mean', 'chroma_cens_var', 'chroma_cqt_mean',
                     'chroma_cqt_var', 'melspectrogram_mean', 'melspectrogram_var', 'mfcc_mean', 'mfcc_var', 'rms_mean',
                     'rms_var', 'spec_bandwith_mean', 'spec_bandwith_var', 'spec_centroid_mean', 'spec_centroid_var',
                     'spec_contrast_mean', 'spec_contrast_var', 'spec_flatness_mean', 'spec_flatness_var', 'spec_rolloff_mean',
                     'spec_rolloff_var', 'tonnetz_mean', 'tonnetz_var', 'crossing_rate_mean', 'crossing_rate_var', 'labels'],
                      data=Features)

### Conversion of Dataframe into csv file ..

df.to_csv("Audio_Features_Extraction.csv")
print('.....%s seconds.....'% (time.time() - Start))   # 1 hr 43 mins