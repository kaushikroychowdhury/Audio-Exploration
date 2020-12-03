# Audio-Exploration
I love music, so I tried to understand how audio works and what's the difference between sound and audio. So I started exploring Audio Signal Processing (ADC).
For introduction the the problem of Genre Classification, I implemented both "Traditional Machine learning Pipeline" and also by using "Deep Learning".

For traditional Machine learning pipeline , the major task was to Extract different time and frequency domain features from Audio signal. As the Data is unstructured because
audio is Analogue Signal, so we need to convert it into digital signal. Methods are like "Fourier Transform" and "Short Time Fourier Transform".

The dataset I used is here : http://marsyas.info/downloads/datasets.html

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.
