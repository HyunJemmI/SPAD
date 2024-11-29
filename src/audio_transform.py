import os
import time
import wave
import math
import librosa
import pandas as pd
import numpy as np
import pickle
import time
import sounddevice as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score



def record_audio(filename, duration):
    print("녹음 시작")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("녹음 완료")

    # Save the audio to a WAV file
    wavefile = wave.open('', 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(sample_rate)
    wavefile.writeframes(recording.tobytes())
    wavefile.close()

# 마이크 샘플링 속도 설정
sample_rate = 44100

# 녹음 시간과 저장 파일 이름 설정
duration = 5  # 녹음 시간 (초)
output_filename = "1.wav"  # 저장 파일 이름

# 녹음 시작
record_audio(output_filename, duration)

def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

model = getModel("myRandomForest.pkl")

def chop_new_audio(filename, folder):
    print(folder)
    print(filename)
    handle = wave.open(folder+'/'+filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 1 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    #print filename
    last_number_frames = 0
    #Slicing Audio file
    for i in range(num_secs):

        shortfilename = filename.split(".")[0]
        snippetfilename = folder + '/' + shortfilename + 'snippet' + str(i+1) + '.wav'
        #print snippetfilename
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        #snippet.setsampwidth(2)
        #snippet.setframerate(11025)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        try:
          handle.setpos(handle.tell() - 1 * frame_rate)
        except:
          print('1 Error')
          pass
        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()

        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            #os.rename(snippetfilename, snippetfilename+".bak")
            print('This file is no longer than 1sec')
        snippet.close()

    #handle.close()

chop_new_audio("1.wav", "filename")

X = pd.DataFrame(columns=np.arange(10), dtype='float32').astype(np.float32)
j = 0
k = 0
directory_path = "filename"

for i, filename in enumerate(os.listdir(directory_path)):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load(os.path.join(directory_path, filename))
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype='float32')
        x[9] = 'test'
        X.loc[i] = x.loc[0]
        j = i  # 여기서 변수 j에 값을 할당합니다.
        X.head()


predictions = []

directory_path = "filename"

for i, filename in enumerate(os.listdir(directory_path)):
    last_number_frames = -1
    if filename.endswith(".wav"):
        # print filename
        audiofile, sr = librosa.load("filename" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=9)
        x = pd.DataFrame(fingerprint, dtype='float32')
        x_reshaped = x.values[:, :9]
        x_reshaped = x_reshaped[:1, :9]
        print(x_reshaped.shape)
        print(x_reshaped[:1, :9])  # Print the first 5 rows and 5 columns
        # print(x.shape)

        #         x_trans = x.transpose()
        #         print(x_trans.shape)
        prediction = model.predict(x_reshaped)
        predictions.append(prediction[0])

from collections import Counter
data = Counter(predictions)
print(data.most_common())   # Returns all unique items and their counts
print(data.most_common(1))
time.sleep(0.5)