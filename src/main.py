#Store all audio files in dictionary where key: filename, value: label
#step1
import os
import wave
import math
import pandas as pd
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score, classification_report

raw_audio = dict()

directory = 'state_directory/hungry'
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        raw_audio[os.path.join(directory, filename)] = 'state_directory/hungry'
    else:
        continue

directory = 'state_directory/tired'
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        raw_audio[os.path.join(directory, filename)] = 'state_directory/tired'
    else:
        continue

directory = 'state_directory/discomfort'
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        raw_audio[os.path.join(directory, filename)] = 'state_directory/discomfort'
    else:
        continue

directory = 'state_directory/etc'
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        raw_audio[os.path.join(directory, filename)] = 'state_directory/etc'
    else:
        continue

print(raw_audio)

#step2

snippet_index = 1

def chop_and_store_snippets(filename, folder, label):

    global snippet_index
    # calculate audio second
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    print(filename)
    last_number_frames = 0

    # Create a folder for the snippets based on the label
    snippet_folder = "state_directory/" + label + "_snippet"
    if not os.path.exists(snippet_folder):
        os.makedirs(snippet_folder)

    # Slicing Audio file and storing in the corresponding folder
    for i in range(num_secs):
        shortfilename = filename.split("/")[1].split(".")[0]
        snippetfilename = os.path.join(snippet_folder, shortfilename + '_snippet' + str(snippet_index) + '.wav')
        snippet_index = snippet_index + 1
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        try:
            handle.setpos(handle.tell() - int(1 * frame_rate))
        except:
            print('error occur!')
            print(filename)

    handle.close()

# For each audio file, chop and store snippets in the corresponding folder with a label
for audio_file in raw_audio:
    label = raw_audio[audio_file].split("/")[-1]
    chop_and_store_snippets(audio_file, raw_audio[audio_file], label)

        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()

        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            #os.rename(snippetfilename, snippetfilename+".bak")
            print("This file is less than 1sec. Then we just pass")
            print(filename)
            print('------------------')
        snippet.close()

    #handle.close()

for audio_file in raw_audio:
    chop_and_store_snippets(audio_file, raw_audio[audio_file])

#step3

'''Chop and Transform each track'''


X = pd.DataFrame(columns = np.arange(10), dtype = 'float32').astype(np.float32)
j = 0
k = 0
tired_path = 'state_directory/tired/'
hungry_path = 'state_directory/hungry/'
discomfort_path = 'state_directory/discomfort/'
etc_path = 'state_directory/etc/'

for i, filename in enumerate(os.listdir(tired_path)):
    last_number_frames = -1
    if filename.endswith(".wav"):
        print(filename)
        audiofile, sr = librosa.load("state_directory/tired/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=9)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[9] = 'tired'
        X.loc[i] = x.loc[0]
        j = i


for i, filename in enumerate(os.listdir(hungry_path)):
    if filename.endswith(".wav"):
        print(filename)
        audiofile, sr = librosa.load("state_directory/hungry/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=9)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[9] = 'hungry'
        X.loc[i+j] = x.loc[0]
        k = i

for i, filename in enumerate(os.listdir(discomfort_path)):
    if filename.endswith(".wav"):
        print(filename)
        audiofile, sr = librosa.load("state_directory/discomfort/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=9)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[9] = 'discomfort'
        X.loc[i+j+k] = x.loc[0]
        l = i

for i, filename in enumerate(os.listdir(etc_path)):
    if filename.endswith(".wav"):
        print(filename)
        audiofile, sr = librosa.load("state_directory/etc/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=9)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[9] = 'etc'
        X.loc[i+j+k+l] = x.loc[0]

#Do something with missing values. you might want to do something more sophisticated with missing values later
X = X.fillna(0)

#step4



y = X[9]
del X[9]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    if classifier == LogisticRegression:
        kwargs['max_iter'] = 10000  # max_iter 값을 추가
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return (
        model.score(X_test, y_test),
        precision_score(y_test, y_predict, average='weighted', zero_division=0),
        recall_score(y_test, y_predict, average='weighted', zero_division=0)
    )

get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
print ("    Model, Accuracy, Precision, Recall")
print ("    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5))
print ("    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test))
print ("    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test))
print ("    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test))
#print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)


def pickle_model(model, modelname, path):
    with open(os.path.join(path, str(modelname) + '.pkl'), 'wb') as f:
        return pickle.dump(model, f)

path = 'state_directory/test'
model = RandomForestClassifier()
model.fit(X,y)
pickle_model(model, "myRandomForest", path)