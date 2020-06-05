# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:46:17 2020

@author: PURUSHOTTAM
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft

%matplotlib inline

audio_path = 'C:/Users/PURUSHOTTAM/Desktop/Barks/WavFiles/Train/'
pict_Path = 'C:/Users/PURUSHOTTAM/Desktop/Barks/PicsGenerated/Train/'
test_pict_Path = 'C:/Users/PURUSHOTTAM/Desktop/Barks/PicsGenerated/Test/'
test_audio_path = 'C:/Users/PURUSHOTTAM/Desktop/Barks/WavFiles/Test/'
samples = []

#-------------------------------------------------------------------------------------
#CODE TO MAKE DIRECTORES AND FORM WAVE FORM IMAGES FROM THE AUDIO FILES GATHERED (WAV FORMAT)

subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + x):
        subFolderList.append(x)
        if not os.path.exists(pict_Path  + x):
            os.makedirs(pict_Path + x)
        if not os.path.exists(test_pict_Path  + x):
            os.makedirs(test_pict_Path + x)    
            
            
sample_audio = []
total = 0

for x in subFolderList:    
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path +x) if '.wav' in y]
    total += len(all_files)
    # collect the first file from each dir
    sample_audio.append(audio_path  + x +'/'+ all_files[0])
    
    # show file counts
    print('count: %d : %s' % (len(all_files), x ))
print(total)            
           
# Make the Waveforms
def wav2img_waveform(wav_path, targetdir='', figsize=(4,4)):
    samplerate,test_sound  = wavfile.read(wav_path)
    fig = plt.figure(figsize=figsize)
    plt.plot(test_sound)
    plt.axis('off')
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    plt.savefig('%s.png' % output_file)
    plt.close() 
    
#Build the training pics    
for i, x in enumerate(subFolderList[:3]):
    print(i, ':', x)
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
    for file in all_files[:]:
        wav2img_waveform(audio_path + x + '/' + file, pict_Path+ x)
        
#Build the testing pics        
for i, x in enumerate(subFolderList[:3]):
    print(i, ':', x)
    # get all the wave files
    all_files = [y for y in os.listdir(test_audio_path + x) if '.wav' in y]
    for file in all_files:
        wav2img_waveform(test_audio_path + x + '/' + file, test_pict_Path + x)
         
        
#-------------------------------------------------------------------------------------

#BEGINNING THE TRAINING
        
import tensorflow as tf
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 5} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#CNN Building
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json        
        
        
#WE WILL ADD ANOTHER CONVOLUTION/POOLING LAYER AS FOLLOWS
classifier1 = Sequential()

#Convolution1
classifier1.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))   # number of feature detectors,number of rows, number of columns, input shape{for theano backend=>(3,64,64)}

#Pooling1
classifier1.add(MaxPooling2D(pool_size = (2,2)))  #The size of filter is 2X2

#Convolution2
classifier1.add(Convolution2D(32,3,3, activation = 'relu'))   # number of feature detectors,number of rows, number of columns, {NO NEED HERE=>}input shape{for theano backend=>(3,64,64)}

#Pooling2
classifier1.add(MaxPooling2D(pool_size = (2,2)))  #The size of filter is 2X2

#Flattening
classifier1.add(Flatten())  

#classifier1.add(Dense(output_dim=5))

#Full Connection....Starting with ANN
classifier1.add(Dense(output_dim = 128, activation = 'relu'))
classifier1.add(Dense(output_dim = 5, activation = 'softmax'))  #Final Output        

#Compiling the Model
classifier1.compile(optimizer = 'adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])

#Fitting the image

from keras.preprocessing.image import ImageDataGenerator
 
#Code taken from Keras Documentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/PURUSHOTTAM/Desktop/Barks/PicsGenerated/Train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'C:/Users/PURUSHOTTAM/Desktop/Barks/PicsGenerated/Test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier1.fit_generator(
        training_set,
        steps_per_epoch=4000,    #Total images we have
        epochs=10,
        validation_data=test_set,
        validation_steps=1000
        )   #Total images we have





#Saving the model
# serialize model to JSON
model_json = classifier1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier1.save_weights("model.h5")
print("Saved model to disk")


#Loading the Model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier1 = model_from_json(loaded_model_json)
# load weights into new model
classifier1.load_weights("model.h5")
print("Loaded model from disk")
 




#---------------------------------------------------------------------------------
# Let's try with the spectogram representation

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('C:/Users/PURUSHOTTAM/Desktop/Barks/WavFiles/Test/Angry/28_Sd_A_Angry-converted.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#--------------------------------------------------------------------------------------
import librosa
sample_rate, samples = wavfile.read('C:/Users/PURUSHOTTAM/Desktop/Barks/WavFiles/Test/Angry/28_Sd_A_Angry-converted.wav')
S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)


