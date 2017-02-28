# Week 9 - Beyond Text

Intro stuff ...

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
import scipy #For frequency analysis
import scipy.fftpack
import nltk #the Natural Language Toolkit
import requests #For downloading our datasets
import numpy as np #for arrays
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn as sns #Makes the graphics look nicer

import IPython
import pydub #Requires ffmpeg to be installed
import speech_recognition
import soundfile

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning, it
%matplotlib inline

import os
import os.path
import csv
```

We will looking at a few different media for this week. First lets talk about one one that's older than text, spoken word. Audio files (as well as video) come in two major categories, lossy or lossless. Lossless files save all the information the microphone recorded, while lossy drop sections that humans will not notice. The recorded frequencies, for both types, are then compressed, which can introduce more loss still. To work with audio files we want a format that is not very compressed and preferably lossless. So `mp3` is not acceptable, instead we will work with `wav` files. Since many of you likely will not have `wav` files we can use python to convert to `wav`.

``` python
IPython.display.Audio('data/audio_samples/SBC060.mp3')
```

``` python
# We are using a different package to convert than the in the rest of the code
def convertToWAV(sourceFile, outputFile):
    if os.path.isfile(outputFile):
        print("{} exists already".format(outputFile))
        return
    #Naive format extraction
    sourceFormat = sourceFile.split('.')[-1]
    sound = pydub.AudioSegment.from_file(sourceFile, format=sourceFormat)
    sound.export(outputFile, format="wav")
    print("{} created".format(outputFile))
convertToWAV('data/audio_samples/SBC060.mp3', 'data/audio_samples/SBC060.wav')
```

Now that we have created our `wav`, notice that it is much large than the source `mp3`. We can load it with `soundfile` and work with like np array.

``` python
soundArr, soundSampleRate = soundfile.read('data/audio_samples/SBC060.wav')
soundArr.shape
```

This is the raw data in the file, in this file there are two channels but some files will have many more. The data is a series of numbers giving the location of the speaker membrane, with 0 being the resting location. By quickly and rhythmically changing the location a note can be achieved. The large the variation from the center the louder the sound and the faster the oscillations the higher the pitch. Note that the center of the oscillations does not have to be at 0.

``` python
soundSampleRate
```

The other piece of information we get is the sample rate, this tells us how many measurements there are per second. Which allows us to see how long the whole recording is:

``` python
numS = soundArr.shape[1] // soundSampleRate
print("The sample is {} seconds long".format(numS))
print("Or {:.2f} minutes".format(numS / 60))
```

Lets look at the first second of the recording

``` python
plt.plot(soundArr[:soundSampleRate])
```

We get 2 nearly flat lines, this means there is very little noise at this part of the recording. What variation there is, is due to compression or inference and is the slight hiss you sometimes hear in low quality recordings.

Lets expand our scope and look at the first 10 seconds

``` python
plt.plot(soundArr[:soundSampleRate * 10])
```

Now we can see definite spikes, each of these is a word or sound.

If we want to see what the different parts correspond to we will need a transcript. Since we got this file from the [Santa Barbara Corpus of Spoken American English
](http://www.linguistics.ucsb.edu/research/santa-barbara-corpus#Contents). We just need to load it.

``` python
def loadTranscript(targetFile):
    dfDict = {
        'time_start' : [],
        'time_end' : [],
        'speaker' : [],
        'text' : [],
    }
    with open(targetFile, encoding='latin-1') as f:
        reader = csv.reader(f, delimiter = '\t')
        for row in reader:
            dfDict['time_start'].append(float(row[0]))
            dfDict['time_end'].append(float(row[1]))
            dfDict['text'].append(row[3])
            if len(row[2]) > 0:
                dfDict['speaker'].append(row[2])
            else:
                dfDict['speaker'].append(dfDict['speaker'][-1])
    return pandas.DataFrame(dfDict)

transcriptDF = loadTranscript('data/audio_samples/SBC060.trn')
transcriptDF[:10]
```

Now let's look at a few sub sections, but first to make things easier we will convert the seconds markers to sample indices

``` python
#Need to be ints for indexing, luckily being off by a couple indices doesn't matter
transcriptDF['index_start'] = (transcriptDF['time_start'] * soundSampleRate).astype('int')
transcriptDF['index_end'] = (transcriptDF['time_end'] * soundSampleRate).astype('int')
```

Lets see what `'Rae and I and Sue and Buddy,'` looks like, its row 6 so:

``` python
subSample1 = soundArr[transcriptDF['index_start'][6]: transcriptDF['index_end'][6]]
plt.plot(subSample1)
```

And lets see what that sounds like:

``` python
soundfile.write('data/audio_samples/sample1.wav', subSample1, soundSampleRate)
IPython.display.Audio('data/audio_samples/sample1.wav')
```
and in frequency space

``` python
sample1FFT = scipy.fftpack.ifft(subSample1)
N = len(subSample1)
#We want the magnitude not the exact values, and the distribution is symmetric so only half
yf = abs(sample1FFT[:(N//2-1)])
k = np.linspace(0, N //2 - 1, N //2 - 1)
T = N / soundSampleRate
plt.plot(k / T, yf)
plt.xlabel('Frequency ($Hz$)')
```

What does a sniff look like?

``` python
subSample2 = soundArr[transcriptDF['index_start'][9]: transcriptDF['index_end'][9]]
plt.plot(subSample2)
```

And lets see what that sounds like:

``` python
soundfile.write('data/audio_samples/sample2.wav', subSample2, soundSampleRate)
IPython.display.Audio('data/audio_samples/sample2.wav')
```
and in frequency space

``` python
sample2FFT = scipy.fftpack.ifft(subSample2)
N = len(subSample1)
#We want the magnitude not the exact values, and the distribution is symmetric so only half
yf = abs(sample2FFT[:(N//2-1)])
k = np.linspace(0, N //2 - 1, N //2 - 1)
T = N / soundSampleRate
plt.plot(k / T, yf)
plt.xlabel('Frequency ($Hz$)')
```

Notice how there isn't a dominant frequency for the sniff
