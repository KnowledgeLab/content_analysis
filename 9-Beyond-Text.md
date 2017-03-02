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
import seaborn #Makes the graphics look nicer


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
import re


from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import numpy as np
import PIL
import PIL.Image

from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from skimage.util.colormap import viridis
import numpy as np
import PIL
import PIL.Image
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

```

We will looking at a few different media for this week. First lets talk about one one that's older than text, spoken word. Audio files (as well as video) come in two major categories, lossy or lossless. Lossless files save all the information the microphone recorded, while lossy drop sections that humans will not notice. The recorded frequencies, for both types, are then compressed, which can introduce more loss still. To work with audio files we want a format that is not very compressed and preferably lossless. So `mp3` is not acceptable, instead we will work with `wav` files. Since many of you likely will not have `wav` files we can use python to convert to `wav`.

``` python
samplePath = 'data/audio_samples/SBC014.mp3'
transcriptPath = 'data/audio_samples/SBC014.trn'
IPython.display.Audio(samplePath)
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
convertToWAV(samplePath, 'data/audio_samples/workingSample.wav')
```

Now that we have created our `wav`, notice that it is much large than the source `mp3`. We can load it with `soundfile` and work with like np array.

``` python
soundArr, soundSampleRate = soundfile.read('data/audio_samples/workingSample.wav')
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
    #Regex because the transcripts aren't consistent enough to use csv
    regex = re.compile(r"(\d+\.\d\d)\s(\d+\.\d\d)\s(.+:)?\s+(.*)")
    dfDict = {
        'time_start' : [],
        'time_end' : [],
        'speaker' : [],
        'text' : [],
    }
    with open(targetFile, encoding='latin-1') as f:
        for line in f:
            r = re.match(regex, line)
            dfDict['time_start'].append(float(r.group(1)))
            dfDict['time_end'].append(float(r.group(2)))
            if r.group(3) is None:
                dfDict['speaker'].append(dfDict['speaker'][-1])
            else:
                dfDict['speaker'].append(r.group(3))
            dfDict['text'].append(r.group(4))
    return pandas.DataFrame(dfDict)

transcriptDF = loadTranscript(transcriptPath)
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
N = len(sample1FFT)
freq = scipy.fftpack.fftfreq(N, d = 1 / soundSampleRate)
plt.plot(freq[:N//2], abs(sample1FFT)[:N//2]) #Only want positive frequencies
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Intensity')
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
N = len(sample2FFT)
freq = scipy.fftpack.fftfreq(N, d = 1 / soundSampleRate)
plt.plot(freq[:N//2], abs(sample2FFT)[:N//2]) #Only want positive frequencies
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Intensity')
```

Notice how there isn't a dominant frequency for the sniff

What are the dominant frequencies for the entire record?

``` python
#This takes a while
fullFFT = scipy.fftpack.ifft(soundArr)
N = len(fullFFT)
freq = scipy.fftpack.fftfreq(N, d = 1 / soundSampleRate)
plt.plot(freq[:N//2], abs(fullFFT)[:N//2]) #Only want positive frequencies
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Intensity')
```

We can use these to compare speakers. Lets look at the most common frequencies for each segment:

``` python
def maxfreq(sample, topN = 100):
    sampleFFT = scipy.fftpack.ifft(sample)
    N = len(sample)
    freqs = scipy.fftpack.fftfreq(N, d = 1 / soundSampleRate)
    tops =  np.argpartition(abs(sampleFFT[:N//2, 0]), -topN)[-topN:]
    np.mean(freqs[tops])

    return np.mean(freqs[tops])

freqs = []
for i, row in transcriptDF.iterrows():
    freqs.append(maxfreq(soundArr[row['index_start']: row['index_end']]))

transcriptDF['frequency'] = freqs
```

``` python
fg = seaborn.FacetGrid(data=transcriptDF, hue='speaker', aspect = 3)
fg.map(plt.scatter, 'time_start', 'frequency', linewidth = 1.5).add_legend()
```

We can do speech recognition on audio, to do this requires a complex machine learning system, luckily there are many online services to do this. We have a function that uses Google's API. There are two API's one is free but limited the other is not you can provided the function `speechRec` with a file containing the API keys, using `jsonFile=` if you wish. For more about this look [here](https://stackoverflow.com/questions/38703853/how-to-use-google-speech-recognition-api-in-python) or the `speech_recognition` [docs](https://github.com/Uberi/speech_recognition),

``` python
#Using another library so we need to use files again
def speechRec(targetFile, language = "en-US", raw = False, jsonFile = 'data/googleAPIKeys.json'):
    r = speech_recognition.Recognizer()
    if not os.path.isfile(jsonFile):
        jsonString = None
    else:
        with open(jsonFile) as f:
            jsonString = f.read()
    with speech_recognition.AudioFile(targetFile) as source:
        audio = r.record(source)
    try:
        if jsonString is None:
            print("Sending data to Google Speech Recognition")
            dat =  r.recognize_google(audio)
        else:
            print("Sending data to Google Cloud Speech")
            dat =  r.recognize_google_cloud(audio, credentials_json=jsonString)
    except speech_recognition.UnknownValueError:
        print("Google could not understand audio")
    except speech_recognition.RequestError as e:
        print("Could not request results from Google service; {0}".format(e))
    else:
        print("Success")
        return dat
```

The example is to low quality so we will be using another file `data/audio_samples/english.wav`

``` python
speechRec('data/audio_samples/english.wav')
```

In addition to audio we can also analysis images. This requires major computing resources so we will limit our analysis to basic features.

First we need to load an image, lets grab `data/image_samples/Citrus_fruits.jpg`:

``` python
IPython.display.Image('data/image_samples/Citrus_fruits.jpg')
```

We load the image with the Python Image Library `PIL`:

``` python
image = PIL.Image.open('data/image_samples/Citrus_fruits.jpg')
imageArr = np.asarray(image)
imageArr.shape
```

The image we have loaded is raster image, meaning it is a grid of pixels, each pixel contains 1-4 numbers giving the amounts of color in it. In this case we can see it has 3 values per pixel, these are RGB or Red, Green and Blue. If we want to see just the green we can look at just that array:

``` python
plt.imshow(imageArr[:,:,2], cmap='Greens') #The order is R G B, so 2 is the Green
```

White is caused by all the values of pixel being maximal so while the white patches look dark here, when added together they are white. We only really need the image in black and white so lets convert it to that:

``` python
image_gray = image.convert('L')
image_grayArr = np.asarray(image_gray)
```

We can find blobs in cople of ways, here we will use the [laplacian of Gaussian](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html):

```
image_gray = image.convert('L')
image_grayArr = np.asarray(image_gray)
blobs_log = blob_log(image_grayArr, max_sigma=200, num_sigma=10, threshold=.1)
fig, ax = plt.subplots()

plt.imshow(image_gray, interpolation='nearest')
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, linewidth=2, fill=False)
    ax.add_patch(c)
```
It wasn't very good at finding the blobs, was it. We can try to find them another way, we can look for edge:


``` python
labels = segmentation.slic(image_gray, compactness=30, n_segments=100)
edges = filters.sobel(image_grayArr)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)

out = graph.draw_rag(labels, g, edges_rgb, node_color="#999999", colormap=viridis)

io.imshow(out)
io.show()
```

The the edges are marked in white

``` python
thresh = threshold_otsu(image_grayArr)
bw = closing(image_grayArr > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image_grayArr)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)
for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
```
