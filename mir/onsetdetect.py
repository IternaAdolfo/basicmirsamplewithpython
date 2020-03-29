import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
#%matplotlib inline
import mir_eval
from IPython.display import Audio, display
np.set_printoptions(precision=3, linewidth=52)
import pandas as pd
pd.options.display.precision = 3
import mir_eval.display

import mir_eval

# Load in an audio signal
y, sr = librosa.load('audio/loop.wav')

# Estimate the onset times
est_onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

# Load in the reference annotation
ref_onsets = np.loadtxt('piano.onsets')
print(ref_onsets[:5], '\n', est_onsets[:5])

plt.figure(figsize=(10, 3))
M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
librosa.display.specshow(M, x_axis='time', y_axis='mel')
mir_eval.display.events(ref_onsets, color='w', alpha=0.8, linewidth=3)
mir_eval.display.events(est_onsets, color='c', alpha=0.8, linewidth=3, linestyle='--');
Audio(data=y, rate=sr)

mir_eval.onset.evaluate(ref_onsets, est_onsets)

plt.show()

