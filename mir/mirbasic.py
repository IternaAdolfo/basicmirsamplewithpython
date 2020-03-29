import crepe
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
audio, sr = librosa.load('audio/loop.wav')

time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
