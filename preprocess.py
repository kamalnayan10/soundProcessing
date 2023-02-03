import librosa , librosa.display
import matplotlib.pyplot as plt
import numpy as np

FILE = "blues.00000.wav"

# WAVEFORM

signal , sr =  librosa.load(FILE , sr = 22050)

# librosa.display.waveshow(signal , sr = sr)

# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# Perofrming a FFT(Fast Fourier Transform)

fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(left_frequency,left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# Creating STFT(Short Time Fourier Transform)
n_fft = 2048
hop_len = 512
stft = librosa.core.stft(signal , n_fft = n_fft , hop_length = hop_len)

spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram , sr = sr , hop_length=hop_len)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

# Calculating MFCC(Mel-Frequency Cepstral Coefficients)

MFCCs = librosa.feature.mfcc(signal ,n_fft = n_fft , hop_length = hop_len , n_mfcc = 13)
librosa.display.specshow(MFCCs , sr = sr , hop_length=hop_len)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()