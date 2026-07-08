import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from bciflow.modules.tf.bandpass import bandpass_conv, chebyshevII
from scipy import stats
from scipy import signal
def bandpass_conv_corrigido(
    eegdata,
    low_cut=4,
    high_cut=40,
    transition=None,
    window_type='hamming',
    kind='same'
):
    """
    Band-pass FIR filter by convolution.

    Parameters
    ----------
    eegdata : dict
        Dictionary containing:
            eegdata['X'] : EEG data (..., time)
            eegdata['sfreq'] : sampling frequency (Hz)

    low_cut : float
        Lower cutoff frequency (Hz)

    high_cut : float
        Upper cutoff frequency (Hz)

    transition : float or list
        Transition bandwidth (Hz). If None, uses half of passband width.

    window_type : {'hamming', 'blackman'}

    kind : {'same', 'full', 'valid'}

    Returns
    -------
    output : dict
        Copy of eegdata with filtered signals.
    """
    print("Starting bandpass convolution corrigido...")

    X = eegdata['X'].copy()
    fs = eegdata['sfreq']

    original_shape = X.shape
    X = X.reshape((-1, original_shape[-1]))

    # ----------------------------------------------------------
    # Transition bandwidth
    # ----------------------------------------------------------

    if transition is None:
        transition = (high_cut - low_cut) / 2

    if np.isscalar(transition):
        transition = [transition, transition]

    # ----------------------------------------------------------
    # Filter lengths
    # ----------------------------------------------------------
    print(f"Transition: {transition}")
    NL = int(np.ceil(4 * fs / transition[0]))
    NH = int(np.ceil(4 * fs / transition[1]))

    # odd lengths
    if NL % 2 == 0:
        NL += 1

    if NH % 2 == 0:
        NH += 1
    print(f"NL: {NL}, NH: {NH}")
    # ----------------------------------------------------------
    # Windows
    # ----------------------------------------------------------

    if window_type.lower() == 'hamming':
        winL = np.hamming(NL)
        winH = np.hamming(NH)

    elif window_type.lower() == 'blackman':
        winL = np.blackman(NL)
        winH = np.blackman(NH)

    else:
        raise ValueError("window_type must be 'hamming' or 'blackman'")

    # ----------------------------------------------------------
    # Low-pass
    # ----------------------------------------------------------

    nH = np.arange(NH) - (NH - 1) / 2
    fc = high_cut / fs

    hlpf = np.sinc(2 * fc * nH)
    hlpf *= winH
    hlpf /= np.sum(hlpf)

    # ----------------------------------------------------------
    # High-pass (spectral inversion)
    # ----------------------------------------------------------

    nL = np.arange(NL) - (NL - 1) / 2
    fc = low_cut / fs
    
    hhpf = np.sinc(2 * fc * nL)
    hhpf *= winL
    hhpf /= np.sum(hhpf)

    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    # ----------------------------------------------------------
    # Band-pass kernel
    # ----------------------------------------------------------
    print(f"hlpf length: {len(hlpf)}, hhpf length: {len(hhpf)}")
    kernel = np.convolve(hlpf, hhpf)

    # ----------------------------------------------------------
    # Filtering
    # ----------------------------------------------------------

    filtered = np.empty_like(X)

    for i in range(X.shape[0]):
        filtered[i] = np.convolve(
            X[i],
            kernel,
            mode=kind
        )

    filtered = filtered.reshape(original_shape)

    output = eegdata.copy()
    output['X'] = filtered

    return output
# Frequência de amostragem
fs = 250

# Tempo (10 segundos)
t = np.arange(0, 10, 1/fs)

# Sinal composto
x = (
    1.0*np.sin(2*np.pi*2*t) +    # 2 Hz
    1.0*np.sin(2*np.pi*10*t) +   # 10 Hz
    1.0*np.sin(2*np.pi*20*t) +   # 20 Hz
    1.0*np.sin(2*np.pi*60*t)     # 60 Hz
)

# Formato esperado pela função
eegdata = {
    "X": x.reshape(1, 1, -1),
    "sfreq": fs
}
eegdata_origin = eegdata.copy()
eegdata_corrigido = eegdata.copy()
print("começando a filtragem")
filtered = bandpass_conv(
    eegdata,
    low_cut=50,
    high_cut=70,
    transition=2
)
print("começando a filtragem")
filtered_corrigido = bandpass_conv_corrigido(
     eegdata_corrigido,
    low_cut=50,
    high_cut=70,
    transition=2
)

y = filtered["X"][0,0]

plt.figure(figsize=(12,4))

plt.plot(t, x, label="Original")
plt.plot(t, y, label="Filtrado")
plt.plot(t, filtered_corrigido["X"][0,0], label="Filtrado Corrigido")

plt.xlim(0,2)

plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.show()


freq = np.fft.rfftfreq(len(x), d=1/fs)

Xf = np.abs(np.fft.rfft(x))
Yf = np.abs(np.fft.rfft(y))
corrigido_f = np.abs(np.fft.rfft(filtered_corrigido["X"][0,0]))

plt.figure(figsize=(12,5))

plt.plot(freq, Xf, label="Original")
plt.plot(freq, Yf, label="Filtrado")
plt.plot(freq, corrigido_f, label="Filtrado Corrigido")

plt.xlim(0,80)

plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")

plt.grid(True)
plt.legend()

plt.show()

f, t, Sxx1 = signal.spectrogram(eegdata_origin["X"][0,0], fs=fs, nperseg=int(fs/4))

f, t, Sxx2 = signal.spectrogram(y, fs=fs, nperseg=int(fs/4))

f, t, Sxx3 = signal.spectrogram(filtered_corrigido["X"][0,0], fs=fs, nperseg=int(fs/4))
#plot spectrogram of the first trial
# multiple plots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].pcolormesh(t, f, Sxx1, vmin = np.percentile(Sxx1, 5), vmax = np.percentile(Sxx1, 95))
axs[0].set_ylabel('Frequency [Hz]')
axs[0].set_xlabel('Time [sec]')
axs[0].set_title('Spectrogram of trial 1 subject 1 ')
axs[1].pcolormesh(t, f, Sxx2, vmin = np.percentile(Sxx2, 5), vmax = np.percentile(Sxx2, 95))
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')
axs[1].set_title('Spectrogram of trial 1 subject 1 (Bandpass Filtered convolution)')
axs[2].pcolormesh(t, f, Sxx3, vmin = np.percentile(Sxx3, 5), vmax = np.percentile(Sxx3, 95))
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_xlabel('Time [sec]')
axs[2].set_title('Spectrogram of trial 1 subject 1 (Bandpass Filtered convolution corrigido)')

plt.tight_layout()
plt.show()