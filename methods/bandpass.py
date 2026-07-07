import numpy as np
from scipy import signal

def bandpass_conv(
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
    NL = int(np.ceil(4 * fs / transition[0]))
    NH = int(np.ceil(4 * fs / transition[1]))

    # odd lengths
    if NL % 2 == 0:
        NL += 1

    if NH % 2 == 0:
        NH += 1
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

    hlpf = 2 * fc * np.sinc(2 * fc * nH)
    hlpf *= winH
    hlpf /= np.sum(hlpf)

    # ----------------------------------------------------------
    # High-pass (spectral inversion)
    # ----------------------------------------------------------

    nL = np.arange(NL) - (NL - 1) / 2
    fc = low_cut / fs
    
    hhpf = 2 * fc * np.sinc(2 * fc * nL)
    hhpf *= winL
    hhpf /= np.sum(hhpf)

    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    # ----------------------------------------------------------
    # Band-pass kernel
    # ----------------------------------------------------------
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




def bandpass_conv_kernel(
        eegdata,
        kernel,
        kind='same'
    ):
    """
    Band-pass FIR filtering by convolution.

    Parameters
    ----------
    eegdata : dict
        Dictionary containing:
            eegdata['X']
            eegdata['sfreq']

    kernel : ndarray
        The filter kernel.

    kind : str
        'same', 'full', or 'valid'

    Returns
    -------
    filtered : ndarray
        Filtered EEG data with the same dimensions as input (if kind='same').
    """
    data = eegdata.copy()
    X = data['X']
    fs = data['sfreq']


    # -----------------------------------------
    # Design Kernel filter
    # -----------------------------------------
    h = kernel

    # -----------------------------------------
    # Allocate output
    # -----------------------------------------
    filtered = np.zeros_like(X)
    # -----------------------------------------
    # Apply convolution
    # -----------------------------------------
    if X.ndim == 4:

        n_trials, n_samples, n_channels, _ = X.shape

        for trial in range(n_trials):
            for sample in range(n_samples):
                for ch in range(n_channels):

                    filtered[trial, sample, ch] = signal.convolve(
                        X[trial, sample, ch],
                        h,
                        mode=kind
                    )

    elif X.ndim == 3:

        n_trials, n_channels, _ = X.shape

        for trial in range(n_trials):
            for ch in range(n_channels):

                filtered[trial, ch] = signal.convolve(
                    X[trial, ch],
                    h,
                    mode=kind
                )
    else:
        raise ValueError("Input data must be a 3D or 4D array.")
    data['X'] = filtered 
    return data