# pip install tftb pyemd scikit-learn soundfile scipy numpy matplotlib librosa streamlit PyWavelets
# streamlit run advanced_audio_analysis.py

from tftb.processing import WignerVilleDistribution
from pyemd import EMD
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display
from librosa.core import griffinlim
import scipy
from scipy.signal import hilbert
from scipy.stats import zscore
import soundfile as sf
import io
import streamlit as st

# Function to load audio file
def load_audio(file):
    y, sr = sf.read(io.BytesIO(file.read()))
    if len(y.shape) > 1:
        y = y[:, 0]
    return y, sr

# Function to perform DWT
def perform_dwt(y, wavelet, level):
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
    return coeffs

# Function to reconstruct signal from DWT coefficients
def reconstruct_signal(coeffs, selected_wavelet):
    y_reconstructed = pywt.waverec(coeffs, wavelet=selected_wavelet)
    return y_reconstructed

# Function to perform STFT
def perform_stft(y, sr, n_fft, hop_length):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return D

# Function to perform CWT
def perform_cwt(y, scales, wavelet):
    coefficients, frequencies = pywt.cwt(y, scales, wavelet)
    return coefficients, frequencies

# Function to perform EMD
def perform_emd(y):
    emd = EMD()
    imfs = emd.emd(y)
    return imfs

# Function to create multivariate signal using delayed versions
def create_multivariate_signal(y, delays):
    n_samples = len(y)
    data = np.zeros((n_samples - delays, delays + 1))
    for i in range(delays + 1):
        data[:, i] = y[i:n_samples - delays + i]
    return data

# Function to perform ICA on delayed signal
def perform_ica_delayed(y, n_components, delays):
    X = create_multivariate_signal(y, delays)
    ica = FastICA(n_components=n_components, random_state=0)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    return S_, A_

# Function to perform ICA on spectrogram
def perform_ica_spectrogram(y, sr, n_components, n_fft, hop_length):
    # Compute the magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S = S.T  # Transpose to have shape (n_frames, n_frequencies)
    ica = FastICA(n_components=n_components, random_state=0)
    S_ = ica.fit_transform(S)
    return S_, S, ica

# Function to perform NMF
def perform_nmf(y, sr, n_components, n_fft, hop_length):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(S)
    H = model.components_
    return W, H

# Function to perform HHT
def perform_hilbert_huang(y, sr):
    emd = EMD()
    imfs = emd.emd(y)
    hilbert_spectra = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * sr
        hilbert_spectra.append((amplitude_envelope, instantaneous_frequency))
    return hilbert_spectra

def perform_bispectrum(y):
    # Placeholder for bispectrum computation
    # Requires custom implementation or specialized library
    pass

def perform_cyclostationary(y, sr):
    # Placeholder for cyclostationary analysis
    # May use third-party libraries or custom code
    pass


def perform_wvd(y):
    wvd = WignerVilleDistribution(y)
    tfr, t, f = wvd.run()
    return tfr, t, f

# Function to perform Modulation Spectrum Analysis
def perform_modulation_spectrum(y, sr):
    # Compute the spectrogram
    S = np.abs(librosa.stft(y))
    # Compute modulation spectrum
    mod_spec = np.abs(np.fft.fft(S, axis=1))
    return mod_spec

# Function to perform Cross-Correlation Analysis
def perform_cross_correlation(y1, y2):
    correlation = np.correlate(y1, y2, mode='full')
    lag = np.arange(-len(y1) + 1, len(y2))
    return correlation, lag

# Function to perform Sparse Representation
def perform_sparse_representation(y, n_nonzero_coefs):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    y = y.reshape(-1, 1)
    omp.fit(y, y)
    coef = omp.coef_
    return coef

# Function to perform machine learning techinques?
def train_ml_model(features, labels):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model



# Function to plot waveform
def plot_waveform(y, sr, title='Waveform'):
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot spectrogram
def plot_spectrogram(y, sr, title='Spectrogram'):
    fig, ax = plt.subplots(figsize=(12, 4))
    S = np.abs(librosa.stft(y))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis='time', y_axis='log', ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot DWT coefficients
def plot_dwt_coeffs(coeffs, sr, title='DWT Coefficients'):
    fig, axs = plt.subplots(len(coeffs), 1, figsize=(12, 2 * len(coeffs)))
    for i, coeff in enumerate(coeffs):
        axs[i].plot(coeff)
        axs[i].set_title(f'Level {i} Coefficients')
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot STFT magnitude
def plot_stft(D, sr, hop_length, title='STFT Magnitude'):
    fig, ax = plt.subplots(figsize=(12, 4))
    S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        S_dB, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='log', ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    st.pyplot(fig)

# Function to plot CWT scalogram
def plot_cwt(coefficients, scales, y, sr, wavelet, title='CWT Scalogram'):
    # Compute the magnitude of the complex coefficients
    magnitude = np.abs(coefficients)

    # Compute the time vector
    t = np.linspace(0, len(y) / sr, num=magnitude.shape[1])

    # Compute corresponding frequencies
    frequencies = pywt.scale2frequency(wavelet, scales) * sr

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the scalogram
    ax.imshow(
        magnitude, extent=[t.min(), t.max(), frequencies.min(), frequencies.max()],
        cmap='viridis', aspect='auto',
        vmax=magnitude.max(), vmin=magnitude.min(),
        origin='lower'
    )
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    st.pyplot(fig)

# Function to plot EMD IMFs
def plot_imfs(imfs, sr, title='Intrinsic Mode Functions'):
    num_imfs = imfs.shape[0]
    fig, axs = plt.subplots(num_imfs, 1, figsize=(12, 2 * num_imfs))
    for i in range(num_imfs):
        axs[i].plot(imfs[i])
        axs[i].set_title(f'IMF {i + 1}')
    plt.tight_layout()
    st.pyplot(fig)

# Function to extract statistical features
def extract_statistical_features(y):
    features = {}
    features['Mean'] = np.mean(y)
    features['Variance'] = np.var(y)
    features['Skewness'] = scipy.stats.skew(y)
    features['Kurtosis'] = scipy.stats.kurtosis(y)
    features['Entropy'] = scipy.stats.entropy(np.abs(y))
    return features

# Function to extract speaker features
def extract_features(y, sr):
    # Ensure y is a 1D array
    if y.ndim > 1:
        y = y.flatten()
    # Trim silence from the beginning and end
    y, _ = librosa.effects.trim(y)
    # Normalize the audio
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    
    # Spectral Features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Combine features into a single vector
    features = np.concatenate((
        mfccs_mean,
        mfccs_std,
        [pitch_mean, pitch_std, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
    ))
    return features

def build_speaker_model(features_array, n_components=16):
    n_samples = features_array.shape[0]
    n_components = min(n_components, n_samples)
    if n_components < 1:
        n_components = 1
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=3)
    gmm.fit(features_array)
    return gmm

# Compute SNR
def compute_snr(original, watermarked):
    noise = watermarked - original
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Main Streamlit app
def main():
    st.title('Advanced Audio Analysis Tool')

    # Sidebar for file upload and method selection
    st.sidebar.header('Upload Audio File')
    uploaded_file = st.sidebar.file_uploader(
        'Choose an audio file', type=['wav', 'mp3', 'flac']
    )

    if uploaded_file is not None:
        # Load the audio file
        y, sr = load_audio(uploaded_file)
        duration = len(y) / sr

        # Audio playback
        st.subheader('Original Audio')
        st.audio(uploaded_file, format='audio/wav')

        # Plot original waveform
        plot_waveform(y, sr, title='Original Audio Waveform')

        # Select analysis method
        st.sidebar.header('Analysis Method')
        method = st.sidebar.selectbox(
            'Choose a method',
            ('DWT', 'DWT+', 'STFT', 'CWT', 'emd', 'Statistical Features', 'ICA', 'WPD', 'NMF', 'Adaptive Filtering', 'Modulation Spectrum Analysis', 'Speaker Characteristic Analysis', 'Watermark Analysis')
        )

        if method == 'DWT':
            # DWT Parameters
            wavelet_family = st.sidebar.selectbox(
                'Select Wavelet Family', pywt.families()
            )
            wavelet_list = pywt.wavelist(wavelet_family)
            selected_wavelet = st.sidebar.selectbox(
                'Select Wavelet', wavelet_list
            )
            max_level = pywt.dwt_max_level(
                len(y), pywt.Wavelet(selected_wavelet).dec_len
            )
            level = st.sidebar.slider(
                'Select Decomposition Level', min_value=1,
                max_value=min(10, max_level), value=5
            )

            # Perform DWT
            coeffs = perform_dwt(y, selected_wavelet, level)

            # Plot DWT coefficients
            st.subheader('DWT Coefficients')
            st.write(f'Using wavelet: {selected_wavelet}, Level: {level}')
            plot_dwt_coeffs(coeffs, sr)

        elif method == 'DWT+':
            st.sidebar.header('DWT Parameters')
            # DWT Parameters
            wavelet = st.sidebar.selectbox('Select Wavelet', pywt.wavelist(kind='discrete'))
            max_level = pywt.dwt_max_level(len(y), pywt.Wavelet(wavelet).dec_len)
            level = st.sidebar.slider('Decomposition Level', min_value=1, max_value=max_level, value=3)

            # Perform DWT
            coeffs = pywt.wavedec(y, wavelet, level=level)

            # Initialize a list to keep track of selected components
            selected_components = []

            # Plot DWT Components (Waveforms and Spectrograms)
            st.subheader('DWT Components')
            num_components = len(coeffs)
            component_labels = []

            for i in range(num_components):
                if i == 0:
                    label = f'Approximation Coefficients (Level {level})'
                else:
                    label = f'Detail Coefficients (Level {level - i + 1})'
                component_labels.append(label)
                st.write(f'**{label}**')

                # Create a checkbox for selecting the component
                is_selected = st.checkbox(f'Select {label}', key=f'select_{i}')
                if is_selected:
                    selected_components.append(i)

                # Get the component data
                component_data = coeffs[i]

                # Plot waveform
                fig_waveform, ax_waveform = plt.subplots(figsize=(12, 2))
                ax_waveform.plot(component_data)
                ax_waveform.set_title(f'Waveform of {label}')
                st.pyplot(fig_waveform)

                # Compute and plot spectrogram
                n_fft = 1024  # Adjust as needed
                hop_length = n_fft // 4
                D = librosa.stft(component_data, n_fft=n_fft, hop_length=hop_length)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                fig_spectrogram, ax_spectrogram = plt.subplots(figsize=(12, 4))
                img = librosa.display.specshow(
                    S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax_spectrogram
                )
                fig_spectrogram.colorbar(img, ax=ax_spectrogram, format="%+2.0f dB")
                ax_spectrogram.set_title(f'Spectrogram of {label}')
                st.pyplot(fig_spectrogram)

                # Normalize the component audio
                component_audio_norm = component_data / (np.max(np.abs(component_data)) or 1)

                # Provide play and download buttons
                audio_buffer = io.BytesIO()
                sf.write(audio_buffer, component_audio_norm, sr, subtype='PCM_16', format='WAV')
                audio_buffer.seek(0)

                col1, col2 = st.columns(2)
                with col1:
                    st.audio(audio_buffer, format='audio/wav')
                with col2:
                    st.download_button(
                        label=f'Download {label} as WAV',
                        data=audio_buffer,
                        file_name=f'{label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("_coefficients", "")}.wav',
                        mime='audio/wav'
                    )

            # Buttons for Reconstruction and Subtraction
            st.subheader('Reconstruction and Subtraction')

            # Check if any components are selected
            if selected_components:
                col_reconstruct, col_subtract = st.columns(2)
                with col_reconstruct:
                    if st.button('Reconstruct Audio from Selected Components'):
                        # Create a list of coefficients for reconstruction
                        coeffs_selected = [None] * num_components
                        for idx in selected_components:
                            coeffs_selected[idx] = coeffs[idx]
                        # Reconstruct the signal using inverse DWT
                        reconstructed_audio = pywt.waverec(coeffs_selected, wavelet)
                        # Ensure the reconstructed audio has the same length as the original
                        reconstructed_audio = reconstructed_audio[:len(y)]
                        # Normalize and export
                        reconstructed_audio /= np.max(np.abs(reconstructed_audio)) or 1
                        reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)
                        audio_buffer_reconstructed = io.BytesIO()
                        sf.write(audio_buffer_reconstructed, reconstructed_audio_int16, sr, subtype='PCM_16', format='WAV')
                        audio_buffer_reconstructed.seek(0)
                        st.write('Playing Reconstructed Audio (Using Selected Components)')
                        st.audio(audio_buffer_reconstructed, format='audio/wav')
                        st.download_button(
                            label='Download Reconstructed Audio',
                            data=audio_buffer_reconstructed,
                            file_name='reconstructed_audio.wav',
                            mime='audio/wav'
                        )

                with col_subtract:
                    if st.button('Subtract Selected Components from Original Audio'):
                        # Create a list of coefficients for selected components
                        coeffs_selected = [None] * num_components
                        for idx in selected_components:
                            coeffs_selected[idx] = coeffs[idx]
                        # Reconstruct the selected components
                        components_audio = pywt.waverec(coeffs_selected, wavelet)
                        # Ensure the reconstructed components have the same length as the original
                        components_audio = components_audio[:len(y)]
                        components_audio /= np.max(np.abs(components_audio)) or 1
                        # Subtract from original audio
                        reconstructed_audio = y - components_audio
                        reconstructed_audio /= np.max(np.abs(reconstructed_audio)) or 1
                        reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)
                        audio_buffer_subtract = io.BytesIO()
                        sf.write(audio_buffer_subtract, reconstructed_audio_int16, sr, subtype='PCM_16', format='WAV')
                        audio_buffer_subtract.seek(0)
                        st.write('Playing Audio (Original Minus Selected Components)')
                        st.audio(audio_buffer_subtract, format='audio/wav')
                        st.download_button(
                            label='Download Subtracted Audio',
                            data=audio_buffer_subtract,
                            file_name='subtracted_audio.wav',
                            mime='audio/wav'
                        )
            else:
                st.info('Please select at least one component to reconstruct or subtract.')

        elif method == 'STFT':
            # STFT Parameters
            n_fft = st.sidebar.slider(
                'Window Size (n_fft)', min_value=256, max_value=4096,
                value=1024, step=256
            )
            hop_length = st.sidebar.slider(
                'Hop Length', min_value=64, max_value=1024,
                value=512, step=64
            )

            # Perform STFT
            D = perform_stft(y, sr, n_fft, hop_length)

            # Plot STFT magnitude
            st.subheader('STFT Magnitude')
            plot_stft(D, sr, hop_length)

        elif method == 'CWT':
            # CWT Parameters
            wavelet_list = pywt.wavelist(kind='continuous')
            selected_wavelet = st.sidebar.selectbox(
                'Select Wavelet', wavelet_list
            )
            max_scale = st.sidebar.slider(
                'Select Maximum Scale', min_value=1,
                max_value=128, value=32
            )
            scales = np.arange(1, max_scale + 1)

            # Perform CWT
            coefficients, frequencies = perform_cwt(
                y, scales, selected_wavelet
            )

            # Plot CWT scalogram
            st.subheader('CWT Scalogram')
            plot_cwt(coefficients, scales, y, sr, selected_wavelet)

        elif method == 'emd':
            # Perform EMD
            imfs = perform_emd(y)

            # Plot IMFs
            st.subheader('Intrinsic Mode Functions (IMFs)')
            plot_imfs(imfs, sr)
        
        elif method == 'WPD':
            st.sidebar.header('WPD Parameters')
            # WPD Parameters
            wavelet = st.sidebar.selectbox('Select Wavelet', pywt.wavelist(kind='discrete'))
            max_level = pywt.dwt_max_level(len(y), pywt.Wavelet(wavelet).dec_len)
            level = st.sidebar.slider('Decomposition Level', min_value=1, max_value=max_level, value=3)

            # Perform WPD
            wp = pywt.WaveletPacket(data=y, wavelet=wavelet, mode='symmetric', maxlevel=level)

            # Collect all nodes at the final level
            nodes = wp.get_level(level, order='freq')
            labels = [node.path for node in nodes]

            # Extract coefficients
            coefficients = [node.data for node in nodes]

            # Plot Wavelet Packet Coefficients
            st.subheader('Wavelet Packet Coefficients at Level {}'.format(level))
            fig, axes = plt.subplots(len(coefficients), 1, figsize=(12, 2 * len(coefficients)))
            for i, coeff in enumerate(coefficients):
                axes[i].plot(coeff)
                axes[i].set_title('Node {}'.format(labels[i]))
            plt.tight_layout()
            st.pyplot(fig)
        
        elif method == 'NMF':
            st.sidebar.header('NMF Parameters')
            n_components = st.sidebar.slider(
                'Number of Components', min_value=1, max_value=10, value=2
            )
            n_fft = st.sidebar.slider(
                'Window Size (n_fft)', min_value=256, max_value=4096,
                value=1024, step=256
            )
            hop_length = st.sidebar.slider(
                'Hop Length', min_value=64, max_value=1024,
                value=512, step=64
            )

            # Compute the magnitude spectrogram
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            # Apply NMF
            nmf_model = NMF(n_components=n_components, init='random', random_state=0)
            W = nmf_model.fit_transform(S)  # Basis spectra
            H = nmf_model.components_       # Activations

            # Visualize the basis spectra
            st.subheader('NMF Basis Spectra (Components)')
            fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components))
            for i in range(n_components):
                axes[i].plot(W[:, i])
                axes[i].set_title('Component {}'.format(i + 1))
            plt.tight_layout()
            st.pyplot(fig)

        elif method == 'Adaptive Filtering':
            st.sidebar.header('Adaptive Filtering Parameters')
            noise_reduction_level = st.sidebar.slider(
                'Noise Reduction Level', min_value=0.0, max_value=1.0, value=0.5
            )

            # Simple noise reduction using spectral gating
            # Compute the short-time Fourier transform
            stft = librosa.stft(y)
            magnitude, phase = np.abs(stft), np.angle(stft)

            # Estimate the noise profile from the first second
            noise_frames = int(sr * 1 / (stft.shape[1] / len(y)))
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

            # Apply spectral gating
            magnitude_reduced = np.maximum(magnitude - noise_reduction_level * noise_profile, 0)

            # Reconstruct the signal
            stft_reduced = magnitude_reduced * np.exp(1j * phase)
            y_reconstructed = librosa.istft(stft_reduced)

            # Plot original and reconstructed waveforms
            st.subheader('Original vs. Noise-Reduced Signal')
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            ax[0].plot(y)
            ax[0].set_title('Original Signal')
            ax[1].plot(y_reconstructed)
            ax[1].set_title('Noise-Reduced Signal')
            plt.tight_layout()
            st.pyplot(fig)

            # Provide playback and download options
            st.write('Playing Noise-Reduced Audio')
            y_reconstructed /= np.max(np.abs(y_reconstructed)) or 1
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, y_reconstructed, sr, subtype='PCM_16', format='WAV')
            audio_buffer.seek(0)
            st.audio(audio_buffer, format='audio/wav')
            st.download_button(
                label='Download Noise-Reduced Audio',
                data=audio_buffer,
                file_name='noise_reduced_audio.wav',
                mime='audio/wav'
            )

        elif method == 'Modulation Spectrum Analysis':
            st.sidebar.header('Modulation Spectrum Parameters')
            n_fft = st.sidebar.slider(
                'Window Size for STFT (n_fft)', min_value=256, max_value=4096,
                value=1024, step=256
            )
            hop_length = st.sidebar.slider(
                'Hop Length for STFT', min_value=64, max_value=1024,
                value=512, step=64
            )
            modulation_hop_length = st.sidebar.slider(
                'Hop Length for Modulation Spectrum', min_value=1, max_value=256,
                value=32, step=1
            )

            # Compute the spectrogram
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

            # Compute the modulation spectrum for each frequency bin
            modulation_spectrogram = []
            for freq_bin in S:
                # Compute the temporal envelope of the frequency bin
                envelope = freq_bin
                # Compute the FFT of the envelope
                modulation_spectrum = np.abs(np.fft.fft(envelope))
                modulation_spectrogram.append(modulation_spectrum)

            modulation_spectrogram = np.array(modulation_spectrogram)

            # Plot the modulation spectrogram
            st.subheader('Modulation Spectrogram')
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(
                librosa.amplitude_to_db(modulation_spectrogram, ref=np.max),
                sr=sr,
                hop_length=modulation_hop_length,
                x_axis='linear',
                y_axis='log',
                ax=ax
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title('Modulation Spectrogram')
            st.pyplot(fig)

        elif method == 'Speaker Characteristic Analysis':
            st.sidebar.header('Speaker Analysis Parameters')
    
            # File uploaders
            original_files = st.sidebar.file_uploader(
                'Upload Original Recordings',
                type=['wav', 'mp3', 'ogg', 'flac'],
                accept_multiple_files=True
            )
            suspect_file = st.sidebar.file_uploader(
                'Upload Suspected Recording',
                type=['wav', 'mp3', 'ogg', 'flac']
            )
            
            # Model parameters
            n_components = st.sidebar.slider(
                'Number of GMM Components',
                min_value=1, max_value=32, value=16, step=1
            )
            
            if original_files and suspect_file:
                st.subheader('Speaker Characteristic Analysis')

                # Process Original Recordings
                original_features_list = []
                for idx, file in enumerate(original_files):
                    st.write(f'Processing Original Recording {idx + 1}')
                    # Read the audio file
                    y_orig, sr_orig = sf.read(file)
                    y_orig = y_orig.astype(np.float32)
                    # Resample if necessary
                    if sr_orig != 22050:
                        y_orig = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=22050)
                        sr_orig = 22050
                    # Extract features
                    features = extract_features(y_orig, sr_orig)
                    original_features_list.append(features)

                original_features = np.array(original_features_list)
                n_samples = len(original_features)

                # Calculate maximum number of components
                max_components = min(32, n_samples)
                if max_components < 1:
                    max_components = 1

                # Model parameters
                n_components = st.sidebar.slider(
                    'Number of GMM Components',
                    min_value=1, max_value=max_components, value=min(16, max_components), step=1
                )

                st.write(f'Setting number of GMM components to {n_components}')

                # Build Speaker Model
                st.write('Building Speaker Model...')
                gmm = build_speaker_model(original_features, n_components=n_components)
                
                # Process Suspected Recording
                st.write('Processing Suspected Recording')
                y_suspect, sr_suspect = sf.read(suspect_file)
                y_suspect = y_suspect.astype(np.float32)
                if sr_suspect != 22050:
                    y_suspect = librosa.resample(y_suspect, orig_sr=sr_suspect, target_sr=22050)
                    sr_suspect = 22050
                # Extract features
                suspect_features = extract_features(y_suspect, sr_suspect)
                
                # Compare Suspected Recording with Speaker Model
                st.write('Analyzing Suspected Recording...')
                log_likelihood = gmm.score(suspect_features.reshape(1, -1))
                original_scores = gmm.score_samples(original_features)
                mean_score = np.mean(original_scores)
                std_score = np.std(original_scores)
                suspect_z_score = (log_likelihood - mean_score) / std_score
                
                # Display Results
                st.write(f'Log-Likelihood of Suspected Recording: **{log_likelihood:.2f}**')
                st.write(f'Mean Log-Likelihood of Original Recordings: **{mean_score:.2f}**')
                st.write(f'Z-score of Suspected Recording: **{suspect_z_score:.2f}**')
                
                # Interpretation
                if np.abs(suspect_z_score) > 2:
                    st.warning('The suspected recording significantly deviates from the speaker model.')
                    st.write('This may indicate the presence of a watermark or alterations.')
                else:
                    st.success('The suspected recording is consistent with the speaker model.')
                    st.write('No significant anomalies detected.')
                
                # Visualizations
                # Plot Mean MFCCs
                st.write('Comparing MFCCs')
                mfccs_original_mean = np.mean(original_features[:, :20], axis=0)
                mfccs_suspect = suspect_features[:20]
                
                fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 5))
                ax_mfcc.plot(mfccs_original_mean, label='Original Mean MFCCs')
                ax_mfcc.plot(mfccs_suspect, label='Suspected MFCCs')
                ax_mfcc.set_title('MFCC Comparison')
                ax_mfcc.set_xlabel('MFCC Coefficient Index')
                ax_mfcc.set_ylabel('Amplitude')
                ax_mfcc.legend()
                st.pyplot(fig_mfcc)
                
                # Pitch Comparison
                st.write('Comparing Pitch')
                pitch_original_mean = np.mean(original_features[:, 40])
                pitch_suspect = suspect_features[40]
                
                fig_pitch, ax_pitch = plt.subplots(figsize=(6, 4))
                ax_pitch.bar(['Original Mean Pitch', 'Suspected Pitch'], [pitch_original_mean, pitch_suspect])
                ax_pitch.set_ylabel('Pitch (Hz)')
                st.pyplot(fig_pitch)
                
                # Spectral Centroid Comparison
                st.write('Comparing Spectral Centroid')
                sc_original_mean = np.mean(original_features[:, 42])
                sc_suspect = suspect_features[42]
                
                fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
                ax_sc.bar(['Original Mean Spectral Centroid', 'Suspected Spectral Centroid'], [sc_original_mean, sc_suspect])
                ax_sc.set_ylabel('Frequency (Hz)')
                st.pyplot(fig_sc)
                
                # Log-Likelihood Scores Distribution
                st.write('Log-Likelihood Scores of Original Recordings')
                fig_ll, ax_ll = plt.subplots(figsize=(8, 4))
                ax_ll.hist(original_scores, bins=10, alpha=0.7, label='Original Recordings')
                ax_ll.axvline(log_likelihood, color='red', linestyle='dashed', linewidth=2, label='Suspected Recording')
                ax_ll.set_xlabel('Log-Likelihood Score')
                ax_ll.set_ylabel('Frequency')
                ax_ll.legend()
                st.pyplot(fig_ll)
            else:
                st.info('Please upload the original recordings and the suspected recording to proceed.')
        
        elif method == 'Watermark Analysis':
            st.sidebar.header('Watermark Analysis Parameters')

            # File uploaders
            original_file = st.sidebar.file_uploader('Upload Original Audio', type=['wav', 'mp3'])
            watermarked_file = st.sidebar.file_uploader('Upload Watermarked Audio', type=['wav', 'mp3'])

            if original_file and watermarked_file:
                # Load audio files
                y_original, sr_original = librosa.load(original_file, sr=None)
                y_watermarked, sr_watermarked = librosa.load(watermarked_file, sr=None)

                # Resample if necessary
                if sr_original != sr_watermarked:
                    y_watermarked = librosa.resample(y_watermarked, orig_sr=sr_watermarked, target_sr=sr_original)
                    sr_watermarked = sr_original

                # Align signals
                min_length = min(len(y_original), len(y_watermarked))
                y_original = y_original[:min_length]
                y_watermarked = y_watermarked[:min_length]

                # Analyze Watermark Components
                st.subheader('Watermark Component Identification')

                # Residual Signal
                residual = y_watermarked - y_original
                st.write('Residual Signal')
                plt.figure(figsize=(12, 4))
                plt.plot(residual)
                plt.title('Residual Signal (Watermarked - Original)')
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                st.pyplot(plt)

                # Spectrogram Comparison
                st.subheader('Spectrogram Comparison')
                # Plot spectrograms
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                librosa.display.specshow(
                    librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max),
                    sr=sr, ax=ax[0], y_axis='log', x_axis='time'
                )
                ax[0].set_title('Original Audio')

                librosa.display.specshow(
                    librosa.amplitude_to_db(np.abs(librosa.stft(y_watermarked)), ref=np.max),
                    sr=sr, ax=ax[1], y_axis='log', x_axis='time'
                )
                ax[1].set_title('Watermarked Audio')

                st.pyplot(fig)

                # Audio Quality Metrics
                st.subheader('Audio Quality Metrics')

                # Compute SNR
                snr_value = compute_snr(y_original, y_watermarked)
                st.write(f'Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB')

                # Feature Space Visualization
                # Extract features from original and watermarked audio
                features_original = extract_features(y_original, sr)
                features_watermarked = extract_features(y_watermarked, sr)

                # Combine features
                features = np.vstack([features_original, features_watermarked])
                labels = ['Original', 'Watermarked']

                # Apply PCA
                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features)

                # Plot
                plt.figure(figsize=(6, 6))
                plt.scatter(features_pca[0, 0], features_pca[0, 1], label='Original')
                plt.scatter(features_pca[1, 0], features_pca[1, 1], label='Watermarked')
                plt.title('Feature Space Visualization using PCA')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend()
                st.pyplot(plt)

                # Conclusions
                st.subheader('Conclusions')
                st.write('Based on the analysis, we can observe how the watermark affects the audio signal...')
            else:
                st.info('Please upload both the original and watermarked audio files to proceed.')
        
        elif method == 'Statistical Features':
            # Extract features
            features = extract_statistical_features(y)
            st.subheader('Statistical Features')
            st.write(features)
        
        elif method == 'ICA':
            # ICA Parameters
            st.sidebar.header('ICA Parameters')

            # Let the user choose between the two ICA methods
            ica_type = st.sidebar.selectbox(
                'Select ICA Implementation',
                ('ICA on Delayed Signal', 'ICA on Spectrogram')
            )

            if ica_type == 'ICA on Delayed Signal':
                # ICA on Delayed Signal Parameters
                delays = st.sidebar.slider(
                    'Number of Delays', min_value=1, max_value=10, value=2
                )
                max_components = delays + 1
                n_components = st.sidebar.slider(
                    'Number of Components', min_value=1, max_value=max_components, value=2
                )

                # Perform ICA on delayed signal
                S_, A_ = perform_ica_delayed(y, n_components, delays)

                # Align the original audio with the components
                y_aligned = y[delays:]  # Truncate to match length

                # Initialize a list to keep track of selected components
                selected_components = []

                # Plot Independent Components (Waveforms and Spectrograms)
                st.subheader('Independent Components (Delayed Signal)')
                for i in range(n_components):
                    st.write(f'**Component {i + 1}**')

                    # Create a checkbox for selecting the component
                    is_selected = st.checkbox(f'Select Component {i + 1}', key=f'select_{i}')
                    if is_selected:
                        selected_components.append(i)

                    # Plot waveform
                    fig_waveform, ax_waveform = plt.subplots(figsize=(12, 2))
                    ax_waveform.plot(S_[:, i])
                    ax_waveform.set_title(f'Waveform of Component {i + 1}')
                    st.pyplot(fig_waveform)

                    # Compute and plot spectrogram
                    component_audio = S_[:, i]
                    n_fft = 1024  # You can adjust this value
                    hop_length = n_fft // 4
                    D = librosa.stft(component_audio, n_fft=n_fft, hop_length=hop_length)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    fig_spectrogram, ax_spectrogram = plt.subplots(figsize=(12, 4))
                    img = librosa.display.specshow(
                        S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax_spectrogram
                    )
                    fig_spectrogram.colorbar(img, ax=ax_spectrogram, format="%+2.0f dB")
                    ax_spectrogram.set_title(f'Spectrogram of Component {i + 1}')
                    st.pyplot(fig_spectrogram)

                    # Normalize the component audio
                    component_audio_norm = component_audio / (np.max(np.abs(component_audio)) or 1)

                    # Provide play and download buttons
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, component_audio_norm, sr, subtype='PCM_16', format='WAV')
                    audio_buffer.seek(0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.audio(audio_buffer, format='audio/wav', start_time=0)
                    with col2:
                        st.download_button(
                            label=f'Download Component {i + 1} as WAV',
                            data=audio_buffer,
                            file_name=f'component_{i + 1}.wav',
                            mime='audio/wav'
                        )

                # Buttons for Reconstruction and Subtraction
                st.subheader('Reconstruction and Subtraction')

                # Check if any components are selected
                if selected_components:
                    col_reconstruct, col_subtract = st.columns(2)
                    with col_reconstruct:
                        if st.button('Reconstruct Audio from Selected Components'):
                            # Reconstruct audio using selected components
                            reconstructed_audio = np.sum(S_[:, selected_components], axis=1)
                            reconstructed_audio /= np.max(np.abs(reconstructed_audio)) or 1
                            reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)
                            audio_buffer_reconstructed = io.BytesIO()
                            sf.write(audio_buffer_reconstructed, reconstructed_audio_int16, sr, subtype='PCM_16', format='WAV')
                            audio_buffer_reconstructed.seek(0)
                            st.write('Playing Reconstructed Audio (Using Selected Components)')
                            st.audio(audio_buffer_reconstructed, format='audio/wav')
                            st.download_button(
                                label='Download Reconstructed Audio',
                                data=audio_buffer_reconstructed,
                                file_name='reconstructed_audio.wav',
                                mime='audio/wav'
                            )

                    with col_subtract:
                        if st.button('Subtract Selected Components from Original Audio'):
                            # Sum selected components
                            components_sum = np.sum(S_[:, selected_components], axis=1)
                            components_sum /= np.max(np.abs(components_sum)) or 1
                            # Subtract from original audio
                            reconstructed_audio = y_aligned - components_sum
                            reconstructed_audio /= np.max(np.abs(reconstructed_audio)) or 1
                            reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)
                            audio_buffer_subtract = io.BytesIO()
                            sf.write(audio_buffer_subtract, reconstructed_audio_int16, sr, subtype='PCM_16', format='WAV')
                            audio_buffer_subtract.seek(0)
                            st.write('Playing Audio (Original Minus Selected Components)')
                            st.audio(audio_buffer_subtract, format='audio/wav')
                            st.download_button(
                                label='Download Subtracted Audio',
                                data=audio_buffer_subtract,
                                file_name='subtracted_audio.wav',
                                mime='audio/wav'
                            )
                else:
                    st.info('Please select at least one component to reconstruct or subtract.')

            elif ica_type == 'ICA on Spectrogram':
                # ICA on Spectrogram Parameters
                n_components = st.sidebar.slider(
                    'Number of Components', min_value=1, max_value=10, value=2
                )
                n_fft = st.sidebar.slider(
                    'Window Size (n_fft)', min_value=256, max_value=4096,
                    value=1024, step=256
                )
                hop_length = st.sidebar.slider(
                    'Hop Length', min_value=64, max_value=1024,
                    value=512, step=64
                )

                # Perform ICA on spectrogram
                S_, S, ica = perform_ica_spectrogram(y, sr, n_components, n_fft, hop_length)

                # Plot Independent Components
                st.subheader('Independent Components (Spectrogram)')
                fig, axs = plt.subplots(n_components, 1, figsize=(12, 2 * n_components))
                for i in range(n_components):
                    axs[i].plot(S_[:, i])
                    axs[i].set_title(f'Component {i + 1}')
                plt.tight_layout()
                st.pyplot(fig)

                # Reconstruct and plot the spectrogram components
                st.subheader('Reconstructed Spectrogram Components')
                for i in range(n_components):
                    component_spectrogram = np.outer(S_[:, i], ica.mixing_[:, i])
                    component_spectrogram = component_spectrogram.T  # Transpose back

                    # **Estimate the phase using Griffin-Lim and reconstruct the time-domain signal**
                    reconstructed_audio = griffinlim(
                        component_spectrogram,
                        n_iter=32,
                        hop_length=hop_length,
                        win_length=n_fft
                    )

                    # Normalize the audio
                    reconstructed_audio /= np.max(np.abs(reconstructed_audio))

                    # Plot the spectrogram
                    fig, ax = plt.subplots(figsize=(12, 4))
                    S_dB = librosa.amplitude_to_db(np.abs(component_spectrogram), ref=np.max)
                    img = librosa.display.specshow(
                        S_dB, sr=sr, hop_length=hop_length,
                        x_axis='time', y_axis='log', ax=ax
                    )
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    ax.set_title(f'Spectrogram of Component {i + 1}')
                    st.pyplot(fig)

                    # **Export the reconstructed audio**
                    st.write(f'Download Component {i + 1} as Audio')
                    # Convert to 16-bit PCM
                    reconstructed_audio_int16 = np.int16(reconstructed_audio * 32767)
                    # Create a BytesIO object
                    audio_buffer = io.BytesIO()
                    # Write the audio to the buffer
                    sf.write(audio_buffer, reconstructed_audio_int16, sr, subtype='PCM_16', format='WAV')
                    audio_buffer.seek(0)
                    # Provide a download button
                    st.download_button(
                        label=f'Download Component {i + 1} as WAV',
                        data=audio_buffer,
                        file_name=f'component_{i + 1}.wav',
                        mime='audio/wav'
                    )

    else:
        st.info('Please upload an audio file to proceed.')

if __name__ == '__main__':
    main()
