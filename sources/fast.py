import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Helper function for interpolation and segmentation
def _interpolate_and_segment(signal_values, max_nan_interpolation):
    """
    Interpolates short NaN gaps and identifies segments separated by long gaps.
    
    Args:
        signal_values (np.ndarray): The 1D signal array, potentially containing NaNs.
        max_nan_interpolation (int): Max consecutive NaNs to interpolate linearly.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The signal with short NaN gaps interpolated.
            - list[tuple[int, int]]: A list of (start_idx, end_idx) tuples
                                      representing valid data segments.
    """
    signal_processed = signal_values.copy().astype(float) # Work with float copy
    nan_mask = np.isnan(signal_processed)
    valid_segments = []
    start_idx = 0

    if not np.any(nan_mask): # No NaNs, return the whole signal as one segment
        return signal_processed, [(0, len(signal_processed))]

    i = 0
    while i < len(signal_processed):
        if nan_mask[i]:
            # Found start of a NaN sequence
            nan_start = i
            while i < len(signal_processed) and nan_mask[i]:
                i += 1
            nan_end = i
            nan_count = nan_end - nan_start

            # Define the segment before the NaN gap
            current_segment_end = nan_start
            if current_segment_end > start_idx:
                valid_segments.append((start_idx, current_segment_end))

            if 0 < nan_count <= max_nan_interpolation:
                # Interpolate short gaps
                start_val = signal_processed[nan_start - 1] if nan_start > 0 else np.nan
                end_val = signal_processed[nan_end] if nan_end < len(signal_processed) else np.nan

                if not np.isnan(start_val) and not np.isnan(end_val):
                    interp_values = np.linspace(start_val, end_val, num=nan_count + 2)[1:-1]
                    signal_processed[nan_start:nan_end] = interp_values
                elif not np.isnan(start_val): # Fill forward
                    signal_processed[nan_start:nan_end] = start_val
                elif not np.isnan(end_val): # Fill backward
                    signal_processed[nan_start:nan_end] = end_val
                # Else: Cannot interpolate.

            # Regardless of interpolation or gap size, the next valid segment search starts after the gap
            start_idx = nan_end
        else:
            i += 1

    # Add the last segment if it wasn't followed by NaNs
    if start_idx < len(signal_processed):
        valid_segments.append((start_idx, len(signal_processed)))

    # Refine segments: Remove potential NaNs at segment edges (important if interp failed at edges)
    final_segments = []
    for start, end in valid_segments:
        seg_data_for_check = signal_processed[start:end]
        
        first_valid = 0
        nan_at_start = np.where(np.isnan(seg_data_for_check))[0]
        if len(nan_at_start) > 0 and nan_at_start[0] == 0:
             first_valid_indices = np.where(~np.isnan(seg_data_for_check))[0]
             if len(first_valid_indices) > 0:
                 first_valid = first_valid_indices[0]
             else: continue # Skip segment if all NaNs
        start += first_valid

        seg_data_for_check = signal_processed[start:end] # Re-check with potentially new start
        if len(seg_data_for_check) == 0: continue # Skip if segment became empty
        
        last_valid_rel = len(seg_data_for_check) - 1
        nan_at_end = np.where(np.isnan(seg_data_for_check))[0]
        if len(nan_at_end) > 0 and nan_at_end[-1] == len(seg_data_for_check) - 1:
             last_valid_rel_indices = np.where(~np.isnan(seg_data_for_check))[0]
             if len(last_valid_rel_indices) > 0:
                 last_valid_rel = last_valid_rel_indices[-1]
             else: continue # Skip segment if all NaNs

        end = start + last_valid_rel + 1
                 
        if start < end:
            final_segments.append((start, end))

    return signal_processed, final_segments


def qrs_detector(signal_data, freq_sampling: int, 
                 max_nan_interpolation=0, min_segment_len=50,
                 return_signal = False):
    """
    Detects QRS complexes in an ECG signal.

    Args:
        signal_data (np.ndarray | pd.DataFrame): The ECG signal data.
        freq_sampling (int): The sampling frequency of the signal in Hz.
        max_nan_interpolation (int, optional): For 2-column DF input. Max consecutive
            NaNs to interpolate linearly. Defaults to 0.
        min_segment_len (int, optional): For 2-column DF input. Minimum length of a
            data segment (after splitting by large NaN gaps) to process. Defaults to 50.
        return_signal (bool, optional): If True, returns a tuple containing the
            QRS detections and the processed signal (after NaN interpolation for
            2-column DataFrame input). Defaults to False.

    Returns:
        np.ndarray | pd.Index | tuple[np.ndarray | pd.Index, np.ndarray]:
            - If `return_signal` is False (default): Indices (for np.ndarray/1-col DF)
              or timestamps (for 2-col DF) of the detected QRS peaks.
            - If `return_signal` is True: A tuple `(qrs_detections, processed_signal)`,
              where `qrs_detections` is as above, and `processed_signal` is a
              1D NumPy array of the signal after NaN interpolation (if applicable).

    Raises:
        ValueError: If input format is incorrect or contains NaNs inappropriately.
        TypeError: If input `signal_data` is not a NumPy array or pandas DataFrame.
    """
    is_two_col_dataframe = False
    timestamps = None
    signal_values = None

    # --- Input Validation and Data Extraction --- 
    if isinstance(signal_data, pd.DataFrame):
        num_cols = signal_data.shape[1]
        if num_cols == 1:
            col_name = signal_data.columns[0]
            if pd.api.types.is_numeric_dtype(signal_data[col_name]) and not pd.api.types.is_bool_dtype(signal_data[col_name]):
                signal_values = signal_data[col_name].to_numpy()
                if np.any(np.isnan(signal_values)):
                    raise ValueError("NaN values found in 1-column DataFrame input. Interpolation is only supported for 2-column DataFrame input.")
            else:
                raise ValueError("Input DataFrame with one column must have a numeric signal column.")
        elif num_cols == 2:
            signal_col = None
            time_col = None
            for col in signal_data.columns:
                if pd.api.types.is_numeric_dtype(signal_data[col]) and not pd.api.types.is_bool_dtype(signal_data[col]):
                    if signal_col is not None: raise ValueError("Input DataFrame has more than one numeric column.")
                    signal_col = col
                elif pd.api.types.is_datetime64_any_dtype(signal_data[col]) or pd.api.types.is_timedelta64_dtype(signal_data[col]):
                     if time_col is not None: raise ValueError("Input DataFrame has more than one potential time column.")
                     time_col = col
            if signal_col is None or time_col is None: raise ValueError(f"Could not identify one signal and one time column. Found columns: {signal_data.columns.tolist()}")
            timestamps = signal_data[time_col]
            signal_values = signal_data[signal_col].to_numpy()
            is_two_col_dataframe = True
        else: raise ValueError(f"Input DataFrame must have one or two columns. Found {num_cols}.")
    elif isinstance(signal_data, np.ndarray):
        if signal_data.ndim != 1: raise ValueError("Input NumPy array must be 1-dimensional.")
        signal_values = signal_data
        if np.any(np.isnan(signal_values)): raise ValueError("NaN values found in NumPy array input.")
    else: raise TypeError("Input 'signal_data' must be a pandas DataFrame or a NumPy array.")
    # --- End Input Validation --- #

    # --- Processing --- #
    all_qrs_indices = []
    signal_for_return = None # Pour stocker le signal à retourner si demandé

    if is_two_col_dataframe:
        # --- Interpolation and Segmentation --- # 
        signal_processed_df, valid_segments = _interpolate_and_segment(
            signal_values, max_nan_interpolation
        )
        signal_for_return = signal_processed_df # Signal après interpolation

        # --- Process EACH valid segment --- #
        for start_idx, end_idx in valid_segments:
            segment_signal = signal_for_return[start_idx:end_idx] # Utiliser le signal potentiellement interpolé
            
            if len(segment_signal) <= min_segment_len:
                 continue

            if np.any(np.isnan(segment_signal)):
                 print(f"Warning: Segment {start_idx}-{end_idx} contains NaNs after interpolation attempt. Skipping.")
                 continue

            # Apply QRS detection algorithm to the valid segment
            try:
                cleaned_ecg = preprocess_ecg(segment_signal, freq_sampling, 5, 22, size_window=int(0.1 * freq_sampling))
                peaks = detect_peaks(cleaned_ecg, no_peak_distance=int(freq_sampling * 0.65), distance=int(freq_sampling * 0.33))
                qrs_indices_segment = threshold_detection(cleaned_ecg, peaks, freq_sampling, initial_search_samples=int(freq_sampling * 0.83), long_peak_distance=int(freq_sampling * 1.111))
            except Exception as e_algo:
                 print(f"Warning: Algorithm failed on segment {start_idx}-{end_idx}: {e_algo}. Skipping segment.")
                 qrs_indices_segment = np.array([]) 

            qrs_indices_original = qrs_indices_segment + start_idx
            all_qrs_indices.extend(qrs_indices_original)

    else:
        # --- Direct processing (NumPy array or 1-column DataFrame) --- #
        # Pas d'interpolation de NaN via _interpolate_and_segment ici.
        # Les NaNs auraient levé une erreur plus tôt.
        signal_for_return = signal_values.copy() # Le signal "traité" est le signal d'origine

        cleaned_ecg = preprocess_ecg(signal_values, freq_sampling, 5, 22, size_window=int(0.1 * freq_sampling))
        peaks = detect_peaks(cleaned_ecg, no_peak_distance=int(freq_sampling * 0.65), distance=int(freq_sampling * 0.33))
        qrs_indices = threshold_detection(cleaned_ecg, peaks, freq_sampling, initial_search_samples=int(freq_sampling * 0.83), long_peak_distance=int(freq_sampling * 1.111))
        all_qrs_indices = qrs_indices

    # --- Output --- #
    final_qrs_output_indices = np.sort(np.unique(all_qrs_indices)) # Renommé pour clarté
    
    qrs_detections_to_return = None # Variable pour stocker la sortie QRS finale

    if is_two_col_dataframe:
        try:
            # Utiliser final_qrs_output_indices qui sont les indices numériques
            qrs_detections_to_return = timestamps.iloc[final_qrs_output_indices] 
        except IndexError as ie:
             print(f"ERROR: IndexError selecting timestamps. Indices might be out of bounds.") 
             print(f"  Max index requested: {final_qrs_output_indices.max() if len(final_qrs_output_indices)>0 else 'N/A'}, Timestamps length: {len(timestamps)}")
             raise ie 
        except Exception as oe:
             print(f"ERROR: Unexpected error selecting timestamps: {oe}")
             raise oe
    else:
        qrs_detections_to_return = final_qrs_output_indices

    if return_signal:
        return qrs_detections_to_return, signal_for_return
    else:
        return qrs_detections_to_return

# --- Core Algorithm Functions --- #
def detect_peaks(cleaned_ecg, no_peak_distance, distance=0):
    last_max = -np.inf  # The most recent encountered maximum value
    last_max_pos = -1  # Position of the last_max in the array
    peaks = [np.argmax(cleaned_ecg[:no_peak_distance])]  # Detected peaks positions
    peak_values = [cleaned_ecg[peaks[0]]]  # Detected peaks values

    
    for i, current_value in enumerate(cleaned_ecg):
       
        # Update the most recent maximum if the current value is greater
        if current_value > last_max:
            last_max = current_value
            last_max_pos = i
        
        # Check if the current value is less than half the last max
        # or if we are beyond the no_peak_distance from the last max
        if current_value <= last_max / 2 or (i - last_max_pos >= no_peak_distance):
            # Check if the last peak is within the `distance` of the current peak
            if last_max_pos - peaks[-1] < distance:
                # If within the distance, choose the higher peak
                if last_max > peak_values[-1]:
                    peaks[-1] = last_max_pos
                    peak_values[-1] = last_max
            else:
                # Otherwise, start a new peak group
                peaks.append(last_max_pos)
                peak_values.append(last_max)
            
            # Reset the last max after adding a peak
            last_max = current_value
            last_max_pos = i
    
    return np.array(peaks)

def threshold_detection(cleaned_ecg, peaks, fs, initial_search_samples=300, long_peak_distance=400):

    spk = 0.13 * np.max(cleaned_ecg[:initial_search_samples])
    npk = 0.1 * spk
    threshold = 0.25 * spk + 0.75 * npk
    
    qrs_peaks = []
    noise_peaks = []
    qrs_buffer = []
    last_qrs_time = 0
    min_distance = int(fs * 0.12)
    
    for i, peak in enumerate(peaks):
        peak_value = cleaned_ecg[peak]
        
        if peak_value > threshold:
            if qrs_peaks and (peak - qrs_peaks[-1] < min_distance):
                if peak_value > cleaned_ecg[qrs_peaks[-1]]:
                    qrs_peaks[-1] = peak
            else:
                qrs_peaks.append(peak)
                last_qrs_time = peak
            
            spk = 0.25 * peak_value + 0.75 * spk
            
            qrs_buffer.append(peak)
            if len(qrs_buffer) > 10:
                qrs_buffer.pop(0)
        else:
            noise_peaks.append(peak)
            npk = 0.25 * peak_value + 0.75 * npk
        
        threshold = 0.25 * spk + 0.75 * npk
        
        if peak - last_qrs_time > long_peak_distance:
            spk *= 0.5
            threshold = 0.25 * spk + 0.75 * npk
            for lookback_peak in peaks[i-5:i+1]:
                if lookback_peak != last_qrs_time:
                    if last_qrs_time < lookback_peak < peak and cleaned_ecg[lookback_peak] > threshold:
                        qrs_peaks.append(lookback_peak)
                        spk = 0.875 * spk + 0.125 * cleaned_ecg[lookback_peak]
                        threshold = 0.25 * spk + 0.75 * npk
                        last_qrs_time = lookback_peak
                        break
        
        if len(qrs_buffer) > 1:
            rr_intervals = np.diff(qrs_buffer)
            mean_rr = np.mean(rr_intervals)
            if peak - last_qrs_time > 1.5 * mean_rr:
                spk *= 0.5
                threshold = 0.25 * spk + 0.75 * npk
    
    return np.array(qrs_peaks)

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    y = filtfilt(b, a, data)
    return y

def differentiate(data):
    return np.diff(data, prepend=data[0])

def squaring(data):
    return np.square(data)

def moving_window_integration(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_ecg(data, fs, high, low, size_window):
    signal = highpass_filter(data, high, fs)
    signal = lowpass_filter(signal, low, fs)
    signal = differentiate(signal)
    signal = squaring(signal)
    signal = moving_window_integration(signal, size_window)
    return signal