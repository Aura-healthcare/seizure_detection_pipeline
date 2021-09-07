import numpy as np
from ecgdetectors import Detectors
from wfdb import processing
import biosppy.signals.ecg as bsp_ecg
import biosppy.signals.tools as bsp_tools


class QRSDetector:

    def _detect_qrs_swt(self, ecg_data, fs):
        qrs_frames = []
        try:
            detectors = Detectors(fs)
            qrs_frames = detectors.swt_detector(ecg_data)
        except Exception:
            print("Exception in detect_qrs_swt")
        return qrs_frames

    def _detect_qrs_xqrs(self, ecg_data, fs):
        qrs_frames = []
        try:
            qrs_frames = processing.xqrs_detect(sig=ecg_data, fs=fs)
        except Exception:
            print("Exception in detect_qrs_xqrs")
        return qrs_frames.tolist()

    def _detect_qrs_gqrs(self, ecg_data, fs):
        qrs_frames = []
        try:
            qrs_frames = processing.qrs.gqrs_detect(sig=ecg_data, fs=fs*2)
        except Exception:
            print("Exception in detect_qrs_gqrs")
        return qrs_frames.tolist()

    def _detect_qrs_hamilton(self, ecg_data, fs):
        qrs_frames = []
        try:
            qrs_frames = bsp_ecg.ecg(
                signal=np.array(ecg_data), sampling_rate=fs, show=False)[2]
        except Exception:
            print("Exception in detect_qrs_gqrs")
        return qrs_frames.tolist()

    def _detect_qrs_engelsee(self, ecg_data, fs):
        qrs_frames = []
        try:
            qrs_frames = bsp_ecg.ecg(
                signal=ecg_data, sampling_rate=fs, show=False)[2]
            order = int(0.3 * fs)
            filtered, _, _ = bsp_tools.filter_signal(signal=ecg_data,
                                                     ftype='FIR',
                                                     band='bandpass',
                                                     order=order,
                                                     frequency=[3, 45],
                                                     sampling_rate=fs)
            rpeaks, = bsp_ecg.engzee_segmenter(
                signal=filtered, sampling_rate=fs)
            rpeaks, = bsp_ecg.correct_rpeaks(
                signal=filtered, rpeaks=rpeaks, sampling_rate=fs, tol=0.05)
            _, qrs_frames = bsp_ecg.extract_heartbeats(
                signal=filtered, rpeaks=rpeaks, sampling_rate=fs,
                before=0.2, after=0.4)
        except Exception:
            print("Exception in detect_qrs_gqrs")
        return qrs_frames.tolist()

    def _to_rr_intervals(self, frame_data, fs):
        rr_intervals = np.zeros(len(frame_data) - 1)
        for i in range(0, (len(frame_data) - 1)):
            rr_intervals[i] = (frame_data[i+1] - frame_data[i]) * 1000.0 / fs

        return rr_intervals

    # def _to_hr(self, rr_intervals):
    #     hr = np.zeros(len(rr_intervals))
    #     for i in range(0, len(rr_intervals)):
    #         hr[i] = (int)(60 * 1000 / rr_intervals[i])

    #     return hr

    def get_cardiac_infos(self, ecg_data, sampling_frequency, method):
        if method == "xqrs":
            qrs_frames = self._detect_qrs_xqrs(ecg_data, sampling_frequency)
        elif method == "gqrs":
            qrs_frames = self._detect_qrs_gqrs(ecg_data, sampling_frequency)
        elif method == "swt":
            qrs_frames = self._detect_qrs_swt(ecg_data, sampling_frequency)
        elif method == "hamilton":
            qrs_frames = self._detect_qrs_hamilton(
                ecg_data, sampling_frequency)
        else:
            qrs_frames = self._detect_qrs_engelsee(
                ecg_data, sampling_frequency)

        rr_intervals = np.zeros(0)
        # hr = np.zeros(0)
        if len(qrs_frames):
            rr_intervals = self._to_rr_intervals(
                qrs_frames, sampling_frequency)
        #   hr = to_hr(rr_intervals)
        return qrs_frames, rr_intervals  # , hr
