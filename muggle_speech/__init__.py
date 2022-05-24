#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import io
import time
import resampy
import numba
import wave
import librosa
from pydub import AudioSegment
import numpy as np
import onnxruntime
from muggle_speech.label_map import label_map


class PostProcess(object):

    def __init__(self):
        self.vec2char = label_map
        self.penalty = 0.6
        self.lamda = 5
        self.EOS = 0

    def decode(self, predicts, scores):
        scores = scores.reshape(1, 1)
        predicts = predicts.reshape(1, 1, -1)
        lengths = np.sum(np.not_equal(predicts, self.EOS), axis=-1)
        lp = np.power((self.lamda + lengths) / (self.lamda + 1), self.penalty)
        scores /= lp
        max_indices = np.argmax(scores, axis=-1)
        max_indices += np.arange(1, dtype=np.long) * 1
        preds = predicts.reshape(1 * 1, -1)
        best_preds = preds[max_indices, ...]
        best_preds = best_preds[:, 1:]
        results = []
        for pred in best_preds.tolist():
            preds = []
            for i in pred:
                if int(i) == 0:
                    break
                preds.append(self.vec2char[int(i)])
            results.append(''.join(preds))
        return "".join(results)


class PreProcess(object):

    def __init__(self, mode):
        self.process_audio_bytes_func = {
            'pydub': self.process_audio_bytes_from_pydub,
            'wave': self.process_audio_bytes_from_wave,
            'librosa': self.process_audio_bytes_from_librosa
        }[mode]

    @classmethod
    def normalization(cls, feature):
        mean = np.mean(feature)
        std = np.std(feature)
        return (feature - mean) / std

    def process_audio_bytes_from_pydub(self, wav_bytes):
        sound = AudioSegment(wav_bytes)
        sig = np.array(sound.get_array_of_samples(), dtype=np.float32) / 32768  # 16 bit
        sig = resampy.core.resample(sig, sound.frame_rate, 22050, res_type='kaiser_best')
        return self.extract_fbank(sig, 22050)

    def process_audio_bytes_from_wave(self, wav_bytes):
        with io.BytesIO(wav_bytes) as data_stream:
            f = wave.Wave_read(data_stream)
            params = f.getparams()
            channels, sample_width, frame_rate, n_frames = params[:4]
            str_data = f.readframes(n_frames)
            sig = np.frombuffer(str_data, dtype=np.int16)/32768
            sig = resampy.core.resample(sig, frame_rate, 22050, res_type='kaiser_best')
            return self.extract_fbank(sig, 22050)

    def process_audio_bytes_from_librosa(self, wav_bytes):
        with io.BytesIO(wav_bytes) as data_stream:
            sig, sr = librosa.load(data_stream, mono=True)
            return self.extract_fbank(sig, sr)

    def process_audio_file(self, file_name):
        wav_bytes = open(file_name, "rb").read()
        return self.process_audio_bytes_func(wav_bytes)

    @classmethod
    def extract_fbank(cls, sig, sr):

        emphasized_signal = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
        frame_length, frame_step = 0.025 * sr, 0.01 * sr
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(
            np.arange(0, frame_length),
            (num_frames, 1)
        ) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)
        nfft, nfilt = 512, 40
        mag_frames = np.absolute(np.fft.rfft(frames, nfft))
        pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
        low_freq_mel = (2595 * np.log10(1 + 0 / 700))
        high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        bin = np.floor((nfft + 1) * hz_points / sr)

        fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        filter_banks = filter_banks.transpose()
        fbank_feature = filter_banks
        fbank_feature = fbank_feature.transpose()
        fbank_feature = np.asarray(fbank_feature, dtype=np.float32)
        fbank_feature = cls.normalization(fbank_feature)
        return fbank_feature[np.newaxis, :]


class MuggleSpeech:
    def __init__(self, max_len=1000, mode='wave'):
        self.max_len = max_len
        self.encoder_sess = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "encoder.onnx"))
        self.decoder_sess = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "decoder.onnx"))
        self.preprocess = PreProcess(mode=mode)
        self.postprocess = PostProcess()

    def from_file(self, filepath):
        return self.preprocess.process_audio_file(filepath)

    def from_bytes(self, wav_bytes):
        return self.preprocess.process_audio_bytes_func(wav_bytes)

    def predict_file(self, inputs):
        preds, beam_enc_states, beam_enc_mask, scores, ending_flag = self.encoder_sess.run(
            ["preds", "beam_enc_states", "beam_enc_mask", "scores", "ending_flag"],
            {
                "inputs": inputs
            }
        )
        for step in range(1, self.max_len + 1):
            preds, scores, ending_flag = self.decoder_sess.run(
                ["preds_symbol", "scores", "end_flag"],
                {
                    "preds": preds, "beam_enc_states": beam_enc_states, "beam_enc_mask": beam_enc_mask,
                    "scores.1": scores,
                    "ending_flag": ending_flag
                }
            )
            if ending_flag.sum() == 1 * 1:
                break

        predict_text = self.postprocess.decode(preds, scores)
        return predict_text
