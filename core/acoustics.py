import math
import numpy as np


def rt60(signal: np.array, fs: int = 16000) -> float:
    signal = signal / (np.sqrt(np.sum(signal**2)) + 1e-12)

    edc = np.cumsum(signal[::-1]**2)[::-1]
    edc_db = 10*np.log10(edc/np.max(edc) + 1e-12)
    start, end = np.argmax(edc_db < -5), np.argmax(edc_db < -35)
    if end <= start:
        return np.nan
    t = np.arange(len(signal))/fs
    slope, intercept = np.polyfit(t[start:end], edc_db[start:end], deg=1)

    return -60/slope


def edt(signal: np.array, fs: int = 16000) -> float:
    signal = signal / (np.sqrt(np.sum(signal**2)) + 1e-12)

    edc = np.cumsum(signal[::-1]**2)[::-1]
    edc_db = 10*np.log10(edc/np.max(edc) + 1e-12)
    start, end = 0, np.argmax(edc_db < -10)
    if end <= start:
        return np.nan
    t = np.arange(len(signal))/fs
    slope, intercept = np.polyfit(t[start:end], edc_db[start:end], deg=1)

    return -60/slope


def c50(signal: np.array, fs: int = 16000) -> float:
    signal = signal / (np.sqrt(np.sum(signal**2)) + 1e-12)  # normalize
    ind = math.floor(0.05 * fs)
    early_energy = np.sum(signal[:ind] ** 2)
    late_energy = np.sum(signal[ind:] ** 2)
    ratio = early_energy / (late_energy + 1e-12)
    ratio = np.clip(ratio, 1e-6, 1e6)  # clip to avoid extreme values
    return 10 * np.log10(ratio)


def d50(signal: np.array, fs: int = 16000) -> float:
    signal = signal / (np.sqrt(np.sum(signal**2)) + 1e-12)

    ind = math.floor(0.05*fs)
    early_energy = np.sum(signal[:ind]**2)
    total_energy = np.sum(signal**2)
    d50 = early_energy/(total_energy + 1e-12)
    return d50
