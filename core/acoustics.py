import math
import numpy as np


def rt60(signal: np.array, fs: int = 16000) -> float:
    edc = np.cumsum(signal[::-1]**2)[::-1]
    edc_db = 10*np.log10(edc/np.max(edc) + 1e-12)
    start, end = np.argmax(edc_db < -5), np.argmax(edc_db < -35)
    if end <= start:
        return np.nan
    t = np.arange(len(signal))/fs
    slope, intercept = np.polyfit(t[start:end], edc_db[start:end], deg=1)

    return -60/slope


def edt(signal: np.array, fs: int = 16000) -> float:
    edc = np.cumsum(signal[::-1]**2)[::-1]
    edc_db = 10*np.log10(edc/np.max(edc) + 1e-12)
    start, end = 0, np.argmax(edc_db < -10)
    if end <= start:
        return np.nan
    t = np.arange(len(signal))/fs
    slope, intercept = np.polyfit(t[start:end], edc_db[start:end], deg=1)

    return -60/slope


def c50(signal: np.array, fs: int = 16000) -> float:
    ind = math.floor(0.05*fs)
    c50 = 10*np.log10(np.sum(signal[:ind]**2)/(np.sum(signal[ind:]**2) + 1e-12))
    return c50

def d50(signal: np.array, fs: int = 16000) -> float:
    ind = math.floor(0.05*fs)
    early_energy = np.sum(signal[:ind]**2)
    total_energy = np.sum(signal**2)
    d50 = early_energy/(total_energy + 1e-12)
    return d50
