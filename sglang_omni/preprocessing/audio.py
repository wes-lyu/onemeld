# SPDX-License-Identifier: Apache-2.0
"""Decode PCM or floating-point WAV for speech inputs."""

import struct

import numpy as np
import numpy.typing as npt


def parse_wav_bytes(
    data: bytes, source: str = "bytes"
) -> tuple[npt.NDArray[np.float32], int]:
    """Parse PCM/IEEE-float WAV from bytes without external deps."""
    if len(data) < 12:
        raise ValueError(f"Invalid WAV header: {source}")
    else:
        pass

    header = data[:12]
    riff, _, wave = struct.unpack("<4sI4s", header)
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {source}")
    else:
        pass

    fmt_tag = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    data_bytes = b""

    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            break
        else:
            pass
        chunk_header = data[offset : offset + 8]
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        offset += 8

        if offset + chunk_size > len(data):
            break
        else:
            pass
        chunk_data = data[offset : offset + chunk_size]
        offset += chunk_size
        if chunk_size % 2 == 1:
            offset += 1
        else:
            pass

        if chunk_id == b"fmt ":
            if len(chunk_data) >= 16:
                fmt_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", chunk_data[:16]
                )
            else:
                pass
        elif chunk_id == b"data":
            data_bytes = chunk_data
        else:
            pass

    if fmt_tag is None or sample_rate is None or bits_per_sample is None:
        raise ValueError(f"Missing fmt chunk in WAV: {source}")
    else:
        pass
    if not data_bytes:
        raise ValueError(f"Missing data chunk in WAV: {source}")
    else:
        pass

    if fmt_tag == 3:  # IEEE float
        if bits_per_sample == 32:
            audio = np.frombuffer(data_bytes, dtype="<f4")
        elif bits_per_sample == 64:
            audio = np.frombuffer(data_bytes, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"Unsupported float WAV bit depth: {bits_per_sample}")
    elif fmt_tag == 1:  # PCM
        if bits_per_sample == 16:
            audio_i16 = np.frombuffer(data_bytes, dtype="<i2")
            audio = (audio_i16.astype(np.float32) / 32768.0).astype(np.float32)
        elif bits_per_sample == 32:
            audio_i32 = np.frombuffer(data_bytes, dtype="<i4")
            audio = (audio_i32.astype(np.float32) / 2147483648.0).astype(np.float32)
        elif bits_per_sample == 8:
            audio_u8 = np.frombuffer(data_bytes, dtype="u1")
            audio = ((audio_u8.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported PCM WAV bit depth: {bits_per_sample}")
    else:
        raise ValueError(f"Unsupported WAV format tag: {fmt_tag}")

    if channels and channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    else:
        pass

    return audio.astype(np.float32, copy=False), int(sample_rate)
