import subprocess
import warnings
from typing import Union, Optional, BinaryIO, Tuple, Iterable

import numpy as np
import torch
import torchaudio

from ..whisper_compatibility import SAMPLE_RATE


def is_ytdlp_available():
    return subprocess.run('yt-dlp -h', shell=True, capture_output=True).returncode == 0


def load_source(
        source: Union[str, bytes],
        verbose: Optional[bool] = False,
        only_ffmpeg: bool = False,
        return_dict: bool = False
) -> Union[bytes, dict]:
    """
    Return content downloaded by YT-DLP if ``source`` is a URL else return ``source``.
    """
    if not only_ffmpeg and isinstance(source, str) and '://' in source:
        if is_ytdlp_available():
            if return_dict:
                stderr = subprocess.PIPE
                verbosity = ' --no-simulate --print title,duration,is_live --no-warnings'
            else:
                verbosity = ' -q' if verbose is None else (' --progress' if verbose else ' --progress -q')
                stderr = None
            p = subprocess.Popen(
                f'yt-dlp "{source}" -f ba/w -I 1{verbosity} -o -',
                shell=True,
                stderr=stderr,
                stdout=subprocess.PIPE,
                bufsize=0
            )
            if return_dict:
                title = p.stderr.readline().decode('utf-8', errors='ignore').strip('\n') or None
                duration = p.stderr.readline().decode('utf-8', errors='ignore').strip('\n')
                is_live = p.stderr.readline().decode('utf-8', errors='ignore').strip('\n')
                try:
                    duration = int(duration)
                except ValueError:
                    duration = None
                is_live = (True if is_live == 'True' else False) if is_live in ('True', 'False') else None
                if verbose is not None:
                    print(f'Media Info (YT-DLP):\n'
                          f'-Title: "{title or "N/A"}"\n'
                          f'-Duration: {duration}s\n'
                          f'-Live: {is_live}')
                return dict(popen=p, title=title, duration=duration, is_live=is_live)
            return p.communicate()[0]
        else:
            warnings.warn('URL detected but yt-dlp not available. '
                          'To handle a greater variety of URLs (i.e. non-direct links), '
                          'install yt-dlp, \'pip install yt-dlp\' (repo: https://github.com/yt-dlp/yt-dlp).')
    return source


def load_audio(
        file: Union[str, bytes, BinaryIO],
        sr: int = None,
        verbose: Optional[bool] = True,
        only_ffmpeg: bool = False,
        mono: bool = True

):
    """
    Open an audio file and read as mono waveform then resamples as necessary.

    Parameters
    ----------
    file : str or bytes or BinaryIO
        The audio file to open, bytes of file, or URL to audio/video.
    sr : int, default whisper.model.SAMPLE_RATE
        The sample rate to resample the audio if necessary.
    verbose : bool or None, default True
        Verbosity for yd-dlp and displaying content metadata when ``file`` is a URL. If not ``None``, display metadata.
        For yd-dlp: ``None`` is "--quiet"; ``True`` is "--progress"; ``False`` is "--progress" + "--quiet".
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of yt-dlp) for URls.
    mono : bool, default True
        Whether to load as mono audio.

    Returns
    -------
    numpy.ndarray
        A array containing the audio waveform in float32.
    """
    if sr is None:
        sr = SAMPLE_RATE
    file = load_source(file, verbose=verbose, only_ffmpeg=only_ffmpeg, return_dict=False)
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI in PATH.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file if isinstance(file, str) else "pipe:",
            "-f", "s16le",
            "-ac", "1" if mono else "2",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        if isinstance(file, str):
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        else:
            cmd = cmd[:1] + ["-loglevel", "error"] + cmd[1:]
            stdin = subprocess.PIPE if isinstance(file, bytes) else file
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=stdin)
            out = out.communicate(input=file if isinstance(file, bytes) else None)[0]
            if not out:
                raise RuntimeError(f"FFmpeg failed to load audio from bytes ({len(file)}).")
    except (subprocess.CalledProcessError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"FFmpeg failed to load audio: {e.stderr.decode()}") from e

    waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    if not mono:
        return waveform.reshape(-1, 2).transpose(1, 0)
    return waveform


def resample(audio: torch.Tensor, in_sr: int, out_sr: int, **kwargs) -> torch.Tensor:
    return torchaudio.functional.resample(audio, in_sr, out_sr, **kwargs)


def voice_freq_filter(
        wf: Union[torch.Tensor, np.ndarray],
        sr: int,
        upper_freq: int = None,
        lower_freq: int = None
) -> torch.Tensor:
    if isinstance(wf, np.ndarray):
        wf = torch.from_numpy(wf)
    if upper_freq is None:
        upper_freq = 5000
    if lower_freq is None:
        lower_freq = 200
    assert upper_freq > lower_freq, f'upper_freq {upper_freq} must but greater than lower_freq {lower_freq}'
    return torchaudio.functional.highpass_biquad(
        torchaudio.functional.lowpass_biquad(wf, sr, upper_freq),
        sr,
        lower_freq
    )


def get_metadata(audiofile: Union[str, bytes, np.ndarray, torch.Tensor]) -> dict:
    if isinstance(audiofile, (np.ndarray, torch.Tensor)):
        return dict(sr=SAMPLE_RATE, duration=audiofile.shape[-1]/SAMPLE_RATE)
    import re
    cmd = ['ffmpeg', '-hide_banner', '-i']
    if isinstance(audiofile, str):
        cmd.append(audiofile)
        metadata = subprocess.run(
            cmd, capture_output=True,
        ).stderr.decode(errors="ignore")
    else:
        cmd.append('-')
        p = subprocess.Popen(
            cmd,  stderr=subprocess.PIPE, stdin=subprocess.PIPE,
        )
        try:
            p.stdin.write(audiofile)
        except BrokenPipeError:
            pass
        finally:
            metadata = p.communicate()[-1]
            if metadata is not None:
                metadata = metadata.decode(errors="ignore")
    sr = re.findall(r'\n.+Stream.+Audio.+\D+(\d+) Hz', metadata)
    duration = re.findall(r'Duration: ([\d:]+\.\d+),', metadata)
    if duration:
        h, m, s = duration[0].split(':')
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        duration = None
    return dict(sr=int(sr[0]) if sr else None, duration=duration)


def get_samplerate(audiofile: Union[str, bytes]) -> Union[int, None]:
    return get_metadata(audiofile).get('sr')


def audio_to_tensor_resample(
        audio: Union[torch.Tensor, np.ndarray, str, bytes],
        original_sample_rate: Optional[int] = None,
        target_sample_rates: Optional[Union[int, Iterable[int]]] = None,
        **kwargs
) -> torch.Tensor:
    """
    Return ``audio`` as Tensor and resample as needed.
    """
    if target_sample_rates and isinstance(target_sample_rates, int):
        target_sample_rates = [target_sample_rates]

    if isinstance(audio, (str, bytes)):
        if target_sample_rates:
            original_sample_rate = target_sample_rates[0]
        audio = load_audio(audio, sr=original_sample_rate,**kwargs)
    elif not original_sample_rate:
        original_sample_rate = SAMPLE_RATE
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.float()

    if target_sample_rates and original_sample_rate not in target_sample_rates:
        audio = resample(audio, original_sample_rate, target_sample_rates[0])

    return audio


def standardize_audio(
        audio: Union[torch.Tensor, np.ndarray, str, bytes],
        resample_sr: Tuple[Optional[int], Union[int, Tuple[int]]] = None,
) -> torch.Tensor:
    warnings.warn('This `standardize_audio()` is deprecated and will be removed. '
                  'Use `stable_whisper.audio.utils.audio_to_tensor_resample()` instead.')
    return audio_to_tensor_resample(audio, *(resample_sr or ()))
