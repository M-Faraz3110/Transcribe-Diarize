import os
import subprocess

import librosa
from matplotlib import pyplot as plt
from pydub import AudioSegment
from pydub.utils import mediainfo_json
from scipy.io import wavfile

SOX_PATH = ".\\sox\\sox.exe"


def normalize_audio(audio_file, bits=16, rate=48000):
    '''
    Automatically invoke the gain effect to guard against clipping and to normalise the audio.

    Default File settings set to 16 bits and 48000 sample rate
    '''

    # set bits and rate to None and uncomment below if needed
    # if bits is None:
    #     bits = get_sample_format(audio_file)
    # if rate is None:
    #     rate, _ = wavfile.read(audio_file)

    normalized_audio_file = os.path.splitext(audio_file)[0] + '_normalized.wav'
    command = '"{sox}" --norm "{wav_in}" -b {bits} "{wav_out}" rate {rate} dither -s'.format(
        sox=SOX_PATH, wav_in=audio_file, wav_out=normalized_audio_file, bits=bits, rate=rate
    )

    print("normalize command: ", command)
    print("running script: ", subprocess.run(command))

    return normalized_audio_file


def denoise_sox(audio_file, silence_audio_file=None):
    audio_file = normalize_audio(audio_file)

    # get a silence audio segment from the beginning of audio if none is given, as it will be needed for noise profiling
    if silence_audio_file is None:
        silence_audio_file = os.path.splitext(audio_file)[0] + '_silence_segment.wav'

        command = '"{sox}" "{wav_in}" "{wav_out}" trim 0 5.00'.format(
            sox=SOX_PATH, wav_in=audio_file, wav_out=silence_audio_file
        )

        print("trimming silence segment command: ", command)
        print("running script: ", subprocess.run(command))

    # generate a noise profile for the silence audio segment
    silence_audio_segment = AudioSegment.from_wav(file=silence_audio_file)

    segment_length_sec = len(silence_audio_segment) / 1000.0

    silence_profile_file = os.path.splitext(audio_file)[0]+'.prof'
    command = '"{sox}" "{wav_in}" -n trim 0 {silence_len} noiseprof "{prof_out}"'.format(sox=SOX_PATH,
                                                                                     wav_in=silence_audio_file,
                                                                                     silence_len=segment_length_sec,
                                                                                     prof_out=silence_profile_file)
    print("command: ", command)
    print("running script: ", subprocess.run(command))

    # reduce noise according the noise profile
    reduced_noise_file = os.path.splitext(audio_file)[0] + '_cleaned.wav'

    command = '"{sox}" "{wav_in}" "{wav_out}" noisered "{prof_in}" 0.3'.format(sox=SOX_PATH,
                                                                               wav_in=audio_file,
                                                                               wav_out=reduced_noise_file,
                                                                               prof_in=silence_profile_file)

    print("command: ", command)
    print("running script: ", subprocess.call(command))

    return reduced_noise_file, silence_audio_file, silence_profile_file


def filter_out_silence(audio_file):
    '''
    trims silence from beginning and end and pads some silence in the beginning
    '''

    # sox in.wav out.wav
    filtered_out_silence_file = os.path.splitext(audio_file)[0] + '_filtered_out_silence.wav'
    command = '"{sox}" "{wave_in}" "{wave_out}" silence 1 0.1 0.1% reverse silence 1 0.1 0.1% reverse pad 0.75 0'.format(
        sox=SOX_PATH, wave_in=audio_file, wave_out=filtered_out_silence_file
    )

    print("command: ", command)
    print("running script: ", subprocess.call(command))

    return filtered_out_silence_file


def apply_high_pass_filter(audio_file):
    high_passed_file = os.path.splitext(audio_file)[0] + '_highpass_filter.wav'
    command = '"{sox}" "{wave_in}" "{wave_out}" sinc 3k-500'.format(
        sox=SOX_PATH, wave_in=audio_file, wave_out=high_passed_file
    )

    print("command: ", command)
    print("running script: ", subprocess.call(command))

    return high_passed_file


def get_sample_format(audio_file):
    info = mediainfo_json(audio_file)
    audio_streams = [x for x in info['streams'] if x['codec_type'] == 'audio']
    sample_rate = audio_streams[0].get('sample_fmt')

    print("INFO: Sample rate of passed audio is", sample_rate)

    return sample_rate


def convert_audio_to_spectogram(filename):
    """
    convert_audio_to_spectogram -- using librosa to simply plot a spectogram

    Arguments:
    filename -- filepath to the file that you want to see the waveplot for

    Returns -- None
    """

    # sr == sampling rate
    x, sr = librosa.load(filename, sr=44100)

    # stft is short time fourier transform
    X = librosa.stft(x)

    # convert the slices to amplitude
    Xdb = librosa.amplitude_to_db(abs(X))

    # ... and plot, magic!
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


def preprocess(audio_file):
    normalized_audio = normalize_audio(audio_file)
    preprocessed_audio, silence_segment, silence_profile = denoise_sox(normalized_audio)

    # clean up
    os.remove(normalized_audio)
    os.remove(silence_segment)
    os.remove(silence_profile)

    return preprocessed_audio

