import requests
import os
import json
import argparse
import numpy as np
from glob import glob
from functools import partial
from pydub import AudioSegment

from utils import parallel_run, remove_file, backup_file, write_json
from audio import load_audio, save_audio, resample_audio, get_duration

def text_recognition(path, config):
    # hasn't text script
    # make text script using STT

    # Waston API credential

    username = 'de7ac1ac-3d80-410c-bb26-972ab9737a68'
    password = 'TEL5C5Ve7Lg3'
    
    headers = {'Content-Type': 'audio/wav'}
    baseUrl = 'https://stream.aibril-watson.kr/speech-to-text/api/v1/recognize?model='
    model = "ko-KR_BroadbandModel"
    url = baseUrl + model 

    url += '&timestamps=true'
    
    out = {}
    error_count = 0

    while True:
        try:

            # STT using full audio
            with open(path, 'rb') as f:
                response = requests.post(url, auth=(username,password), data=f, headers=headers).json()

            # parsing duration, transcript
            if len(response) > 0:
                alternatives = response['results']
                results = []
                for alternative in alternatives:
                    start = alternative['alternatives'][0]['timestamps'][0][1]
                    end = alternative['alternatives'][0]['timestamps'][-1][-1]
                    transcript = alternative['alternatives'][0]['transcript']
                    print(transcript) 
                    print('duration : {} ~ {}'.format(start, end))
                    results.append([transcript, start, end])

            # Split audio and save splited audio
            txt_outs = {}
            for idx, (transcript, start, end) in enumerate(results):
                start *= 1000 # works in milliseconds
                end *= 1000 # works in milliseconds
                split_audio = AudioSegment.from_wav(path)
                split_audio = split_audio[start:end]

                output_path = "{}.{:04d}.{}".format(
                        path.replace('.wav',''), idx, 'wav')

                split_audio.export(output_path, format='wav')

                out = { output_path: transcript}

                txt_path = output_path.replace('.wav', '.txt')
                with open(txt_path, 'w') as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                
                txt_outs.update(out)
            break
        
        
        except Exception as err:
            raise Exception("OS error: {0}".format(err))

            error_count += 1
            print("Skip warning for {} for {} times". \
                    format(path, error_count))

            if error_count > 5:
                break
            else:
                continue

    return txt_outs

def text_recognition_batch(paths, config):
    paths.sort()

    results = {}
    items = parallel_run(
            partial(text_recognition, config=config), paths,
            desc="text_recognition_batch", parallel=False)
    for item in items:
        results.update(item)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_pattern', required=True)
    parser.add_argument('--recognition_filename', default="recognition.json")
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--pre_silence_length', default=1, type=int)
    parser.add_argument('--post_silence_length', default=1, type=int)
    parser.add_argument('--max_duration', default=60, type=int)
    config, unparsed = parser.parse_known_args()

    audio_dir = os.path.dirname(config.audio_pattern)

    # get audio files that have config.audio_pattern type 
    paths = glob(config.audio_pattern)
    paths.sort()

    # STT and split audio using duration by stt results 
    results = text_recognition_batch(paths, config)

    # write json file (audio, text pair)
    base_dir = os.path.dirname(audio_dir)
    recognition_path = \
            os.path.join(base_dir, config.recognition_filename)

    if os.path.exists(recognition_path):
        backup_file(recognition_path)

    write_json(recognition_path, results)
