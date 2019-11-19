import requests
import os
import json
import argparse
import numpy as np
from glob import glob
from functools import partial

from utils import parallel_run, remove_file, backup_file, write_json
from audio import load_audio, save_audio, resample_audio, get_duration

def text_recognition(path, config):
    root, ext = os.path.splitext(path)
    txt_path = root + ".txt"

    # has text script
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            out = json.loads(open(txt_path).read())
            return out

    # hasn't text script
    # make text script using STT

    # Waston API credential
    username = 'de7ac1ac-3d80-410c-bb26-972ab9737a68'
    password = 'TEL5C5Ve7Lg3'

    headers = {'Content-Type': 'audio/wav'}
    baseUrl = 'https://stream.aibril-watson.kr/speech-to-text/api/v1/recognize?model='
    model = "ko-KR_BroadbandModel"
    url = baseUrl + model 
    
    out = {}
    error_count = 0

    tmp_path = os.path.splitext(path)[0] + ".tmp.wav"

    while True:
        try:
            # audio processing 
            content = load_audio(
                    path, pre_silence_length=config.pre_silence_length,
                    post_silence_length=config.post_silence_length)

            max_duration = config.max_duration - \
                    config.pre_silence_length - config.post_silence_length
            audio_duration = get_duration(content)

            if audio_duration >= max_duration:
                print(" [!] Skip {} because of duration: {} > {}". \
                        format(path, audio_duration, max_duration))
                return {}

            content = resample_audio(content, config.sample_rate)
            save_audio(content, tmp_path, config.sample_rate)

            # STT
            with open(tmp_path, 'rb') as f:
                response = requests.post(url, auth=(username,password), data=f, headers=headers).json()
            
            if len(response) > 0:
                alternatives = response['results']
                results = [alternative['alternatives'][0]['transcript'] for alternative in alternatives]
                
                #assert len(results) == 1, "More than 1 results: {}".format(results)

                out = { path: "" if len(results) == 0 else results[0]}
                if not len(results) == 0:
                    print(path, results[0])
                break
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

    with open(txt_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out

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

    for tmp_path in glob(os.path.join(audio_dir, "*.tmp.*")):
        remove_file(tmp_path)

    paths = glob(config.audio_pattern)
    paths.sort()
    results = text_recognition_batch(paths, config)

    base_dir = os.path.dirname(audio_dir)
    recognition_path = \
            os.path.join(base_dir, config.recognition_filename)

    if os.path.exists(recognition_path):
        backup_file(recognition_path)

    write_json(recognition_path, results)
