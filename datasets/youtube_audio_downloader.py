from __future__ import unicode_literals

import os
import argparse
import json
import youtube_dl
from pytube import Playlist

def get_url_list(playlist_url):
    playlist = Playlist(playlist_url)
    print("get youtube playlistr urls.........")
    url_list = []
    for linkprefix in playlist.parse_links():
        url_list.append('https://www.youtube.com' + linkprefix)

    return url_list

def extract_audio(folder, url_list):
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': folder + '/' + folder + '-%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False
    }

    print("============={} audio extraction==============".format(folder))
    for url in url_list:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
            except Exception as e: # copyright error. 
                print(e)
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="youtube video to wav. ")

    parser.add_argument('--url_type', required=True, help='url type is playlist or video')
    parser.add_argument('--json_filename', required=True, help='json file that has url and folder info')

    args = parser.parse_args()
    
    url_type = args.url_type
    json_filename = args.json_filename

    # 1. get folder name and url_list by json paring 
    with open(json_filename) as json_file:
        json_data = json.load(json_file)['data']
        for data in json_data:
            folder = data['folder']
            if url_type == "playlist":
                playlist_url = data['url']
                url_list = get_url_list(playlist_url)
            else: # video list
                url_list = data['url_list']
             
            # 2. extract audio file from youtube video
            #    and save audio file in folder
            extract_audio(folder, url_list)