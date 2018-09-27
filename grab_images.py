import json
import logging
import os
import urllib

import requests

logging.basicConfig(level=logging.INFO)

SEEN_IMAGES = {}

def grab_images(directory='./data'):
    files = os.listdir(directory)
    print(files)
    for filename in files:
        with open(os.path.join(directory, filename)) as json_data:
            data = json.load(json_data)
            for sample in data['samples']:
                for thumb in sample['thumbs']:
                    grab_image(thumb)


def grab_image(url, directory='images'):
    if url in SEEN_IMAGES:
        return None

    file_path = urllib.parse.urlparse(url)
    file_path = file_path.path.split('/')
    filename = os.path.join(os.getcwd(), directory, file_path[-1])

    if os.path.isfile(filename):
        logging.info("got {}".format(url))
        return None

    logging.info("grab {}".format(url))
    content = requests.get(url).content

    try:
        with open(filename, 'wb') as out:
            out.write(content)
    except Exception as e: # yes i know, broad except
        logging.info(e)

    SEEN_IMAGES[url] = 1


if __name__ == '__main__':
    grab_images()
