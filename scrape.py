import logging
import argparse
import json
import os

import requests
from bs4 import BeautifulSoup, Comment

logging.basicConfig(level=logging.INFO)

MIKROTAX = 'http://www.mikrotax.org'
NANNOTAX = "{}/{}".format(MIKROTAX, 'Nannotax3/index.php')

SEEN_PAGES = {}

def scrape(module='Mesozoic'):
    start_here = request_page(module=module, taxon=module)
    traverse(start_here, module=module, taxon=module)


def save_data(data, module, taxon):
    directory = os.path.join(os.getcwd(), 'data', module)
    if not os.path.isdir(directory): os.makedirs(directory)
    """Write the extracted data to a file, guess there will be redundancy"""
    with open(os.path.join(directory, f'{taxon}.json'), 'w') as out:
        out.write(json.dumps(data))


def traverse(page, module=None, taxon=None):
    """Recursively try to follow all the taxonomic links from a given page"""
    data = extract_data(page)
    if not data:
        return None

    save_data(data, module, taxon)
    for sample in data['samples']:
        if sample['taxon'] not in SEEN_PAGES:
            new_page = request_page(module=module, taxon=sample['taxon'])
            SEEN_PAGES[sample['taxon']] = 1
            traverse(new_page, module=module, taxon=sample['taxon'])

    logging.info('seen everything')


def request_page(module=None, taxon=None):
    logging.info(taxon)
    response = requests.get(NANNOTAX, params={'taxon': taxon, 'module': module})
    response.raise_for_status()
    return response.content


def extract_data(html):
    """Extract taxonomic hierarchy and summary from individual page html"""
    soup = BeautifulSoup(html, features="lxml")
    logging.info(soup.find('title').text)
    hierarchy = hierarchy_summary(soup)
    samples = daughter_taxa(soup)
    if not samples:
        return None

    return {'hierarchy': hierarchy,
            'samples': samples }


def hierarchy_summary(soup):
    """Summarise the hierarchy usingthe classification text"""
    classification_next = ' Links to navigate up taxonomy and to siblings get written in here'

    hierarchy = []
    for comment in soup.findAll(text=lambda text:isinstance(text, Comment)):
        if comment == classification_next:
            next_strings = comment.find_all_next(string=True)
            in_hierarchy = 0
            for s in next_strings:
                if s == 'Classification: ':
                    in_hierarchy = 1
                    continue
                if s == 'Sister taxa: ': break

                if in_hierarchy:
                    taxon = s.strip().replace('->', '').replace(' ','')
                    if taxon: hierarchy.append(taxon)

    return hierarchy


def daughter_taxa(soup):
    """Extract thumbnail images and taxonomic names from the table"""

    taxa = []

    table = soup.find('table')
    rows = table.find_all('tr')
    if not rows:
        return None

    for row in rows[1:-1]:  # skip the first row
        thumbs = []
        taxon = ''
        columns = row.find_all('td')
        for col in columns:
            thumbnail = col.find('img')
            if thumbnail:
                thumbs.append("{}{}".format(MIKROTAX, thumbnail['src']))
            else:
                taxon_m = col.find('span', class_='taxon_m')
                if taxon_m: taxon = taxon_m.text

                smallcaps_m = col.find('span', class_='smallcaps_m')
                if smallcaps_m: taxon = smallcaps_m.text

        if taxon: taxa.append({'taxon': taxon, 'thumbs': thumbs})

    return taxa


if __name__ == '__main__':

    scrape(module='Coccolithophores')
    #extract_data(open('test/fixtures/mesozoic.html').read())
