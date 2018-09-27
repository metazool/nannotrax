import logging
import argparse

import requests
from bs4 import BeautifulSoup, Comment

logging.basicConfig(level=logging.INFO)

MIKROTAX = 'http://www.mikrotax.org'
NANNOTAX = "{}/{}".format(MIKROTAX, 'Nannotax3/index.php')


def scrape():
    response = requests.get(NANNOTAX, params={'taxon': 'Thiersteinia', 'module': 'Mesozoic'})
    with open('test.html', 'wb') as out:
        out.write(response.content)


def extract_data(html):
    """Extract taxonomic hierarchy and summary from individual page html"""
    soup = BeautifulSoup(html, features="lxml")
    logging.info(soup.find('title').text)
    hierarchy = hierarchy_summary(soup)
    samples = daughter_taxa(soup)


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
    #scrape()
    extract_data(open('test/fixtures/mesozoic.html').read())
