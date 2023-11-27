"""
movielib

> movielib --download_imdb_data
> movielib --extract_data <movies rep>
> movielib --make_pages <movies rep>

Note:
- when renaming a file, extract_data must be done again
"""


import sys
import os
import re
import json
import pprint
import pickle
import types
from subprocess import check_output, CalledProcessError, STDOUT
from collections import defaultdict

import requests
from PIL import Image
from imdb import Cinemagoer

import galerie


EMPTY = {
    'imdb_id': None,
    'title': None,
    'year': None,
    'director': None,
    'cast': None,
    'cover_url': None,
    'runtime': None,
    'filesize': None,
    'width': None,
    'height': None
}


def load_movie_tsv(filename):
    # https://developer.imdb.com/non-commercial-datasets/
    movies = {}
    titles = defaultdict(set)
    with open(filename, encoding='utf-8') as f:
        f.readline()
        for line in f:
            tconst, _, primary_title, original_title, _, year, _, runtime_minutes, genres = line.split('\t')
            movies[tconst] = (primary_title, original_title, year, runtime_minutes, genres)

            primary_title = primary_title.replace(':', '')
            original_title = original_title.replace(':', '')
            original_title = original_title.replace('  ', ' ')
            primary_title = primary_title.replace('.', '')
            original_title = original_title.replace('.', '')
            original_title = original_title.replace('  ', ' ')

            primary_title = primary_title.lower()
            original_title = original_title.lower()

            titles[primary_title].add(tconst)
            titles[original_title].add(tconst)
            titles[f'{primary_title}-{year}'].add(tconst)
            titles[f'{original_title}-{year}'].add(tconst)
    return movies, dict(titles)


def search_movie_tsv(titles, title, year=None):
    key = title if (year is None) else f'{title}-{year}'
    return titles.get(key.lower(), None)


def get_dimensions(filename):
    # ffmpeg must be in path
    command = 'ffprobe -v error -select_streams v:0 -show_entries stream=height,width -of csv=s=x:p=0 "' + filename + '"'

    try:
        output = check_output(command, stderr=STDOUT).decode()
        width, height = [int(_) for _ in output.strip().split('x')]
        return width, height
    except CalledProcessError as e:
        output = e.output.decode()
        return None, None


def movie_gen(rep):
    """
    Find recursively all movies in rep.
    """
    for dirpath, _, filenames in os.walk(rep):
        for filename in filenames:
            barename, ext = os.path.splitext(filename)
            if ext in ('.mp4', '.avi', '.mkv'):
                yield dirpath, filename, barename


def create_missing_records(rep, tsvfile):
    """
    Find recursively all movies in rep. Create json file (with same name as
    movie) if absent. Fill record with imdb data if imdb id can be found, plus
    data relative to file.
    """
    ia = Cinemagoer()
    movie_number = 0
    new_movie_number = 0
    movie_found = 0
    print('Loading...')
    movies, titles = load_movie_tsv(tsvfile)
    print('Loaded')

    for dirpath, filename, barename in movie_gen(rep):
        movie_number += 1

        jsonname = os.path.join(dirpath, barename + '.json')
        if os.path.isfile(jsonname):
            continue
        new_movie_number += 1

        if re.search(r'\(\d\d\d\d\)\s*$', barename):
            match = re.match(r'\s*(.*)\s*\((\d\d\d\d)\)\s*$', barename)
            name = match.group(1).strip()
            year = match.group(2)
            imdb_id = search_movie_tsv(titles, name, year)
        else:
            name = barename.strip()
            imdb_id = search_movie_tsv(titles, name)

        if imdb_id is None:
            print(name, 'not found')
            continue
        elif len(imdb_id) > 1:
            print(name, 'ambiguous', imdb_id)
            continue
        else:
            # movie found
            imdb_id = list(imdb_id)[0]

        movie_found += 1
        record = EMPTY.copy()

        # set size
        record['filesize'] = os.path.getsize(os.path.join(dirpath, filename))

        # set dimensions
        width, height = get_dimensions(os.path.join(dirpath, filename))
        record['width'] = width
        record['height'] = height

        # set imdb information
        movie = ia.get_movie(imdb_id[2:])
        record['imdb_id'] = movie.movieID
        if movie.get('countries')[0] == 'France':
            record['title'] = movie.get('original title')
        else:
            record['title'] = movie.get('title')
        record['year'] = movie.get('year')
        record['runtime'] = movie.get('runtimes')[0]
        record['director'] = [_.get('name') for _ in movie.get('director')]
        record['cast'] = [_.get('name') for _ in movie.get('cast')[:5]]
        record['cover_url'] = movie.get('cover url')

        # save record
        jsonname = os.path.join(dirpath, barename + '.json')
        with open(jsonname, 'w') as f:
            json.dump(record, f, indent=4)

        # load movie cover
        imgname = os.path.join(dirpath, barename + '.jpg')
        if os.path.isfile(imgname) is False:
            imgdata = requests.get(record['cover_url']).content
            with open(imgname, 'wb') as handler:
                handler.write(imgdata)

        pprint.pprint(record)
        print()

    print('movie_number', movie_number)
    print('new_movie_number', new_movie_number)
    print('movie_found', movie_found)


def year_title(record):
    return f"{record['year']}: {record['title']}"


def make_index(rep):
    movies = []
    directors = defaultdict(list)
    for index, (dirpath, filename, barename) in enumerate(movie_gen(rep)):
        jsonname = os.path.join(dirpath, barename + '.json')
        if os.path.isfile(jsonname) is False:
            print(jsonname, 'not found')
        else:
            with open(jsonname) as f:
                record = json.loads(f.read())
                record['index'] = index
                record['dirpath'] = dirpath
                record['filename'] = filename
                record['barename'] = barename
            # pprint.pprint(record)
            movies.append(record)
            director = record['director']
            for _ in director:
                directors[_].append(year_title(record))

    with open(os.path.join(rep, 'movies.pickle'), 'wb') as f:
        pickle.dump(movies, f)

    for director, lst in directors.items():
        directors[director] = sorted(lst)
    with open(os.path.join(rep, 'directors.pickle'), 'wb') as f:
        pickle.dump(directors, f)


START = '''\
<html>

<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <title>%s</title>
    <link rel="icon" href="Movies-icon.png" />
<style type="text/css">
    p { margin-top:0px; margin-bottom:10px; }
    span { display:inline-table; width:160px }
 </style>\
</head>

<body>
<div style="width: 95%%; margin-left: auto; margin-right: auto">\
'''
END = '</div>\n</body>\n</html>'
VIDPOSTCAPTION = '''\
<span>
<a href="%s" rel="video"><img src=%s width="%d" height="%d" title="%s"/></a>
<p>%s</p>
</span>
'''
VIDPOSTCAPTION2 = '''\
<span>
<a href="%s" rel="video"><img src=%s width="%d" title="%s"/></a>
<p>%s</p>
</span>
'''
VIDPOSTCAPTION3 = '''\
<span>
<img src="%s" width="%d" alt="%s cover" usemap="#workmap%d" title="%s">
<p>%s</p>
</span>
%s
'''
IMGMAP = '''\
<map name="workmap%d">
  <area shape="rect" coords="%s" title="%s">
  <area shape="rect" coords="%s" href="%s" title="Description">
  <area shape="rect" coords="%s" href="%s" title="Play">
</map>
'''


def urlencode(url):
    url = url.replace('\\', '/')
    url = url.replace(' ', '%20')
    return url


def make_movie_element(movie_num, movie_title, movie_name, thumb_name, html_name, thumb_width, descr):
    width, height = Image.open(thumb_name).size
    imgmap = IMGMAP % (
        movie_num,
        '%d, %d, %d, %d' % (0, 0, width, height // 3),
        descr,
        '%d, %d, %d, %d' % (0, height // 3, width, 2 * height // 3),
        urlencode(html_name[9:]),
        '%d, %d, %d, %d' % (0, 2 * height // 3, width, height),
        urlencode(movie_name[9:])
    )
    movie_element = VIDPOSTCAPTION3 % (
        urlencode(thumb_name[9:]),
        thumb_width,
        movie_title,
        movie_num,
        'foofoofoo',
        movie_title,
        imgmap
    )
    return movie_element


def make_main_page(rep):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)

    # TODO trier dans l'ordre d'insertion (ou un autre ordre pertinent)
    pass

    with open(os.path.join(rep, 'movies.htm'), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for movie_num, record in enumerate(movies):
            movie_name = os.path.join(record['dirpath'], record['filename'])
            image_basename = record['barename'] + '.jpg'
            image_name = os.path.join(record['dirpath'], image_basename)
            thumb_basename = galerie.thumbname(image_basename, 'film')
            thumb_name = os.path.join(rep, '.gallery', '.thumbnails', thumb_basename)
            html_basename = record['barename'] + '.htm'
            html_name = os.path.join(record['dirpath'], html_basename)

            width, height = Image.open(image_name).size
            thumbsize = galerie.size_thumbnail(width, height, maxdim=300)

            args = types.SimpleNamespace()
            args.forcethumb = True
            galerie.make_thumbnail_image(args, image_name, thumb_name, thumbsize)

            descr = f"{record['title']}, {record['year']}, {', '.join(record['director'])}"
            print(make_movie_element(movie_num, record['title'], movie_name, thumb_name, html_name, 160, descr), file=f)
        print(END, file=f)


def make_li_list(liste):
    return '\n'.join([f'<li>{_}</li>' for _ in liste])


OTHER_DIRECTOR_MOVIES = '''\
<h3>Autres films de %s dans la collection</h3>
 <ul>
    %s
</ul>
'''

def make_movie_pages(rep):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)
    with open(os.path.join(rep, 'directors.pickle'), 'rb') as f:
        directors = pickle.load(f)
    with open('template.htm', encoding='utf-8') as f:
        template = f.read()

    for record in movies:
        image_basename = record['barename'] + '.jpg'
        html_basename = record['barename'] + '.htm'
        html_name = os.path.join(record['dirpath'], html_basename)

        first_director = record['director'][0]
        other_directors = record['director'][1:]

        html = template[:]
        html = html.replace('{{cover}}', image_basename)
        html = html.replace('{{title}}', record['title'])
        html = html.replace('{{year}}', str(record['year']))
        if other_directors:
            html = html.replace('{{director}}', make_li_list([first_director, ', '.join(other_directors)]))
        else:
            html = html.replace('{{director}}', make_li_list([first_director]))
        html = html.replace('{{runtime}}', str(record['runtime']))
        html = html.replace('{{width}}', str(record['width']))
        html = html.replace('{{height}}', str(record['height']))
        html = html.replace('{{filesize}}', str(record['filesize']))
        html = html.replace('{{cast}}', '\n'.join([f'<li>{_}</li>' for _ in record['cast']]))

        othermovies1 = [_ for _ in directors[first_director] if year_title(record) != _]
        othermovies2 = set()
        for director in other_directors:
            othermovies2.update([_ for _ in directors[director] if year_title(record) != _])
        othermovies1 = ['Aucun'] if not othermovies1 else sorted(othermovies1)
        othermovies2 = ['Aucun'] if not othermovies2 else sorted(othermovies2)

        othermovieshtml = [OTHER_DIRECTOR_MOVIES % (first_director, make_li_list(othermovies1))]
        if other_directors:
            if len(other_directors) > 1:
                other_directors = other_directors[:1] + ['etc.']
            othermovieshtml.append(OTHER_DIRECTOR_MOVIES % (', '.join(other_directors), make_li_list(othermovies2)))

        html = html.replace('{{other_movies}}', '\n'.join(othermovieshtml))

        with open(html_name, 'wt', encoding='utf-8') as f:
            print(html, file=f)


def clean(rep):
    for dirpath, _, barename in movie_gen(rep):
        jsonname = os.path.join(dirpath, barename + '.json')
        imgname = os.path.join(dirpath, barename + '.jpg')
        if os.path.isfile(jsonname) and not os.path.isfile(imgname):
            print(jsonname)
            os.remove(jsonname)


def test1():
    # create an instance of the Cinemagoer class
    ia = Cinemagoer()

    # get a movie
    # movie = ia.get_movie('0133093')
    movies = ia.search_movie('Crouching Tiger')
    movie = movies[0]
    print(movie, movie.movieID)
    print(repr(movie))
    print(movie.infoset2keys)

    movie = ia.get_movie(movie.movieID)
    print(repr(movie))
    print(movie.infoset2keys)

    print(movie.get('year'))
    director = movie.get('director')[0]
    print(director)
    print(director.get('name'))
    print(movie.get('plot'))
    for actor in movie.get('cast'):
        print(actor.get('name'))

    print(movie['cover url'])
    return

    # print the names of the directors of the movie
    print('Directors:')
    for director in movie['directors']:
        print(director['name'])

    # print the genres of the movie
    print('Genres:')
    for genre in movie['genres']:
        print(genre)

    # search for a person name
    people = ia.search_person('Mel Gibson')
    for person in people:
        print(person.personID, person['name'])


def test2():
    # create an instance of the Cinemagoer class
    ia = Cinemagoer()
    # omdb.set_default('apikey', '7970510d')
    print(ia.get_movie_infoset())
    movie = ia.get_movie('0133093')
    print(movie.infoset2keys)
    print(movie.get('cover url'))


def main():
    if len(sys.argv) < 2:
        print('HELP')
        sys.exit(-1)
    elif sys.argv [1] == '--download_imdb_data':
        # TODO
        pass
    elif sys.argv [1] == '--extract_data' and len(sys.argv) == 3:
        rep = sys.argv[2]
        create_missing_records(rep, 'movie.tsv')
        make_index(rep)
    elif sys.argv [1] == '--make_pages' and len(sys.argv) == 3:
        rep = sys.argv[2]
        make_main_page(rep)
        make_movie_pages(rep)
    else:
        # TODO
        print('HELP')


main()
