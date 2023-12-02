"""
> movielib --extract_movie_tsv
> movielib --extract_data <movies rep>
> movielib --make_pages <movies rep>
> movielib --update <movies rep>

Note:
- when renaming a movie related file (directory, mp4, etc), extract_data must be
  done again.
"""


import os
import re
import json
import pprint
import pickle
import types
import shutil
import gzip
import argparse
from datetime import datetime
from subprocess import check_output, CalledProcessError, STDOUT
from collections import defaultdict

import requests
from PIL import Image
from imdb import Cinemagoer

import galerie


MOVIE_TSV_FN = 'movie.tsv'
MOVIES_VRAC = 'movies-vrac.htm'
MOVIES_YEAR = 'movies-year.htm'
MOVIES_ALPHA = 'movies-alpha.htm'
MOVIES_DIRECTOR = 'movies-director.htm'


EMPTY = {
    'imdb_id': None,
    'title': None,
    'year': None,
    'director': [],
    'cast': [],
    'runtime': None,
    'filesize': None,
    'width': None,
    'height': None
}


def extract_movie_tsv(movie_tsv_filename):
    """
    Assume that the file title.basics.tsv.gz has been downloaded from
    https://developer.imdb.com/non-commercial-datasets/ and that it contains
    the file data.tsv.
    """
    with gzip.open('title.basics.tsv.gz', 'rb') as f_in:
        with open('data.tsv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open('data.tsv', encoding='utf-8') as f:
        with open(movie_tsv_filename, 'wt', encoding='utf-8') as g:
            print(f.readline(), file=g)
            for line in f:
                if '\tmovie\t' in line:
                    print(line, end='', file=g)

    os.remove('data.tsv')


def load_movie_tsv(movie_tsv_filename):
    movies = {}
    titles = defaultdict(set)
    with open(movie_tsv_filename, encoding='utf-8') as f:
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
    command = f'ffprobe -v error -select_streams v:0 -show_entries stream=height,width -of csv=s=x:p=0 "{filename}"'

    try:
        output = check_output(command, stderr=STDOUT).decode()
        width, height = [int(_) for _ in output.strip().split('x')]
        return width, height
    except CalledProcessError as e:
        output = e.output.decode()
        return None, None


def get_duration(filename):
    # ffmpeg must be in path
    command = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filename}"'
    try:
        output = check_output(command, stderr=STDOUT).decode()
        output = output.splitlines()[0]
        return int(float(output) / 60.0)
    except CalledProcessError as e:
        output = e.output.decode()
        return 'undefined'


def extract_image_from_movie(filename, imagename, size, delay):
    # ffmpeg must be in path
    sizearg = '%dx%d' % (size, size)
    command = 'ffmpeg -y -v error -itsoffset -%d -i "%s" -vcodec mjpeg -vframes 1 -an -f rawvideo -s %s "%s"'
    command = command % (delay, filename, sizearg, imagename)
    result = os.system(command)


def movie_gen(rep):
    """
    Find recursively all movies in rep.
    """
    for dirpath, _, filenames in os.walk(rep):
        for filename in filenames:
            barename, ext = os.path.splitext(filename)
            if ext in ('.mp4', '.avi', '.mkv'):
                yield dirpath, filename, barename


def get_title(name, moviestsv, imdbmovie):
    """
    name: title of movie as extracted from file name
    moviestsv: dict[tt-id] --> data extracted from tsv file downloaded from imdb
    imdbmovie: movie object retrieved from imdb
    """
    tt_id = 'tt' + imdbmovie.movieID
    tsvrecord = moviestsv[tt_id]
    if name == tsvrecord[0]:    # primaryTitle
        return name
    elif name == tsvrecord[1]:  # originalTitle
        return name
    elif imdbmovie.get('countries')[0] == 'France':
        return imdbmovie.get('original title')
    else:
        return  imdbmovie.get('title')


def create_minimal_record(dirpath, filename, name, year, movies, ia):
    record = EMPTY.copy()
    fullname = os.path.join(dirpath, filename)

    # set title and year found in filename
    record['title'] = name
    record['year'] = year

    # set size
    record['filesize'] = os.path.getsize(fullname)

    # set dimensions
    width, height = get_dimensions(fullname)
    record['width'] = width
    record['height'] = height

    # set duration
    record['runtime'] = get_duration(fullname)
    return record


def create_record(dirpath, filename, name, movies, ia, imdb_id):
    record = EMPTY.copy()
    fullname = os.path.join(dirpath, filename)

    # set size
    record['filesize'] = os.path.getsize(fullname)

    # set dimensions
    width, height = get_dimensions(fullname)
    record['width'] = width
    record['height'] = height

    # set imdb information
    movie = ia.get_movie(imdb_id[2:])
    record['imdb_id'] = movie.movieID
    record['title'] = get_title(name, movies, movie)
    record['year'] = movie.get('year')
    record['runtime'] = movie.get('runtimes')[0]
    record['director'] = [_.get('name') for _ in movie.get('director')]
    record['cast'] = [_.get('name') for _ in movie.get('cast')[:5]]

    # load default movie cover
    imgname = os.path.splitext(fullname)[0] + '.jpg'
    if os.path.isfile(imgname) is False:
        imgdata = requests.get(movie.get('cover url'), timeout=10).content
        with open(imgname, 'wb') as handler:
            handler.write(imgdata)

    return record


def create_missing_records(rep, tsvfile, forcejson=False):
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
        if os.path.isfile(jsonname) and forcejson is False:
            continue
        new_movie_number += 1

        if re.search(r'\(\d\d\d\d\)\s*$', barename):
            match = re.match(r'\s*(.*)\s*\((\d\d\d\d)\)\s*$', barename)
            name = match.group(1).strip()
            year = int(match.group(2))
            imdb_id = search_movie_tsv(titles, name, year)
        else:
            name = barename.strip()
            year = 9999
            imdb_id = search_movie_tsv(titles, name)

        if imdb_id and len(imdb_id) > 1:
            print(name, 'ambiguous', imdb_id)
            continue
        elif imdb_id is None:
            print(name, 'not found in imdb')
            record = create_minimal_record(dirpath, filename, name, year, movies, ia)
        else:
            # movie found
            imdb_id = list(imdb_id)[0]
            movie_found += 1
            record = create_record(dirpath, filename, name, movies, ia, imdb_id)

        # save record
        jsonname = os.path.join(dirpath, barename + '.json')
        with open(jsonname, 'w') as f:
            json.dump(record, f, indent=4)

        # create default image if absent
        moviename = os.path.join(dirpath, filename)
        imagename = os.path.join(dirpath, barename + '.jpg')
        if os.path.isfile(imagename) is False:
            extract_image_from_movie(moviename, imagename, size=300, delay=60)

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
    for movienum, (dirpath, filename, barename) in enumerate(movie_gen(rep)):
        jsonname = os.path.join(dirpath, barename + '.json')
        if os.path.isfile(jsonname) is False:
            print(jsonname, 'not found')
        else:
            with open(jsonname) as f:
                record = json.loads(f.read())
                record['movienum'] = movienum
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
body {
  font: normal 14px Verdana, Arial, sans-serif;
}
p {
  margin-top:0px;
  margin-bottom:10px;
}
span {
  display:inline-table;
  width:160px
}
h2 {
  padding: 4px;
  line-height: 12px;
  border-top: 2px solid grey;
  border-bottom: 2px solid grey;
}
</style>\
<base target="_parent"></base>
</head>

<body>
<div style="width: 95%%; margin-left: auto; margin-right: auto">\
'''
END = '</div>\n</body>\n</html>'
VIDPOSTCAPTION = '''\
<span>
<img src="%s" width="%d" alt="%s cover" usemap="#workmap%d">
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


def time_ordered(fn1, fn2):
    """
    Check if two files are time ordered.
    """
    return os.path.getmtime(fn1) < os.path.getmtime(fn2)


def make_movie_element(rep, record, thumb_width, forcethumb=False):
    movie_name = os.path.join(record['dirpath'], record['filename'])
    image_basename = record['barename'] + '.jpg'
    image_name = os.path.join(record['dirpath'], image_basename)
    thumb_basename = galerie.thumbname(image_basename, 'film')
    thumb_name = os.path.join(rep, '.gallery', '.thumbnails', thumb_basename)
    html_basename = record['barename'] + '.htm'
    html_name = os.path.join(record['dirpath'], html_basename)

    if forcethumb or os.path.isfile(thumb_name) is False or time_ordered(image_name, thumb_name) is False:
        args = types.SimpleNamespace()
        args.forcethumb = True
        if os.path.isfile(image_name):
            width, height = Image.open(image_name).size
            thumbsize = galerie.size_thumbnail(width, height, maxdim=300)
            galerie.make_thumbnail_image(args, image_name, thumb_name, thumbsize)
        else:
            print('Warning: no image for', movie_name)

    descr = f"{record['title']}, {record['year']}, {', '.join(record['director'])}"

    if os.path.isfile(thumb_name):
        width, height = Image.open(thumb_name).size
        height = int(round(160.0 * height / width))
        width = 160
        imgmap = IMGMAP % (
            record['movienum'],
            '%d, %d, %d, %d' % (0, 0, width - 1, int(round(height / 3))),
            descr,
            '%d, %d, %d, %d' % (0, int(round(height / 3)), width - 1, int(round(2 * height / 3))),
            urlencode(html_name[9:]),
            '%d, %d, %d, %d' % (0, int(round(2 * height / 3)), width - 1, height - 1),
            urlencode(movie_name[9:])
        )
    else:
        imgmap = ''

    movie_element = VIDPOSTCAPTION % (
        urlencode(thumb_name[9:]),
        thumb_width,
        record['title'],
        record['movienum'],
        record['title'],
        imgmap
    )
    return movie_element


def make_vrac_page(rep, forcethumb):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)

    with open(os.path.join(rep, MOVIES_VRAC), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for record in movies:
            print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_year_page(rep, forcethumb):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)

    movies_by_year = defaultdict(list)
    for record in movies:
        movies_by_year[record['year']].append(record)

    with open(os.path.join(rep, MOVIES_YEAR), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for year, records in sorted(movies_by_year.items()):
            print(f'<h2>{year}</h2>', file=f)
            for record in records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_alpha_page(rep, forcethumb):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)

    movies_by_alpha = defaultdict(list)
    for record in movies:
        movies_by_alpha[record['title'][0].upper()].append(record)

    with open(os.path.join(rep, MOVIES_ALPHA), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for char, records in sorted(movies_by_alpha.items()):
            print(f'<h2>{char}</h2>', file=f)
            for record in records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_director_page(rep, forcethumb):
    with open(os.path.join(rep, 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)

    movies_by_director = defaultdict(list)
    for record in movies:
        for director in record['director']:
            movies_by_director[director].append(record)

    with open(os.path.join(rep, MOVIES_DIRECTOR), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for director, records in sorted(movies_by_director.items()):
            print(f'<h2> {director}</h2>', file=f)
            for record in records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_main_page(rep, forcethumb):
    make_vrac_page(rep, forcethumb=forcethumb)
    make_year_page(rep, forcethumb=False)
    make_alpha_page(rep, forcethumb=False)
    make_director_page(rep, forcethumb=False)
    shutil.copy('movies.htm', rep)


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

        html = template[:]
        html = html.replace('{{cover}}', image_basename)
        html = html.replace('{{title}}', record['title'])
        html = html.replace('{{year}}', str(record['year']))
        html = html.replace('{{runtime}}', str(record['runtime']))
        html = html.replace('{{width}}', str(record['width']))
        html = html.replace('{{height}}', str(record['height']))
        html = html.replace('{{filesize}}', str(record['filesize']))
        html = html.replace('{{cast}}', make_li_list(record['cast'] if record['cast'] else ['Non renseigné']))

        if record['director']:
            first_director = record['director'][0]
            other_directors = record['director'][1:]
            if other_directors:
                html = html.replace('{{director}}', make_li_list([first_director, ', '.join(other_directors)]))
            else:
                html = html.replace('{{director}}', make_li_list([first_director]))

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
        else:
            html = html.replace('{{director}}', make_li_list(['Non renseigné']))
            html = html.replace('{{other_movies}}', '\n')

        with open(html_name, 'wt', encoding='utf-8') as f:
            print(html, file=f)


def clean(rep):
    for dirpath, _, barename in movie_gen(rep):
        jsonname = os.path.join(dirpath, barename + '.json')
        imgname = os.path.join(dirpath, barename + '.jpg')
        if os.path.isfile(jsonname) and not os.path.isfile(imgname):
            print(jsonname)
            os.remove(jsonname)


def test():
    # create an instance of the Cinemagoer class
    ia = Cinemagoer()

    # get a movie
    movie = ia.get_movie('0133093')
    # movies = ia.search_movie("C'est arrivé près de chez vous")
    # movie = movies[0]
    print(movie, movie.movieID)
    print(movie.get('countries'))
    print(movie.get('original title'))
    print(movie.get('title'))
    print(movie.infoset2keys)

    # search for a person name
    people = ia.search_person('Mel Gibson')
    for person in people:
        print(person.personID, person['name'])


def parse_command_line():
    parser = argparse.ArgumentParser(add_help=True, usage=__doc__)
    xgroup = parser.add_mutually_exclusive_group()
    xgroup.add_argument('--extract_movie_tsv', action='store_true', default=False)
    xgroup.add_argument('--extract_data', action='store', metavar='<movies rep>')
    xgroup.add_argument('--make_pages', action='store', metavar='<movies rep>')
    xgroup.add_argument('--update', action='store', metavar='<movies rep>')
    xgroup.add_argument('--test', action='store_true')
    parser.add_argument('--force_json', action='store_true', default=False)
    parser.add_argument('--force_thumb', action='store_true', default=False)
    args = parser.parse_args()
    if args.extract_data:
        args.rep = args.extract_data
    if args.make_pages:
        args.rep = args.make_pages
    if args.update:
        args.rep = args.update
    return parser, args


def main():
    parser, args = parse_command_line()
    if args.extract_movie_tsv:
        extract_movie_tsv(MOVIE_TSV_FN)
    elif args.extract_data:
        create_missing_records(args.rep, 'movie.tsv', args.force_json)
        make_index(args.rep)
    elif args.make_pages:
        make_main_page(args.rep, args.force_thumb)
        make_movie_pages(args.rep)
    elif args.update:
        create_missing_records(args.rep, 'movie.tsv', args.force_json)
        make_index(args.rep)
        make_main_page(args.rep, args.force_thumb)
        make_movie_pages(args.rep)
    elif args.test:
        test()
    else:
        parser.print_help()


main()
