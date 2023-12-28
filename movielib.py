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
import tempfile
import glob
from subprocess import check_output, CalledProcessError, STDOUT
from collections import defaultdict
from functools import cache
from pathlib import Path

import requests
from PIL import Image
from imdb import Cinemagoer
from jinja2 import Environment, FileSystemLoader


MOVIE_TSV = 'movie.tsv'
TITLES_INDEX = 'titlestsv.pickle'
MENUFILE = 'menu.htm'
TEMPLATE_GALLERY = 'template-gallery.htm'
TEMPLATE_STATS = 'template-stats.htm'
TEMPLATE_MOVIE = 'template-movie.htm'
MOVIES_VRAC = 'movies-vrac.htm'
MOVIES_YEAR = 'movies-year.htm'
MOVIES_ALPHA = 'movies-alpha.htm'
MOVIES_DIRECTOR = 'movies-director.htm'
MOVIES_ACTOR = 'movies-actor.htm'
MOVIES_STATS = 'movies-stats.htm'


# -- Helpers ------------------------------------------------------------------


def thumbname(name, key):
    return key + '-' + name + '.jpg'


def size_thumbnail(width, height, maxdim):
    if width >= height:
        return maxdim, int(round(maxdim * height / width))
    else:
        return int(round(maxdim * width / height)), maxdim


def make_thumbnail_image(args, image_name, thumb_name, size):
    if os.path.exists(thumb_name) and args.forcethumb is False:
        pass
    else:
        print('Making thumbnail:', thumb_name)
        create_thumbnail_image(image_name, thumb_name, size)


def create_thumbnail_image(image_name, thumb_name, size):
    imgobj = Image.open(image_name)

    if (imgobj.mode != 'RGBA'
        and image_name.endswith('.jpg')
        and not (image_name.endswith('.gif') and imgobj.info.get('transparency'))
       ):
        imgobj = imgobj.convert('RGBA')

    imgobj.thumbnail(size, Image.LANCZOS)
    imgobj = imgobj.convert('RGB')
    imgobj.save(thumb_name)


# -- Pass 1: extract data from title.basics.tsv.gz ----------------------------


@cache
def cachedir():
    tempdir = tempfile.gettempdir()
    cache_dir = os.path.join(tempdir, 'movielib')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def extract_movie_tsv():
    """
    Assume that the file title.basics.tsv.gz has been downloaded from
    https://developer.imdb.com/non-commercial-datasets/ and that it contains
    the file data.tsv.
    """
    with gzip.open('title.basics.tsv.gz', 'rb') as f_in:
        with open(os.path.join(cachedir(), 'data.tsv'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    titles = defaultdict(set)
    with open(os.path.join(cachedir(), 'data.tsv'), encoding='utf-8') as f:
        with open(os.path.join(cachedir(), MOVIE_TSV), 'wt', encoding='utf-8') as g:
            print(f.readline(), end='', file=g)
            for line in f:
                if '\tmovie\t' in line:
                    print(line, end='', file=g)
                    tconst, _, primary_title, original_title, _, year, _, _, _ = line.split('\t')

                    primary_title = primary_title.lower()
                    primary_title = primary_title.replace(':', '')
                    primary_title = primary_title.replace('.', '')
                    primary_title = primary_title.replace('  ', ' ')
                    original_title = original_title.lower()
                    original_title = original_title.replace(':', '')
                    original_title = original_title.replace('.', '')
                    original_title = original_title.replace('  ', ' ')

                    titles[primary_title].add(tconst)
                    titles[original_title].add(tconst)
                    titles[f'{primary_title}-{year}'].add(tconst)
                    titles[f'{original_title}-{year}'].add(tconst)

    os.remove(os.path.join(cachedir(), 'data.tsv'))
    with open(os.path.join(cachedir(), TITLES_INDEX), 'wb') as f:
        pickle.dump(dict(titles), f)


@cache
def titles_index():
    print('Loading titles index...')
    with open(os.path.join(cachedir(), TITLES_INDEX), 'rb') as f:
        titles = pickle.load(f)
    print('Loaded')
    return titles


# -- Pass 2: make json records and download default movie cover if required ---


def title_imdb_id(title, year=None):
    key = title if (year is None) else f'{title}-{year}'
    return titles_index().get(key.lower(), None)


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
    return os.system(command)


def get_title(name, movie):
    """
    name: title of movie as extracted from file name
    movie: movie object retrieved from imdb
    """
    if name == movie.get('title'):              # equals to primaryTitle from title.basics.tsv.gz
        return name
    elif name ==  movie.get('original title'):  # equals to originalTitle from title.basics.tsv.gz
        return name
    elif movie.get('countries')[0] == 'France':
        return movie.get('original title')
    else:
        return movie.get('title')


def wikipedia_url(title, year):
    title = title.replace(' ', '_')
    url1 = f'https://en.wikipedia.org/wiki/{title}_({year}_film)'
    url2 = f'https://en.wikipedia.org/wiki/{title}_(film)'
    url3 = f'https://en.wikipedia.org/wiki/{title}'

    for url in (url1, url2, url3):
        try:
            r = requests.get(url, timeout=10)
        except requests.exceptions.ConnectionError:
            print('Wikipedia connection', 'FAILURE', 'for', title)
            continue
        except requests.exceptions.ReadTimeout:
            print('Wikipedia connection', 'TIMEOUT', 'for', title)
            continue
        if r.status_code == 200:
            return url

    return None


EMPTY = {
    'imdb_id': None,
    'title': None,
    'year': None,
    'director': [],
    'cast': [],
    'runtime': None,
    'filesize': None,
    'width': None,
    'height': None,
    'wikipedia_url': None
}


def create_minimal_record(dirpath, filename, name, year):
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


def create_record(dirpath, filename, name, ia, imdb_id):
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
    record['title'] = get_title(name, movie)
    record['year'] = movie.get('year')
    record['runtime'] = movie.get('runtimes')[0]
    record['director'] = [_.get('name') for _ in movie.get('director')]
    record['cast'] = [_.get('name') for _ in movie.get('cast')]

    # set wikipedia url
    record['wikipedia_url'] = wikipedia_url(record['title'], record['year'])

    # load default movie cover
    imgname = os.path.splitext(fullname)[0] + '.jpg'
    if os.path.isfile(imgname) is False:
        imgdata = requests.get(movie.get('cover url'), timeout=10).content
        with open(imgname, 'wb') as handler:
            handler.write(imgdata)

    return record


def movie_gen(rep):
    """
    Find recursively all movies in rep.
    """
    for dirpath, _, filenames in os.walk(rep):
        for filename in filenames:
            barename, ext = os.path.splitext(filename)
            if ext in ('.mp4', '.avi', '.mkv'):
                yield dirpath, filename, barename


def create_missing_records(rep, forcejson=False):
    """
    Find recursively all movies in rep. Create json file (with same name as
    movie) if absent. Fill record with imdb data if imdb id can be found, plus
    data relative to file.
    forcejson enables to reset content (mainly for dev). Records without IMDB
    id are ignored as their content is assumed to be completed manually.
    """
    ia = Cinemagoer()
    movie_number = 0
    new_movie_number = 0
    movie_found = 0

    for dirpath, filename, barename in movie_gen(rep):
        movie_number += 1

        jsonname = os.path.join(dirpath, barename + '.json')
        if os.path.isfile(jsonname):
            if forcejson:
                with open(jsonname) as f:
                    record = json.loads(f.read())
                if record['imdb_id'] is None:
                    continue
            else:
                continue

        new_movie_number += 1

        if re.search(r'\(\d\d\d\d\)\s*$', barename):
            match = re.match(r'\s*(.*)\s*\((\d\d\d\d)\)\s*$', barename)
            name = match.group(1).strip()
            year = int(match.group(2))
            imdb_id = title_imdb_id(name, year)
        else:
            name = barename.strip()
            year = 9999
            imdb_id = title_imdb_id(name)

        if imdb_id and len(imdb_id) > 1:
            print(name, 'ambiguous', imdb_id)
            continue
        elif imdb_id is None:
            print(name, 'not found in imdb')
            record = create_minimal_record(dirpath, filename, name, year)
        else:
            # movie found
            imdb_id = list(imdb_id)[0]
            movie_found += 1
            record = create_record(dirpath, filename, name, ia, imdb_id)

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


# -- Pass 3: make gallery html files and thumbnails ---------------------------


def load_records(rep):
    records = []
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
            record['cover'] = barename + '.jpg'
            record['year_title'] = f"{record['year']}: {record['title']}"
            nback = len(Path(record['dirpath']).parts) - len(Path(rep).parts)
            record['relpath_to_root'] = '../' * nback
            records.append(record)

    return records


def relevant_cast_v1(record, actor_movies):
    """
    An actor is taken into account for a movie if:
    - he appears in the p first actors in the cast list,
    - or he appears in the q (q > p) first actors and he appears in at least one
      other movie in the database.
    """
    p = 5
    q = 10
    cast = record['cast'][:p]
    for actor in record['cast'][p:q]:
        if len(actor_movies[actor]) >= 2:
            cast.append(actor)
    return cast


def relevant_cast(record, records, actor_movies, yearmovie_num):
    """
    An actor is taken into account for a movie if:
    - he appears in the p first actors in the cast list,
    - or he appears in the q (q > p) first actors and he appears in the p first
      actors of at least one other movie in the database.
    """
    p = 5
    q = 10
    cast = record['cast'][:p]
    for actor in record['cast'][p:q]:
        for yeartitle in actor_movies[actor]:
            cast2 = records[yearmovie_num[yeartitle]]['cast']
            if yeartitle != record['year_title'] and cast2.index(actor) <= p:
                cast.append(actor)
                break
    return cast


def load_main_cast(records):
    actor_movies = defaultdict(list)
    yearmovie_num = {}
    for record in records:
        yearmovie_num[record['year_title']] = record['movienum']
        for _ in record['cast']:
            actor_movies[_].append(record['year_title'])
    for actor, movies in actor_movies.items():
        actor_movies[actor] = sorted(movies)

    for record in records:
        record['main_cast'] = relevant_cast(record, records, actor_movies, yearmovie_num)


def escape_open_url(url):
    url = url.replace('\\', '/')
    url = url.replace("'", "\\'")
    return url


def urlencode(url):
    url = url.replace('\\', '/')
    url = url.replace(' ', '%20')
    return url


def time_ordered(fn1, fn2):
    """
    Check if two files are time ordered.
    """
    return os.path.getmtime(fn1) < os.path.getmtime(fn2)


def update_movie_record(rep, record, forcethumb=False):
    movie_name = os.path.join(record['dirpath'], record['filename'])
    image_basename = record['cover']
    image_name = os.path.join(record['dirpath'], image_basename)
    thumb_basename = thumbname(record['cover'], 'film')
    thumb_name = os.path.join(rep, '.gallery', '.thumbnails', thumb_basename)
    html_basename = record['barename'] + '.htm'
    html_name = os.path.join(record['dirpath'], html_basename)

    record['thumb_path'] = os.path.join('.thumbnails', thumb_basename)
    record['hover_text'] = f"{record['title']}, {record['year']}, {', '.join(record['director'])}"
    record['movie_page'] = escape_open_url(os.path.relpath(html_name, start=os.path.join(rep, '.gallery')))

    if forcethumb or os.path.isfile(thumb_name) is False or time_ordered(image_name, thumb_name) is False:
        args = types.SimpleNamespace()
        args.forcethumb = True
        if os.path.isfile(image_name):
            width, height = Image.open(image_name).size
            thumbsize = size_thumbnail(width, height, maxdim=300)
            make_thumbnail_image(args, image_name, thumb_name, thumbsize)
        else:
            print('Warning: no image for', movie_name)


MENU = '<iframe src="%s" height=220px style="position: fixed; top: 20px; right: 40px; border-style: none!important;"></iframe>'


def make_gallery_page(pagename, rep, records, forcethumb, index, sorted_records, tags, caption):
    for record in records:
        update_movie_record(rep, record, forcethumb=forcethumb)

    file_loader = FileSystemLoader('')
    env = Environment(loader=file_loader)
    template = env.get_template('template-gallery.htm')

    html = template.render(
        records=records,
        index=index,
        sorted_records=sorted_records,
        tags=tags,
        caption=caption,
        menu=MENU % MENUFILE,
        icon='movies-icon.png'
    )
    with open(os.path.join(rep, '.gallery', pagename), 'wt', encoding='utf-8') as f:
        print(html, file=f)


def make_vrac_page(rep, records, forcethumb):
    sorted_records = {None: records}
    tags = {None: None}
    index = None
    make_gallery_page(MOVIES_VRAC, rep, records, forcethumb, index, sorted_records, tags, False)


def make_year_page(rep, records, forcethumb):
    movies_by_year = defaultdict(list)
    for record in records:
        movies_by_year[record['year']].append(record)

    first_year_in_decade = {}
    tags = {}
    for year in sorted(movies_by_year):
        if first_year_in_decade.get(year - year % 10, None) is None:
            first_year_in_decade[year - year % 10] = year
            tags[year] = year - year % 10
        else:
            tags[year] = None
    index = list(sorted(first_year_in_decade))

    make_gallery_page(MOVIES_YEAR, rep, records, forcethumb, index, movies_by_year, tags, False)


def make_alpha_page(rep, records, forcethumb):
    movies_by_alpha = defaultdict(list)
    for record in records:
        movies_by_alpha[record['title'][0].upper()].append(record)

    tags = {}
    for char, movies in movies_by_alpha.items():
        tags[char] = char
        movies_by_alpha[char] = sorted(movies, key=lambda rec: rec['title'])
    index = list(sorted(tags))

    make_gallery_page(MOVIES_ALPHA, rep, records, forcethumb, index, movies_by_alpha, tags, False)


def make_director_page(rep, records, forcethumb):
    movies_by_director = defaultdict(list)
    for record in records:
        for director in record['director']:
            movies_by_director[director].append(record)

    first_director = {}
    tags = {}
    for director in sorted(movies_by_director):
        if first_director.get(director[0], None) is None:
            first_director[director[0]] = director
            tags[director] = director[0]
        else:
            tags[director] = None
    index = list(sorted(first_director))

    make_gallery_page(MOVIES_DIRECTOR, rep, records, forcethumb, index, movies_by_director, tags, True)


def make_actor_page(rep, records, forcethumb):
    movies_by_actor = defaultdict(list)
    for record in records:
        for actor in record['main_cast']:
            movies_by_actor[actor].append(record)

    first_actor = {}
    tags = {}
    for actor in sorted(movies_by_actor):
        movies_by_actor[actor] = sorted(movies_by_actor[actor], key=lambda rec: rec['year_title'])
        if first_actor.get(actor[0], None) is None:
            first_actor[actor[0]] = actor
            tags[actor] = actor[0]
        else:
            tags[actor] = None
    index = list(sorted(first_actor))

    make_gallery_page(MOVIES_ACTOR, rep, records, forcethumb, index, movies_by_actor, tags, True)


def space_thousands(n):
    return f'{n:,}'.replace(',', ' ')


def make_stats_page(rep, records):
    data = []
    total = 0
    for record in records:
        data.append((
            record['title'],
            record['year'],
            record['width'],
            record['height'],
            space_thousands(record["filesize"])
        ))
        total += record["filesize"]

    data.append(('Total', '', '', '', space_thousands(total)))

    file_loader = FileSystemLoader('')
    env = Environment(loader=file_loader)
    template = env.get_template('template-stats.htm')

    html = template.render(
        data=data,
        menu=MENU % MENUFILE,
        icon='movies-icon.png'
    )
    with open(os.path.join(rep, '.gallery', MOVIES_STATS), 'wt', encoding='utf-8') as f:
        print(html, file=f)


# -- Pass 4: make movie html files --------------------------------------------


def imdb_link(record):
    if record['imdb_id']:
        url = 'https://www.imdb.com/title/tt%s/' % record['imdb_id']
        return f'href="javascript:window.open(\'{url}\', \'_top\')"'
    else:
        return 'class="disabled"'


def wikipedia_link(record):
    if record['wikipedia_url']:
        url = escape_open_url(record['wikipedia_url'])
        return f'href="javascript:window.open(\'{url}\', \'_top\')"'
    else:
        return 'class="disabled"'


def google_link(record):
    search =  re.sub(r'[\W ]+', ' ', record['title'], flags=re.U)
    words = search.split() + [str(record['year']), 'movie']
    url = 'https://www.google.com/search?q=' + '+'.join(words)
    return f'href="javascript:window.open(\'{url}\', \'_top\')"'


def relpath_to_menu(record):
    return record['relpath_to_root'] + '.gallery/menu.htm'


def relpath_to_icon(record):
    return record['relpath_to_root'] + '.gallery/movies-icon.png'


def relpath_to_movie(rep, records, record, yearmovie, yearmovie_num):
    record_target = records[yearmovie_num[yearmovie]]
    path = os.path.relpath(record_target['dirpath'], start=rep)
    return record['relpath_to_root'] + os.path.join(path, record_target['barename'] + '.htm')


def movie_record_html(rep, records, record, yearmovie_num, director_movies, actor_movies, template):
    if not record['director']:
        record['director_list'] = []
    else:
        first_director = record['director'][0]
        other_directors = record['director'][1:]
        if other_directors:
            record['director_list'] = [first_director, ', '.join(other_directors)]
        else:
            record['director_list'] = [first_director]

    if record['director']:
        othermovies1 = [_ for _ in director_movies[first_director] if record['year_title'] != _]
        othermovies2 = set()
        for director in other_directors:
            othermovies2.update([_ for _ in director_movies[director] if record['year_title'] != _])
        othermovies1 = sorted(othermovies1)
        othermovies2 = sorted(othermovies2)
        record['dirothermovies'] = [othermovies1, othermovies2]
        record['path_to_dirothermovies'] = [
            [relpath_to_movie(rep, records, record, _, yearmovie_num) for _ in othermovies1],
            [relpath_to_movie(rep, records, record, _, yearmovie_num) for _ in othermovies2],
        ]

    if not record['cast']:
        record['actor_list'] = []
    else:
        maincast = record['main_cast']
        record['actor_list'] = maincast
        if len(record['cast']) > len(maincast):
            record['actor_list'].append(', '.join([_ for _ in record['cast'] if _ not in maincast]))

    if record['cast']:
        record['castothermovies'] = []
        for actor in maincast:
            record['castothermovies'].append([_ for _ in actor_movies[actor] if record['year_title'] != _])
            record['path_to_castothermovies'] = []
            for movies in record['castothermovies']:
                relpaths = [relpath_to_movie(rep, records, record, _, yearmovie_num) for _ in movies]
                record['path_to_castothermovies'].append(relpaths)

    html = template.render(
        title=record['title'],
        menu=MENU % relpath_to_menu(record),
        movie_link=record['filename'],
        imdb_link=imdb_link(record),
        wikipedia_link=wikipedia_link(record),
        google_link=google_link(record),
        record=record,
        icon=record['cover'],  # relpath_to_icon(record),
        zip=zip,
        space_thousands=space_thousands,
    )

    return html


def make_movie_pages(rep, records):
    yearmovie_num = {}
    for record in records:
        yearmovie_num[record['year_title']] = record['movienum']

    director_movies = defaultdict(list)
    for record in records:
        for _ in record['director']:
            director_movies[_].append(record['year_title'])

    actor_movies = defaultdict(list)
    for record in records:
        for _ in record['cast']:
            actor_movies[_].append(record['year_title'])
    for actor, movies in actor_movies.items():
        actor_movies[actor] = sorted(movies)

    file_loader = FileSystemLoader('')
    env = Environment(loader=file_loader)
    template = env.get_template('template-movie.htm')

    for record in records:
        html = movie_record_html(rep, records, record, yearmovie_num, director_movies, actor_movies, template)
        html_basename = record['barename'] + '.htm'
        html_name = os.path.join(record['dirpath'], html_basename)
        with open(html_name, 'wt', encoding='utf-8') as f:
            print(html, file=f)


def make_html_pages(rep, forcethumb):
    os.makedirs(os.path.join(rep, '.gallery', '.thumbnails'), exist_ok=True)
    records = load_records(rep)
    load_main_cast(records)
    make_vrac_page(rep, records, forcethumb=forcethumb)
    make_year_page(rep, records, forcethumb=False)
    make_alpha_page(rep, records, forcethumb=False)
    make_director_page(rep, records, forcethumb=False)
    make_actor_page(rep, records, forcethumb=False)
    make_stats_page(rep, records)
    make_movie_pages(rep, records)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'movies.htm'), rep)
    shutil.copy(os.path.join(os.path.dirname(__file__), MENUFILE), os.path.join(rep, '.gallery'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'movies-icon.png'), os.path.join(rep, '.gallery'))
    shutil.copy(os.path.join(os.path.dirname(__file__), 'top-icon.png'), os.path.join(rep, '.gallery'))


# -- Test functions -----------------------------------------------------------


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


def stats_images(rep):
    records = load_records(rep)
    ratios = []
    for record in records:
        image_basename = record['barename'] + '.jpg'
        image_name = os.path.join(record['dirpath'], image_basename)
        if os.path.isfile(image_name):
            width, height = Image.open(image_name).size
            ratios.append(height / width)
    print('Mean image ratio', sum(ratios) / len(ratios))


def stats_cast(rep):
    records = load_records(rep)
    movies_by_actor = defaultdict(list)
    for record in records:
        for index, actor in enumerate(record['cast'][:10], 1):
            movies_by_actor[actor].append(index)

    for actor, ranks in sorted(movies_by_actor.items()):
        print(actor, sorted(ranks))


# -- Main ---------------------------------------------------------------------


def lang_choices():
    choices = []
    for x in glob.glob('menu-*.htm'):
        match = re.match(r'menu-(\w+)\.htm', x)
        choices.append(match[1])
    return choices


def parse_command_line():
    parser = argparse.ArgumentParser(add_help=True, usage=__doc__)
    xgroup = parser.add_mutually_exclusive_group()
    xgroup.add_argument('--extract_movie_tsv', action='store_true', default=False)
    xgroup.add_argument('--extract_data', action='store', metavar='<movies rep>')
    xgroup.add_argument('--make_pages', action='store', metavar='<movies rep>')
    xgroup.add_argument('--update', action='store', metavar='<movies rep>')
    xgroup.add_argument('--test', action='store_true')
    parser.add_argument('--language', action='store', choices=lang_choices(), default='EN')
    parser.add_argument('--force_json', action='store_true', default=False)
    parser.add_argument('--force_thumb', action='store_true', default=False)
    args = parser.parse_args()
    if args.extract_data:
        args.rep = args.extract_data
    if args.make_pages:
        args.rep = args.make_pages
    if args.update:
        args.rep = args.update

    shutil.copy(f'menu-{args.language}.htm', MENU)
    shutil.copy(f'template-movie-{args.language}.htm', TEMPLATE_MOVIE)
    return parser, args


def main():
    parser, args = parse_command_line()
    if args.extract_movie_tsv:
        extract_movie_tsv()
    elif args.extract_data:
        create_missing_records(args.rep, args.force_json)
    elif args.make_pages:
        make_html_pages(args.rep, args.force_thumb)
    elif args.update:
        create_missing_records(args.rep, args.force_json)
        make_html_pages(args.rep, args.force_thumb)
    elif args.test:
        # test()
        stats_cast('d:/films')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
