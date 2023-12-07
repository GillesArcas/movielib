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
MOVIES_STATS = 'movies-stats.htm'


# -- Pass 1: extract data from title.basics.tsv.gz


def extract_movie_tsv(movie_tsv_filename):
    """
    Assume that the file title.basics.tsv.gz has been downloaded from
    https://developer.imdb.com/non-commercial-datasets/ and that it contains
    the file data.tsv.
    """
    with gzip.open('title.basics.tsv.gz', 'rb') as f_in:
        with open('data.tsv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    titles = defaultdict(set)
    with open('data.tsv', encoding='utf-8') as f:
        with open(movie_tsv_filename, 'wt', encoding='utf-8') as g:
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

    os.remove('data.tsv')
    with open('titlestsv.pickle', 'wb') as f:
        pickle.dump(dict(titles), f)


def load_movie_tsv(movie_tsv_filename):
    with open('titlestsv.pickle', 'rb') as f:
        titles = pickle.load(f)
    return titles


# -- Pass 2: make json records and download default movie cover if required


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
    record['cast'] = [_.get('name') for _ in movie.get('cast')[:5]]

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
    titles = load_movie_tsv(tsvfile)
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


# -- Pass 3: make html files and thumbnails


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
            records.append(record)

    return records


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

IMAGE_ONCLICK = '''
<img src="%s" width="%d" alt="%s cover" title="%s" onclick="window.open('%s', '_self')">
'''

def image_onclick_element(record, rep, thumb_width, thumb_basename, html_name):
    return IMAGE_ONCLICK % (
        urlencode(os.path.join('.thumbnails', thumb_basename)),
        thumb_width,
        record['title'],
        f"{record['title']}, {record['year']}, {', '.join(record['director'])}",
        urlencode(os.path.relpath(html_name, start=os.path.join(rep, '.gallery')))
    )

VIDPOSTCAPTION = '''\
<span>
%s
<p>%s</p>
</span>
%s
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

    movie_element = VIDPOSTCAPTION % (
        image_onclick_element(record, rep, thumb_width, thumb_basename, html_name),
        record['title'],
        ''
    )

    return movie_element


def make_vrac_page(rep, records, forcethumb):
    with open(os.path.join(rep, '.gallery', MOVIES_VRAC), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for record in records:
            print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_year_page(rep, records, forcethumb):
    movies_by_year = defaultdict(list)
    for record in records:
        movies_by_year[record['year']].append(record)

    with open(os.path.join(rep, '.gallery', MOVIES_YEAR), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for year, year_records in sorted(movies_by_year.items()):
            print(f'<h2>{year}</h2>', file=f)
            for record in year_records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_alpha_page(rep, records, forcethumb):
    movies_by_alpha = defaultdict(list)
    for record in records:
        movies_by_alpha[record['title'][0].upper()].append(record)

    with open(os.path.join(rep, '.gallery', MOVIES_ALPHA), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for char, char_records in sorted(movies_by_alpha.items()):
            print(f'<h2>{char}</h2>', file=f)
            for record in char_records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


def make_director_page(rep, records, forcethumb):
    movies_by_director = defaultdict(list)
    for record in records:
        for director in record['director']:
            movies_by_director[director].append(record)

    with open(os.path.join(rep, '.gallery', MOVIES_DIRECTOR), 'wt', encoding='utf-8') as f:
        print(START % 'Films', file=f)
        for director, dir_records in sorted(movies_by_director.items()):
            print(f'<h2> {director}</h2>', file=f)
            for record in dir_records:
                print(make_movie_element(rep, record, 160, forcethumb), file=f)
        print(END, file=f)


STATS_TEMPLATE = '''
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<title>%s</title>
<link rel="icon" href="Movies-icon.png" />
<style type="text/css">
body {
  font: normal 14px Verdana, Arial, sans-serif;
}
td {
  padding-left: 15px;
  padding-right: 15px;
  text-align: right;
}
</style>\
<base target="_parent"></base>
</head>

<body>
<div style="width: 95%%; margin-left: auto; margin-right: auto">\
<table>
{{content}}
</table>
</div>
</body>
</html>
'''


def space_thousands(n):
    return f'{n:,}'.replace(',', ' ')


def make_stats_page(rep, records):
    rows = []
    total = 0
    for record in records:
        data = (
            record['title'],
            record['year'],
            record['width'],
            record['height'],
            space_thousands(record["filesize"])
        )
        rows.extend(['<tr>'] + [f'<td>{_}</td>' for _ in data] + ['</tr>'])
        total += record["filesize"]

    data = ('Total', '', '', '', space_thousands(total))
    rows.extend(['<tr>'] + [f'<td>{_}</td>' for _ in data] + ['</tr>'])
    content = STATS_TEMPLATE.replace('{{content}}', '\n'.join(rows))
    with open(os.path.join(rep, '.gallery', MOVIES_STATS), 'wt', encoding='utf-8') as f:
        print(content, file=f)


def make_li_list(liste):
    return '\n'.join([f'<li>{_}</li>' for _ in liste])


OTHER_DIRECTOR_MOVIES = '''\
<h3>Autres films de %s dans la collection</h3>
 <ul>
    %s
</ul>
'''


def year_title(record):
    return f"{record['year']}: {record['title']}"


def imdb_link(record):
    if record['imdb_id']:
        url = 'https://www.imdb.com/title/tt%s/' % record['imdb_id']
        return f'href="javascript:window.open(\'{url}\', \'_top\')"'
    else:
        return 'class="disabled"'


def wikipedia_link(record):
    if record['wikipedia_url']:
        url = record['wikipedia_url']
        return f'href="javascript:window.open(\'{url}\', \'_top\')"'
    else:
        return 'class="disabled"'


def google_link(record):
    if record['imdb_id'] or record['wikipedia_url']:
        return 'class="hidden"'
    else:
        search =  re.sub(r'[\W ]+', ' ', record['title'], flags=re.U)
        words = search.split() + [str(record['year']), 'movie']
        url = 'https://www.google.com/search?q=' + '+'.join(words)
        return f'href="javascript:window.open(\'{url}\', \'_top\')"'


def movie_record_html(record, template, director_movies):
    movie_name = os.path.join(record['dirpath'], record['filename'])
    image_basename = record['barename'] + '.jpg'

    html = template[:]
    html = html.replace('{{cover}}', image_basename)
    html = html.replace('{{title}}', record['title'])
    html = html.replace('{{year}}', str(record['year']))
    html = html.replace('{{runtime}}', str(record['runtime']))
    html = html.replace('{{width}}', str(record['width']))
    html = html.replace('{{height}}', str(record['height']))
    html = html.replace('{{filesize}}', space_thousands(record["filesize"]))
    html = html.replace('{{cast}}', make_li_list(record['cast'] if record['cast'] else ['Non renseigné']))

    html = html.replace('{{movie_link}}', f'file:///{movie_name}')
    html = html.replace('{{imdb_link}}', imdb_link(record))
    html = html.replace('{{wikipedia_link}}', wikipedia_link(record))
    html = html.replace('{{google_link}}', google_link(record))

    if record['director']:
        first_director = record['director'][0]
        other_directors = record['director'][1:]
        othermovies1 = [_ for _ in director_movies[first_director] if year_title(record) != _]
        othermovies2 = set()
        for director in other_directors:
            othermovies2.update([_ for _ in director_movies[director] if year_title(record) != _])
        othermovies1 = ['Aucun'] if not othermovies1 else sorted(othermovies1)
        othermovies2 = ['Aucun'] if not othermovies2 else sorted(othermovies2)

        if other_directors:
            html = html.replace('{{director}}', make_li_list([first_director, ', '.join(other_directors)]))
        else:
            html = html.replace('{{director}}', make_li_list([first_director]))

        othermovieshtml = [OTHER_DIRECTOR_MOVIES % (first_director, make_li_list(othermovies1))]
        if other_directors:
            if len(other_directors) > 1:
                other_directors = other_directors[:1] + ['etc.']
            othermovieshtml.append(OTHER_DIRECTOR_MOVIES % (', '.join(other_directors), make_li_list(othermovies2)))

        html = html.replace('{{other_movies}}', '\n'.join(othermovieshtml))
    else:
        html = html.replace('{{director}}', make_li_list(['Non renseigné']))
        html = html.replace('{{other_movies}}', '\n')

    return html


def make_movie_pages(rep, records):
    with open(os.path.join(os.path.dirname(__file__), 'template.htm'), encoding='utf-8') as f:
        template = f.read()

    director_movies = defaultdict(list)
    for record in records:
        for _ in record['director']:
            director_movies[_].append(year_title(record))

    for record in records:
        html = movie_record_html(record, template, director_movies)
        html_basename = record['barename'] + '.htm'
        html_name = os.path.join(record['dirpath'], html_basename)
        with open(html_name, 'wt', encoding='utf-8') as f:
            print(html, file=f)


def make_html_pages(rep, forcethumb):
    os.makedirs(os.path.join(rep, '.gallery', '.thumbnails'), exist_ok=True)
    records = load_records(rep)
    make_vrac_page(rep, records, forcethumb=forcethumb)
    make_year_page(rep, records, forcethumb=False)
    make_alpha_page(rep, records, forcethumb=False)
    make_director_page(rep, records, forcethumb=False)
    make_stats_page(rep, records)
    make_movie_pages(rep, records)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'movies.htm'), rep)


# -- Main


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
    elif args.make_pages:
        make_html_pages(args.rep, args.force_thumb)
    elif args.update:
        create_missing_records(args.rep, 'movie.tsv', args.force_json)
        make_html_pages(args.rep, args.force_thumb)
    elif args.test:
        test()
    else:
        parser.print_help()


main()
