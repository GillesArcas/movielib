from setuptools import setup

setup(
    name='movielib',
    version='0.0',
    license='MIT',
    url = 'https://github.com/GillesArcas/movielib',
    author = 'Gilles Arcas',
    author_email = 'gilles.arcas@gmail.com',
    entry_points = {
        'console_scripts': ['movielib=movielib:main'],
    },
    zip_safe=False,
    install_requires = [
        'pillow',
        'requests',
        'cinemagoer',
    ]
)
