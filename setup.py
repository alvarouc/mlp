from setuptools import setup
setup(
    name = 'mlp',
    packages = ['mlp'], # this must be the same as the name above
    version = '0.2.1',
    install_requires=[
        'keras',
        'sklearn',
        'numpy'],
    description = 'A multilayer perceptron implementation using keras and compatible with scikit-learn',
    author = 'Alvaro Ulloa',
    author_email = 'alvarouc@gmail.com',
    url = 'https://github.com/alvarouc/mlp', # use the URL to the github repo
    download_url = 'https://github.com/alvarouc/mlp/tarball/0.1', # I'll explain this in a second
    keywords = ['neural net', 'mlp', 'deep learning'], # arbitrary keywords
    classifiers = [],
)
