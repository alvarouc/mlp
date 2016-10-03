from setuptools import setup
setup(
    name='mlp',
    packages=['mlp'],
    version='0.5',
    install_requires=[
        'keras',
        'sklearn',
        'numpy'],
    description='A multilayer perceptron implementation using keras and compatible with scikit-learn',
    author='Alvaro Ulloa',
    author_email='alvarouc@gmail.com',
    url='https://github.com/alvarouc/mlp',
    download_url='https://github.com/alvarouc/mlp/tarball/0.5',
    keywords=['neural net', 'mlp', 'deep learning'],
    classifiers=[],
)
