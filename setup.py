from setuptools import setup
import setuptools.command.build_py

def readme():
    with open('README.rst') as f:
        return f.read()

class BuildTestCommand(setuptools.command.build_py.build_py):
    """Custom build command."""

    def run(self):

        # run tests using doctest
        import doctest
        from ipfml import processing
        from ipfml import metrics
        from ipfml.filters import noise as noise_filters
        from ipfml.iqa import fr as fr_iqa
        from ipfml import utils

        print("==============================")
        print("Runs test command...")
        doctest.testmod(processing)
        doctest.testmod(metrics)
        doctest.testmod(noise_filters)
        doctest.testmod(fr_iqa)
        doctest.testmod(utils)

        setuptools.command.build_py.build_py.run(self)


setup(
    name='ipfml',
    version='0.3.7',
    description='Image Processing For Machine Learning',
    long_description=readme(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    url='https://github.com/jbuisine/IPFML',
    author='Jérôme BUISINE',
    author_email='jerome.buisine@univ-littoral.fr',
    license='MIT',
    packages=['ipfml', 'ipfml/filters', 'ipfml/iqa'],
    install_requires=[
        'numpy',
        'Pillow',
        'sklearn',
        'scikit-image',
        'scipy',
        'opencv-python',
        'scipy',
    ],
    cmdclass={
        'build_py': BuildTestCommand,
    },
    zip_safe=False)
