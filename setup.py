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

        print("==============================")
        print("Runs test command...")
        doctest.testmod(processing)
        doctest.testmod(metrics)
        doctest.testmod(noise_filters)

        # Run format code using ypaf
        try:
            print("==============================")
            print("Runs format code command...")
            self.spawn(['yapf', '-ir', '-vv', 'ipfml'])
        except RuntimeError:
            self.warn('format pakcage code failed')

        # Run update auto generated documentation
        try:
            print("==============================")
            print("Runs update of auto generated documentation...")
            self.spawn(['bash', './build.sh'])
        except RuntimeError:
            self.warn('Error during documentation rendering')

        setuptools.command.build_py.build_py.run(self)


setup(
    name='ipfml',
    version='0.2.1',
    description='Image Processing For Machine Learning',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    url='https://gogs.univ-littoral.fr/jerome.buisine/IPFML',
    author='Jérôme BUISINE',
    author_email='jerome.buisine@univ-littoral.fr',
    license='MIT',
    packages=['ipfml'],
    install_requires=[
        'matplotlib',
        'numpy',
        'Pillow',
        'sklearn',
        'scikit-image',
        'scipy',
        'opencv-python',
        'scipy',
        'yapf'
    ],
    cmdclass={
        'build_py': BuildTestCommand,
    },
    zip_safe=False)
