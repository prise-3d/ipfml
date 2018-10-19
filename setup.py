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
    from ipfml import image_processing
    from ipfml import metrics
    from ipfml import tf_model_helper

    doctest.testmod(image_processing)
    doctest.testmod(metrics)
    doctest.testmod(tf_model_helper)

    setuptools.command.build_py.build_py.run(self)

setup(name='IPFML',
      version='0.0.9',
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
      ],
      cmdclass={
        'build_py': BuildTestCommand,
      },
      zip_safe=False)