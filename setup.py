from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='IPFML',
      version='0.0.6',
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
          'scipy'
      ],
      zip_safe=False)