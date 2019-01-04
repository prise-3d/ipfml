Image Processing For Machine Learning
=====================================

![ipfml_logo](ipfml_logo.png)


Installation
------------

```bash
pip install ipfml
```

How to use ?
------------

To use, simply do :

```python
from PIL import Image
from ipfml import processing
img = Image.open('path/to/image.png')
s = processing.get_LAB_L_SVD_s(img)
```

Modules
-------

This project contains modules.

- **processing** : *Image processing module*
- **metrics** : *Metrics computation of PIL or 2D numpy image*
- **filters** : *Image filter module*

All these modules will be enhanced during development of the package. Documentation is available [here](https://jbuisine.github.io/IPFML/).

How to contribute
-----------------

This git project uses [git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) implementation. You are free to contribute to it.

## Contributors

* [jbuisine](https://github.com/jbuisine)

## Licence

[MIT](https://github.com/jbuisine/IPFML/blob/master/LICENSE)
