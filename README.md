Image Processing For Machine Learning
=====================================

<p align="center">
    <img src="ipfml_logo.png" width="40%">
</p>

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

Please refer to the [CONTRIBUTION.md](https://github.com/jbuisine/IPFML/blob/master/LICENSE) file if you want to contribute!

## Contributors

* [jbuisine](https://github.com/jbuisine)

## Licence

[MIT](https://github.com/jbuisine/IPFML/blob/master/LICENSE)
