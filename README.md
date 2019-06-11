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
- **utils** : *All utils functions developed for the package*
- **exceptions** : *All customized exceptions*

All these modules will be enhanced during development of the package. Documentation is available [here](https://prise-3d.github.io/IPFML/).

How to contribute
-----------------

Please refer to the [guidelines](https://github.com/prise-3d/IPFML/blob/master/CONTRIBUTING.md) file if you want to contribute!

## Contributors

* [jbuisine](https://github.com/jbuisine)

## License

[MIT](https://github.com/prise-3d/IPFML/blob/master/LICENSE)
