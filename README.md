Image Processing For Machine Learning
=====================================

![](https://img.shields.io/github/workflow/status/prise-3d/ipfml/build?style=flat-square) ![](https://img.shields.io/pypi/v/ipfml?style=flat-square) ![](https://img.shields.io/pypi/dm/ipfml?style=flat-square)

<p align="center">
    <img src="https://github.com/prise-3d/ipfml/blob/master/ipfml_logo.png" alt="" width="40%">
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
from ipfml.processing import transform
img = Image.open('path/to/image.png')
s = transform.get_LAB_L_SVD_s(img)
```

Modules
-------

This project contains modules.

- **metrics** : *Metrics computation for model performance*
- **utils** : *All utils functions developed for the package*
- **exceptions** : *All customized exceptions*
- **filters** : *Image filter module with convolution*
- **iqa** : *Image quality assessments*
- **processing** : *Image processing module*

All these modules will be enhanced during development of the package. Documentation is available [here](https://prise-3d.github.io/ipfml/).

How to contribute
-----------------

Please refer to the [guidelines](CONTRIBUTING.md) file if you want to contribute!

## Contributors

* [jbuisine](https://github.com/jbuisine)

## License

[MIT](LICENSE)
