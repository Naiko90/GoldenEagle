# GoldenEagle

> GoldenEagle is a Python application to compare two images based on different analysis techniques. 

## Supported analysis

### COLOUR

Execute a pixel-by-pixel difference between the two images. There are three different comparison modes that can be applied: 
- **RGB**: The alpha channel is ignored (if present)
- **RGBA**: All image channels are used for the comparison. In case one of the source image doesn't have an alpha channel, the comparison mode will be defaulted to RGB. To avoid this behaviour, the user can use the flag *fillalpha* so as to add an alpha channel to the image that doesn't have one 
- **A**: Only the alpha channel is used for the comparison. This mode can be used only if both images are RGBA

The output of this analysis is a RGBA image where each pixel will be assigned a colour whose meaning will vary. 

By default, this is the formula used to compute the difference between two pixels:

![LinearScale](https://github.com/Naiko90/images/blob/master/linear_scale.png?raw=true)

The user has also the possibility to use a *logarithmic scale* based on this other formula:

![LogScale](https://github.com/Naiko90/images/blob/master/log_scale.png?raw=true)

If we consider *maxDiff* as the result of the formula that the user has decided to use, for each pixel in the output image we can have one of the following colours:

| Colour (RGBA) | LinearScale | LogScale |
| --- | --- | --- |
| (0, 0, 0, 255) | maxDiff == 0 | maxDiff < 0 |
| (0, 0, 255, 255) | maxDiff == 1 | maxDiff == 2 or maxDiff == 3 |
| (255, 255, 0, 255) | maxDiff == 2 | maxDiff >=4 and maxDiff < 8 |
| (255, 165, 0, 255) | maxDiff == 3 or maxDiff == 4 | maxDiff >=8 and maxDiff < 16 |
| (255, 0, 0, 255) | maxDiff > 4 | maxDiff >= 16 |

### SSIM

Compute the Structural Similarity Index. For more information, please check [here](https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html).

### HAARPSI

Compute the Haar wavelet-based perceptual similarity index. For more information, please check [here](http://www.haarpsi.org/)

### BRISK

Apply to each image the BRISK keypoint detector and descriptor extractor. The keypoints are then matched using a Brute-force descriptor matcher. 

By default, the output image will show all the keypoints that were matched in the two images. It is also possible to see the keypoints not matched by using the flag *mismatched*.

For more information, please check [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

## Prerequisites

In order to use the application, you need to install [Python (3.X version)](https://www.python.org/downloads/) along with the following modules:

- [argparse](https://pypi.org/project/argparse/)
- [psutil](https://pypi.org/project/psutil/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [scikit-image](https://scikit-image.org/docs/0.15.x/install.html)

The application also uses two other modules whose source code can be found in the src folder:

- [HaarPSI](http://www.haarpsi.org/software/haarPsi.py)
- [MplInteraction](https://gist.github.com/t20100/e5a9ba1196101e618883)

## Examples

1. Colour Analysis, using RGB mode and log scale
    python .\compare.py -l ..\img\bridge.jpg -r ..\img\bridge_modified.jpg --verbose colour -m rgb --logdiff

2. HaarPSI
    python .\compare.py -l ..\img\bridge.jpg -r ..\img\bridge_modified.jpg --verbose haarpsi

3. BRISK, with mismatched keypoints in the output image
    python .\compare.py -l ..\img\joker.png -r ..\img\joker_modified.png --verbose brisk --mismatched


## License
[MIT](https://choosealicense.com/licenses/mit/)
