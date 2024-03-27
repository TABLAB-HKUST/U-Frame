# Exceeding the Limit for Microscopic Image Transformation with a Deep Learning-based Unified Framework   
This is a pytorch implementation of U-Frame, proposed in "Exceeding the Limit for Microscopic Image Transformation with a Deep Learning-based Unified Framework". U-Frame is a unified framework that unifies supervised and unsupervised learning for microscopic image transformation, including pseudo optical sectioning, virtual histological staining, improvement of signal-to-noise ratio or resolution, prediction of fluorescent labels, etc. 

![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/4acf45520b442979dc8e6e7e606898ab64d85533/examples/fig%201a%20new.jpg)

## Applications and examples
- ### (1) Image transformations from widefield images to confocal images
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/0386d9651f2dae6de5f369203a4d199fd581a47d/examples/confocal2.jpg)

- ### (2) Style transformation from autofluorescence images to histochemically stained images
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/f93de198959810d6927e9b15699f01e2ca2640cc/examples/virtualstaining.jpg)

- ### (3) Improvement of signal-to-noise ratio
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/11d9a3767a8c5a3c570fec9fee9ab4cd1ec35cb7/examples/planaria.jpg)

- ### (4) Improvement of resolution
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/78105b47d0c83b449407fcb01417c12ae68c198d/examples/sr.jpg)

- ### (5) Prediction of fluorescent labels from brightfield images
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/130ed89fa391d33df40a372d4b6afa242e769174/examples/fluo_rubin.jpg)

- ### (6) Prediction of fluorescent labels from phase contrast images
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/3fc619108cf9ab670ab845792ec6f39cb6becc8f/examples/fluo_yusha.jpg)



##  Datasets 
The datasets for "Exceeding the Limit for Microscopic Image Transformation with a Deep Learning-based Unified Framework" can be downloaded in the following links:
- ### Image transformations from widefield images to confocal images:
  #### Widefield image: https://drive.google.com/file/d/1Y5gq-alqE0LoHLEOsZYI-Py7g6dmGIwB/view?usp=sharing
  #### Single-channel confocal image: https://drive.google.com/file/d/1Fp8mMK_2SBjTktA9p9cbv1rv0NMaCAJh/view?usp=sharing
  #### Multi-channel confocal image: https://drive.google.com/file/d/1vNJ9oIg-trZjjClqOrnvXMZfVi9xqJD6/view?usp=sharing

- ### Style transformation from autofluorescence images to H&E stained images:
  #### Autofluorescence image: https://drive.google.com/file/d/1g-PnQyiRgVKGgoy1ndM9uC14QGDDk5kN/view?usp=sharing
  #### H&E image: https://drive.google.com/file/d/1LWR3eYj6bLX16kTD20yKTEfFQaBDK9fd/view?usp=sharing

- ### Style transformation from autofluorescence images to Masson’s trichome stained images:
  #### Autofluorescence image: https://drive.google.com/file/d/1eVqySKxHAq3t_vG3OnkK3zZ_cgB5BCjB/view?usp=sharing
  #### Masson’s trichome image: https://drive.google.com/file/d/1B_4n_PMBJmjf3MWi4JB__-M-sVnkVnvN/view?usp=sharing


## Test
To test images with a trained model, run as:
```python
python test.py 
```
For example, here are the steps to improve the SNR of planaria images using pre-trained U-Frame model:

First, download the pre-trained model and put it in ./checkpoints/planaria. Then run the following script to get the results:
```python
python test.py --input examples/planaria --ckp planaria --output results/planaria --input_dim 1
```

### Options:
```--input```	Input folder name.

```--ckp```	Model name.

```--output```	Output folder name.

```--input_dim```	Number of input channels.

```--sobel```	Whether use sobel feature.
