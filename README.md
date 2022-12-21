# Exceeding the Limit for Microscopic Image Transformation with a Deep Learning-based Unified Framework   
This is a pytorch implementation of U-Frame, proposed in "Exceeding the Limit for Microscopic Image Transformation with a Deep Learning-based Unified Framework". U-Frame is a unified framework that unifies supervised and unsupervised learning for microscopic image transformation, including pseudo optical sectioning, virtual histological staining, improvement of signal-to-noise ratio or resolution, prediction of fluorescent labels, etc. 

![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/5716133a2923db79e19440841a843c5189156253/examples/fig%201.jpg)

## Applications and examples
### 1. Image transformations from widefield images to confocal images:
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/75238f78834a46b836d77428cc71552e65fd23e9/examples/confocal.jpg)
The pre-trained model can be downloaded here : [link](https://drive.google.com/file/d/13fiLkuUmXAznJ76H5ZcOGBYWBOC4Aplq/view?usp=share_link)

### 2. Style transformation from autofluorescence images to histochemically stained images:
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/f93de198959810d6927e9b15699f01e2ca2640cc/examples/virtualstaining.jpg)
The pre-trained model can be downloaded here : [link](https://drive.google.com/file/d/1ENfFuSBBl2yndMYy5MRXZwdloIK4qmNB/view?usp=sharing)

### 3. Improvement of signal-to-noise ratio:
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/11d9a3767a8c5a3c570fec9fee9ab4cd1ec35cb7/examples/planaria.jpg)
The pre-trained model can be downloaded here : [link](https://drive.google.com/file/d/1Kwo4E980RCSC4HzDDUl0l3NrBay6QLeE/view?usp=sharing)

### 4. Prediction of fluorescent labels from brightfield images:
![alt text](https://github.com/TABLAB-HKUST/U-Frame/blob/130ed89fa391d33df40a372d4b6afa242e769174/examples/fluo_rubin.jpg)
The pre-trained model can be downloaded here : [link](https://drive.google.com/file/d/1D1adNkxteuwf7tXHwQN_BMFrnQzgqjJT/view?usp=sharing)

### 4. Prediction of fluorescent labels from phase contrast images:
![alt text]()
The pre-trained model can be downloaded here : [link](https://drive.google.com/file/d/10cWMDy-sMWBeJHTV1FPg6Fx6jzikFeaO/view?usp=sharing)

### 5. Super-resolution:
![alt text]()
The pre-trained model can be downloaded here : [link]()

##  Prepare the dataset for training
To prepare the dataset from the raw data, run the following script:
```python
python dataset_prepare.py 
```
### Options:
```--source```	Path of source image.

```--target```	Path of target image.

```--output```	Path of output folder.

```--tol```	Tolerance size, i.e. the size of exculsive regions.


##  Train a model
To train a model with prepared dataset, run as:
```python
python train.py 
```

### Options:
```--ckp```	Model name to be saved in ./checkpoints.

```--dataroot```	The folder of dataset for training.

```--tol```	Tolerance size, i.e. the size of exculsive regions.

```--overlap_size```	Overlapping size among the exculsive regions.

```--batch_size```	Batch size.

```--lr```	Initial learning rate.

```--n_epochs```	Number of epochs for keeping the initial learning rate.

```--n_epochs_decay```	Number of epochs for decaying learning rate.

```--input_size```	Input size of the network.

```--input_dim```	Number of input channels.

```--iter```	Number of iterations per epoch.

```--sobel```	Whether use sobel feature.

```--save_mode```	Save mode, simple | full.

```--tv_dir```	The folder for displaying the results during training.

## Test a model
To test images with a trained model, run as:
```python
python test.py 
```
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
