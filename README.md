# Weakly-supervised deep generative network for complex image restoration and style transformation (WeCREST)
WeCREST is a weakly supervised framework for complex image transfomation including fluorescence image restoration with low signal-to-noise ratio, image resolution recovery, virtual histological staining, etc. This is a pytorch implementation of WeCREST. 

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
### Options:
```--input```	Input folder name.

```--ckp```	Model name.

```--output```	Output folder name.

```--input_dim```	Number of input channels.

```--sobel```	Whether use sobel feature.

## Examples
### Low-SNR restoration of planaria
In this example, we use a pre-trained model for low-SNR restoration of planaria. The pre-trained model can be found in ./checkpoints/planaria. Run the following script:
```python
python test.py --input examples/planaria --ckp planaria --output results/planaria --input_dim 1
```
The results will be shown in the specified folder.                
![alt text](https://github.com/TABLAB-HKUST/WeCREST/blob/bb34aa78773ad8c65ea3c415cca6347ade56e65e/examples/planaria/input.png)
![alt text](https://github.com/weixingdai/CMIT/blob/94d5b8cc787bdf0c04446403c4d1c1f6f71c36cd/results/planaria/fake_cmit_input.png)
![alt text](https://github.com/TABLAB-HKUST/WeCREST/blob/c60dd08b52062b885b0190bef5133f7b473ce18f/examples/planaria%20ground%20truth/gt.png)

Left: low-SNR image of planaria. Middle: image restored by WeCREST.  Right:  ground truth.

