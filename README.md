# Weakly-supervised deep generative network for complex image restoration and style transformation (WeCREST)
WeCREST is a weakly supervised framework for complex image transfomation including fluorescence image restoration with low signal-to-noise ratio, image resolution recovery, virtual histological staining, etc. This is a pytorch implementation of WeCREST. 

##  Train a model
```python
python train.py 
```

### Options:
```--ckp```	Model name to be saved in ./checkpoints

```--dataroot```	The folder of dataset for training

```--tol```	Tolerance size, i.e. the size of exculsive regions, default: 256

```--overlap_size```	Overlapping size among the exculsive regions, default: 64

```--batch_size```	Batch size, default: 2

```--lr```	Initial learning rate, default: 0.0002

```--n_epochs```	Number of epochs for keeping the initial learning rate, default: 40

```--n_epochs_decay```	Number of epochs for decaying learning rate, default: 40

```--input_size```	Input size of the network, default: 256

```--input_dim```	Number of input channels, default: 3

```--iter```	Number of iterations per epoch, default: 2000

```--sobel```	Whether use sobel feature, default: True

```--save_mode```	Save mode, simple | full, default: simple

```--tv_dir```	The folder for displaying the results during training

## Test a model
```python
python test.py 
```
### Options:
```--input```	Input folder name

```--ckp```	Model name

```--output```	Output folder name

```--input_dim```	Number of input channels, default: 3

```--sobel```	Whether use sobel feature

## Examples
### Low-SNR restoration of planaria
```python
python test.py --input examples/planaria --ckp planaria --output results/planaria --input_dim 1
```
![alt text](https://github.com/weixingdai/CMIT/blob/ab8537d8ab55bf15d2b12e0df6b199901a4c915b/examples/planaria/input.png)
![alt text](https://github.com/weixingdai/CMIT/blob/94d5b8cc787bdf0c04446403c4d1c1f6f71c36cd/results/planaria/fake_cmit_input.png)
![alt text](https://github.com/weixingdai/CMIT/blob/f58098440e8f84e6970808851e2c76fc53b07e06/examples/planaria%20ground%20truth/gt.png)

Left: low-SNR image of planaria. Middle: image restored by CMIT.  Right:  ground truth.

