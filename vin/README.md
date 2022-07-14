# VIN: [Value Iteration Networks](https://arxiv.org/abs/1602.02867)
## Installation
```
pip install -U -r requirements.txt 
```

## Datasets
Each data sample consists of an obstacle image and a goal image followed by the (x, y) coordinates of current state in the gridworld. 

Dataset size | 8x8 | 16x16 | 28x28
-- | -- | -- | --
Train set | 81337 | 456309 | 1529584
Test set | 13846 | 77203 | 251755

### Make the datasets
```
make_datasets.py
```

## How to train
### 28x28 gridworld
```bash
python train.py --datafile dataset/gridworld_28x28.npz --imsize 28 --lr 0.002 --epochs 30 --k 36 --batch_size 128
```
**Flags**: 
- `datafile`: The path to the data files.
- `imsize`: The size of input images. One of: [8, 16, 28]
- `lr`: Learning rate with RMSProp optimizer. Recommended: [0.01, 0.005, 0.002, 0.001]
- `epochs`: Number of epochs to train. Default: 30
- `k`: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
- `l_i`: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
- `l_h`: Number of channels in first convolutional layer. Default: 150, described in paper.
- `l_q`: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.
- `batch_size`: Batch size. Default: 128

## How to test / visualize paths
#### 28x28 gridworld
```bash
python test.py --weights trained/vin_28x28.pth --imsize 28 --k 36
```
To visualize the optimal and predicted paths simply pass:
```bash 
--plot
```

**Flags**: 
- `weights`: Path to trained weights.
- `imsize`: The size of input images. One of: [8, 16, 28]
- `plot`: If supplied, the optimal and predicted paths will be plotted 
- `k`: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
- `l_i`: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
- `l_h`: Number of channels in first convolutional layer. Default: 150, described in paper.
- `l_q`: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.

## Results
Gridworld | Sample One | Sample Two
-- | --- | ---
28x28 | <img src="results/28x28_1.png" width="450"> | <img src="results/28x28_2.png" width="450">
