# pytorch_model_prune
Simple python code to prune pytorch model, by pruning each Conv2d layer in the model  

![image](https://github.com/siyuan0/pytorch_model_prune/blob/master/example.png)

This code is based on ideas from the paper 'Pruning Filters for Efficient ConvNets' by Hao Li, et al (https://arvix.org/abs/1608.08710). The code searches through a given pytorch model and prunes Conv2d layers by evaluating the weights of each channel and removing channels whose weights are close to zero. After that, an additional layer of zero_padding is added to ensure that the output tensor is of the correct dimensions.  

## Usage
Import prune.py into your own pytorch code, and add a line `model = prune_model(model, factor_removed=[PROPORTION OF CHANNELS YOU WANT TO REMOVE])` before you run your test. This pruning is to be called only after you have trained your model. The idea is to prune a trained model's parameters while maintaining its accuracy.

## Results
I have tested this on a SSD object detection model with MobilenetV2 as its base. The model was trained/tested on a custom dataset, which cannot be made available. However, the results below will give a picture of the pruning's effects. It is seen that mAP and fps is maintained for pruning of a large portion of the parameters.  
  
| factor_removed | parameter count | mAP | fps |
|:-:|:-:|:-:|:-:|
| baseline - 0.0 | 3,333,120 | 0.500 | 16.62 |
| 0.3 | 3,238,764 | 0.502 | 16.39 |
| 0.4 | 2,945,836 | 0.505 | 16.19 |
| 0.5 | 2,410,124 | 0.501 | 16.17 |
| 0.6 | 1,596,297 | 0.502 | 16.32 |
| 0.7 | 1,569,638 | 0.500 | 16.04 |
| 0.75 |1,536,326 | 0.497 | 14.48 |
| 0.8 | 1,453,190 | 0.333 | 7.67 |
| 0.825 | 1,396,832 | 0.025 | 7.15 |
| 0.9 | 1,234,942 | 0.001 | 7.45 |


