# pytorch_model_prune
Simple python code to prune pytorch model, by pruning each Conv2d layer in the model  

![image](https://github.com/siyuan0/pytorch_model_prune/blob/master/example.png)

This code is based on ideas from the paper 'Pruning Filters for Efficient ConvNets' by Hao Li, et al (https://arxiv.org/abs/1608.08710). The code searches through a given pytorch model and prunes Conv2d layers by evaluating the weights of each channel and removing channels whose weights are close to zero. After that, an additional layer of zero_padding is added to ensure that the output tensor is of the correct dimensions.  

## Usage
Import prune.py into your own pytorch code, and add a line `model = prune_model(model, factor_removed=[SOME PROPORTION OF CHANNELS YOU WANT TO REMOVE])` before you run your test. This pruning is to be called only after you have trained your model. The idea is to prune a trained model's parameters while maintaining its accuracy.

## Results
I have tested this on an SSD object detection model with MobilenetV2 as its base. The model was trained/tested on a custom dataset, which cannot be made available. However, the results below will give a picture of the pruning's effects. It is seen that mAP and fps is maintained for pruning of a large portion of the parameters.  

The reason why the fps does not drop even when half the parameters are removed is due to the way the output of the neural network is processed in an SSD object detection model. SSD object detection models neural networks typically output a large number of possible bounding boxes, which takes significant amount of time for the programme to process and pick out the most relevant ones. The fact that the fps remains the same even after pruning large portion of parameters, and only starts to fall when the mAP falls, indicates that the neural network retain its capabilities from before pruning.
  
| parameter count | mAP | fps |
|:-:|:-:|:-:|
| 3,333,120 | 0.500 | 16.62 |
| 3,238,764 | 0.502 | 16.39 |
| 2,945,836 | 0.505 | 16.19 |
| 2,410,124 | 0.501 | 16.17 |
| 1,596,297 | 0.502 | 16.32 |
| 1,569,638 | 0.500 | 16.04 |
| 1,536,326 | 0.497 | 14.48 |
| 1,453,190 | 0.333 | 7.67 |
| 1,396,832 | 0.025 | 7.15 |
| 1,234,942 | 0.001 | 7.45 |


