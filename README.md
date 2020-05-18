[TensorFlow 2] ResNeSt: Split-Attention Networks
=====

## Related Repositories
<a href="https://github.com/YeongHyeon/ResNet-TF2">ResNet-TF2</a>  
<a href="https://github.com/YeongHyeon/ResNeXt-TF2">ResNeXt-TF2</a>  
<a href="https://github.com/YeongHyeon/WideResNet_WRN-TF2">WideResNet(WRN)-TF2</a>  
<a href="https://github.com/YeongHyeon/ResNet-with-LRWarmUp-TF2">ResNet-with-LRWarmUp-TF2</a>  
<a href="https://github.com/YeongHyeon/ResNet-with-SGDR-TF2">ResNet-with-SGDR-TF2</a>  

<a href="https://github.com/YeongHyeon/SENet-Simple">SE-Net</a>  
<a href="https://github.com/YeongHyeon/SKNet-TF2">SK-Net</a>  

## Concept
<div align="center">
  <img src="./figures/resnest.png" width="600">  
  <p>Comparing SE-Net, SK-Net, and ResNeSt [1].</p>
</div>

## Performance

|Indicator|Value|
|:---|:---:|
|Accuracy|0.98760|
|Precision|0.98754|
|Recall|0.98760|
|F1-Score|0.98756|

```
Confusion Matrix
[[ 974    0    3    0    0    0    2    1    0    0]
 [   0 1125    2    0    1    0    2    2    3    0]
 [   0    1 1012    6    5    0    1    2    5    0]
 [   1    0    2  998    1    3    1    1    3    0]
 [   0    0    0    0  974    0    1    0    3    4]
 [   2    0    0    3    1  881    3    1    1    0]
 [   3    1    1    0    1    2  949    0    1    0]
 [   0    2    2    2    1    0    0 1016    2    3]
 [   5    0    3    2    0    0    2    3  958    1]
 [   0    2    0    1    4    2    1    4    6  989]]
Class-0 | Precision: 0.98883, Recall: 0.99388, F1-Score: 0.99135
Class-1 | Precision: 0.99469, Recall: 0.99119, F1-Score: 0.99294
Class-2 | Precision: 0.98732, Recall: 0.98062, F1-Score: 0.98396
Class-3 | Precision: 0.98617, Recall: 0.98812, F1-Score: 0.98714
Class-4 | Precision: 0.98583, Recall: 0.99185, F1-Score: 0.98883
Class-5 | Precision: 0.99212, Recall: 0.98767, F1-Score: 0.98989
Class-6 | Precision: 0.98649, Recall: 0.99061, F1-Score: 0.98854
Class-7 | Precision: 0.98641, Recall: 0.98833, F1-Score: 0.98737
Class-8 | Precision: 0.97556, Recall: 0.98357, F1-Score: 0.97955
Class-9 | Precision: 0.99198, Recall: 0.98018, F1-Score: 0.98604

Total | Accuracy: 0.98760, Precision: 0.98754, Recall: 0.98760, F1-Score: 0.98756
```

## Requirements
* Python 3.7.6  
* Tensorflow 2.1.0  
* Numpy 1.18.1  
* Matplotlib 3.1.3  

## Reference
[1] Hang Zhang et al. (2020). <a href="https://arxiv.org/abs/2004.08955">ResNeSt: Split-Attention Networks</a>. arXiv preprint arXiv:2004.08955.
