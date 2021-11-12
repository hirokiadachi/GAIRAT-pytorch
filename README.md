# GAIRAT-pytorch
This repo. is the implementation of "GEOMETRY-AWARE INSTANCE-REWEIGHTED ADVERSARIAL TRAINING" proposed by J. Zhang et al. on ICLR 2021.<br>
paper link: https://arxiv.org/abs/2010.01736<br>
official code: https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training


## Implementation results
* CIFAR-10
### WideResNet34-10
||Clean|PGD20 (non-restart)|PGD20 (5restart)|
|:---:|:---:|:---:|:---:|
|Best|83.69|54.98|55.00|
|Last|84.07|52.33|52.36|


### ResNet18
||Clean|PGD20 (non-restart)|PGD20 (5restart)|
|:---:|:---:|:---:|:---:|
|Best|66.82|53.69|53.73|
|Last|65.07|52.05|52.14|
