# There-Are-Many-Consistent-Explanations-of-Unlabeled-Data-Why-You-Should-Average
### By Boris Sobol & Ariel Amsel
***Only Fast-swa was implemented***

## How to Run:
1. Create a folder named Datasets and fill with directories of datasets to run. Each directory must be in pytorch image loader format https://pytorch.org/docs/stable/data.html
2. To run original Fast-SWA model from paper: python ModelTraining.py 0 - this will run hyper parameter search and train on all datasets
3. To run our improvement with augemented images and multiple students: python ModelTraining.py 1 - this will run hyper parameter search and train on all datasets
4. To run AlexNet - python pretrained_model.py


## Hyper parameters Choosen:
"optimizer_args": {
  "lr": 0.05,
  "momentum": 0.9,
  "weight_decay": 2e-4,
  "nesterov": True,
},
"epoch_args": 1,  # 180,
"batch_size": 128,
"labeled_batch_size_ratio": 0.25,
"lr_rampdown_epochs": 210,
"lr_rampdown_steps": 1800,
"constant_lr_epoch": 0,
"mt_distance_cost": 0.01,
"cycle_interval": 5,  # 30,
"num_cycles": 20,  # 100,
"logit_distance_cost": 0.01,
'ema_decay': 0.97,
"consistency_rampup": 5,
"consistency": 100.0,
'fastswa_freq': '3'

## References
1. Athiwaratkun, B., Finzi, M., Izmailov, P., & Wilson, A. G. (2018). There are many consistent explanations of unlabeled data: Why you should average. arXiv preprint arXiv:1806.05594.  github: https://github.com/benathi/fastswa-semi-sup.
2. Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. arXiv preprint arXiv:1703.01780.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
4. Gastaldi, X. (2017). Shake-shake regularization. arXiv preprint arXiv:1705.07485.
5. Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 702-703).
6. Krizhevsky, A., Sutskever, I., and Hinton, G. E. ImageNet classification 
with deep convolutional neural networks. In NIPS, pp. 1106–1114, 2012
