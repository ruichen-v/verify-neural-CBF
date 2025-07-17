# Python Implementation
This is Python code for CoRL 2024 work "Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation". Currently, Python implementation supports data collection, model training and visualization.

## Preparation
The code has been tested with Python 3.10. Make sure `torch matplotlib scipy tqdm` packages are correctly installed.

## Data Collection
To collect data for Dubins Car, run `python collect_data.py`. Download the pre-collected data from [here](https://drive.google.com/file/d/1VmzQTZ9Zg0t0QdkLXlKXg-IVd60BPuZw/view?usp=sharing) and unzip it under the current folder path.

## Model training 
For the regular model training under Dubins Car, run `python train_ncbf.py`. Change the `.pkl` data paths based on the collected dataset above. Check out the loss curves `loss_curves.png` after training. The final `.pt` neural CBF models are saved in the root path. Feel free to change any training hyper-parameters to see different training results. The pre-trained model `car_naive_model_1_0_0.1_pgd_relu_20.pt` and the training losses curves can be found  in the current folder path.

## Visualization
After the model is trained, run `python visualize_cbf.py` to find the value of CBF in the environment. Run `python visualize_derivative.py` to visualize the derivative of neural CBF based on fixed control input and best-case control input (the optimal control input such that the neural CBF is minimized within the input limit). Note that even though the verification code is currently not implemented, you can check out the best-case CBF derivative heat map to see if the well-trained neural CBF works or not. The visualization of pre-trained model `car_naive_model_1_0_0.1_pgd_relu_20.pt` can be found in the current folder path.


## Citation 
If you find the repo useful, please cite:

H. Hu, Y. Yang, T. Wei and C. Liu
"[Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation](https://openreview.net/forum?id=jnubz7wB2w)", Conference on Robot Learning (CoRL). PMLR, 2024
```
@inproceedings{
hu2024verification,
title={Verification of Neural Control Barrier Functions with Symbolic Derivative Bounds Propagation},
author={Hanjiang Hu and Yujie Yang and Tianhao Wei and Changliu Liu},
booktitle={8th Annual Conference on Robot Learning},
year={2024},
url={https://openreview.net/forum?id=jnubz7wB2w}
}
```


