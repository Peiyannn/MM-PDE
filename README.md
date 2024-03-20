# MM-PDE: Better Neural PDE Solvers Through Data-Free Mesh Movers

[Link to the paper](https://openreview.net/pdf?id=hj9ZuNimRl) (ICLR 2024)

This paper introduces a neural-network-based mesh adapter called **Data-free Mesh Mover (DMM)**, which is trained in a physics-informed data-free way. The DMM can be embedded into the neural PDE solver through proper architectural design, called **MM-PDE**.

<a href="url"><img src="./pics/burgers mesh8.png" align="center" width="700" ></a>

## Environment

Install the environment using [conda](https://docs.conda.io/en/latest/miniconda.html) with attached environment file as follows.

```code
conda env create -f env.yml
```

## Dataset

Download the datasets into the "mesh/data/" folder in the local repo via [this link](https://drive.google.com/drive/folders/1TI2xHsOqAIFNu7EBS6IrkNI7ivZtGXrX?usp=sharing).

## Training of Data-free Mesh Mover (DMM)

- Burgers' equation:
```code
  cd mesh  
  python dmm.py
```
- Flow around a cylinder:
```code
  cd mesh  
  python dmm.py --experiment cy --train_sample_grid 1500 --branch_layers 4,3 --trunk_layers 16,512
```

## Training of MM-PDE

- Burgers' equation:
```code
  python mmpde.py --lr 6e-4
```
- Flow around a cylinder:
```code
  python mmpde.py --experiment cy --base_resolution 30,2521
```

## Training of GNN

- Burgers' equation:
```code
  python mmpde.py --lr 6e-4 --moving_mesh False
```
- Flow around a cylinder:
```code
  python mmpde.py --experiment cy --base_resolution 30,2521 --moving_mesh False
```

## Citation

If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
hu2024better,
title={Better Neural {PDE} Solvers Through Data-Free Mesh Movers},
author={Peiyan Hu and Yue Wang and Zhi-Ming Ma},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```
