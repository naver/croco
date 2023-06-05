# CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion

[[`arXiv`](https://arxiv.org/abs/2210.10716)] [[`project page and demo`](https://croco.europe.naverlabs.com/)]

This repository contains the code for our CroCo model presented in our [NeurIPS'22 paper](https://openreview.net/pdf?id=wZEfHUM5ri):

```bibtex
@inproceedings{croco,
  title={{CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion}},
  author={{Weinzaepfel, Philippe and Leroy, Vincent and Lucas, Thomas and Br\'egier, Romain and Cabon, Yohann and Arora, Vaibhav and Antsfeld, Leonid and Chidlovskii, Boris and Csurka, Gabriela and Revaud J\'er\^ome}},
  booktitle={{NeurIPS}},
  year={2022}
}
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
Some components are based on code from [MAE](https://github.com/facebookresearch/mae) released under the CC BY-NC-SA 4.0 License and [timm](https://github.com/rwightman/pytorch-image-models) released under the Apache 2.0 License.

## Preparation

1. Install dependencies on a machine with a NVidia GPU using e.g. conda

```bash
conda create -n croco python=3.7 cmake=3.14.0
conda activate croco
conda install habitat-sim headless -c conda-forge -c aihabitat
conda install pytorch torchvision -c pytorch
conda install notebook ipykernel matplotlib
conda install ipywidgets widgetsnbextension
# below is only for pretraining / habitat data generation
conda install scikit-learn tqdm quaternion opencv

```

2. Download pretrained model

```bash
mkdir -p pretrained_models/
wget http://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth -P pretrained_models/
```

3. Download test scene

```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path habitat-sim-data/
```

## Interactive demonstration of cross-view completion reconstruction

Run the Notebook demo `interactive_demo.ipynb`.

In this demo, you should be able to sample a random reference viewpoint from an [Habitat](https://github.com/facebookresearch/habitat-sim) test scene. Use the sliders to change viewpoint and select a masked target view to reconstruct using CroCo.
![croco_interactive_demo](https://user-images.githubusercontent.com/1822210/200516576-7937bc6a-55f8-49ed-8618-3ddf89433ea4.jpg)

## Pre-training 

To pre-train CroCo, please first generate the pre-training data from the Habitat simulator, following the instructions in [datasets/habitat_sim/README.MD](datasets/habitat_sim/README.MD) and then run the following command:
```
torchrun --nproc_per_node=4 pretrain.py --output_dir ./output/pretraining/
```

Our CroCo pre-training was launched on a single server with 4 GPUs.
It should take around 10 days with A100 or 15 days with V100 to do the 400 pre-training epochs, but decent performances are obtained earlier in training.
Note that, while the code contains the same scaling rule of the learning rate as MAE, we did not experimented if it is valid in our case.
The first run can take a few minutes to start, to parse all available pre-training pairs.
