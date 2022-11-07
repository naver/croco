# CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion

This repository contains the code for running our CroCo model presented in our [NeurIPS'22 paper](https://openreview.net/pdf?id=wZEfHUM5ri):

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

Run the Notebook demo `interactive_demo.ipynb`