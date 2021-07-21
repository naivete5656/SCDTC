# Semi-supervised Cell Detection in Time-lapseImages Using Temporal Consistency
by Kazuya Nishimura, Hyeonwoo Cho, Ryoma Bise


[[Home]](https://human.ait.kyushu-u.ac.jp/~nishimura/index-e.html) [[Paper]](https://arxiv.org/abs/2107.08639) 

## Prerequisites
- [python >= 3.6](https://www.python.org)
- [ubuntu 18.04](https://ubuntu.com/)
- CPU or GPU(NVIDIA Driver >= 430)

## Installation

Python setting
### Conda user
```bash
conda env create -f=requirement.yml
conda activate pytorch
```

## Dataset
Cell tracking challnege (http://celltrackingchallenge.net/)
Please download and extract DIC-C2DH-HeLa and PhC-C2DH-U373 and PhC-C2DL-PSC to ./image/CTC/

C2C12 (https://osf.io/ysaq2/)


### Docker user
```besh
sh ./docker/build.sh
sh ./docker/run.sh
```


## Training
```bash
python main.py -g
```

## Test
Coming soon
