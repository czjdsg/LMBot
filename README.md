# LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection (WSDM 2024)
Official implementation of [LMBot: Distilling Graph Knowledge into Language Model for Graph-less Deployment in Twitter Bot Detection](https://arxiv.org/abs/2306.17408)

## Requirements
Run following command to create environment for reproduction (for cuda 10.2):
```
conda env create -f lmbot.yaml
conda activate lmbot
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
```
For ```pyg_lib```, ```torch_cluster```, ```torch_scatter```, ```torch_sparse``` and ```torch_spline_conv```, please download [here](https://data.pyg.org/whl/torch-1.12.0%2Bcu102.html) and install locally.
```
pip install pyg_lib-0.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_cluster-1.6.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_scatter-2.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_sparse-0.6.16+pt112cu102-cp39-cp39-linux_x86_64.whl torch_spline_conv-1.2.1+pt112cu102-cp39-cp39-linux_x86_64.whl
```
## Data preperation
Please download our preprocessed datasets [here](https://drive.google.com/drive/folders/1kbI3uJQCn3e8CN3d9iUeUNSIOuJCbDUj?usp=sharing) and put it in the ```datasets``` folder.

## Training
Run the following commands to train on ```Twibot-20```:
```
main.py --project_name lmbot --experiment_name TwiBot-20 --dataset Twibot-20 --device 0 --LM_pretrain_epochs 4.5 --alpha 0.5 --max_iter 10 --batch_size_LM 32 --use_GNN
```


