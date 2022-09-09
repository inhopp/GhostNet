# Ghostnet_pytorch

> [Paper_Review](https://inhopp.github.io/paper/Paper8/)

<br>

## Repository Directory 

``` python 
├── GhostNet_pytorch
        ├── datasets
        │     └── rsp_data
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── hubconf.py
        ├── train.py
        ├── inference.py
        └── README.md
```


<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/GhostNet_pytorch.git
pip3 install requirements.txt
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --data_name {}(default: rsp_data) \
    --lr {}(default: 0.0005) \
    --n_epoch {}(default: 10) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 32) \ 
    --eval_batch_size {}(default: 16)
```

### testset inference
```
python3 inference.py
    --device {}(defautl: cpu) \
    --data_name {}(default: rsp_data) \
    --num_workers {}(default: 4) \
    --eval_batch_size {}(default: 16)
```