# NFGNN
:triangular_flag_on_post:
:triangular_flag_on_post:
:triangular_flag_on_post:
This is the Pytorch implementation of NFGNN (Node-oriented Spectral Filtering for Graph Neural Networks)!  

# Methodology
:heart:*NFGNN_large* is corresponding to the scalable variant of NFGNN introduced in Sect.4.3.

:heart:*NFGNN* is the standard version that includes the settings of transductive and inductive node classification.

# Package dependencies
The project is built with Python3.6, CUDA11.2, NVIDIA GeForce RTX 3090. For package dependencies, you can install them by:
```
pip install -r requirements.txt
```

# To start
Plz put the downloaded data or your own data in the folder below:
```
|-- root
    |-- NFGNN
    |-- NFGNN_large
    |-- data
    |   |-- your own data
```
 You can run the command to obtain the result of NFGNN:
```
python ./NFGNN/train_model.py --dataset $dataset_name
```

:point_right: (Optional) You can also run the code with preset hyperparameters:
```
python ./NFGNN/reproduce.sh
```

:point_right: (Optional) For your own dataset, you can run the command below to search for the best combination of hyperparameters:
```
python ./NFGNN/meta/meta.py
```
