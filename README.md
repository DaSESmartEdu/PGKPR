the code of 
Multi-task Learning for Paraphrase Generation With Keyword and Part-of-Speech Reconstruction


### 1. Data Preprocess

tokenize data, get vocab and split train dev

Generate stage one data for mscoco 

**run data_process.py with correct dataset path**

```shell
python data_process.py
```



### 2. Stage One

#### 2.1 pretrian bert with is_paraphrase task

run stage_one_fine_tune.py

make sure the MODE_MSCOCO and path inforation is correct

```
MODE_MSCOCO = True  # if quora, set False
task_name = 'mscoco_cls'
data_path = ".data/mscoco/is_paraphrase/"
log_path = '.runs/mscoco/cls_runs/' + task_name
model_save_path = 'model_save/cls_model_save/' + task_name
```

#### 2.2 Identify Key words possibility 

run 

stage_one_keywords_identify.py

Make sure

```python
model_weight_path = '..'
in_path = '..'
out_path = '..'
```



### 3. Stage Two

#### 3.1 get key words by possibility 

run stage_two_data.py with correct dataset path

Make sure data path

```shell
python stage_two_data.py
```

#### 3.2 Train and Predict

for quora

```shell
python stage_two_quora.py
```

for mscoco

```python
python stage_two_mscoco.py
```

 

