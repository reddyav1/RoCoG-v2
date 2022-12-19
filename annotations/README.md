# Splits by Experiment
Here, we provide the precise listing of videos used in each experimental setting.

## Source Only
### 	G<sub>S</sub> &#8594; G<sub>R</sub>
Train: `syn_ground_train.txt`
Validation: `syn_ground_val.txt`
Test: `real_ground_test.txt`

### 	A<sub>S</sub> &#8594; A<sub>R</sub>
Train: `syn_air_train.txt`
Validation: `syn_air_val.txt`
Test: `real_air_test.txt`

### 	G<sub>R</sub> &#8594; A<sub>R</sub>
Train: `real_ground_train_tsonly.txt`
Validation: `real_ground_val_tsonly.txt`
Test: `real_air_test.txt`

### 	G<sub>S</sub> &#8594; A<sub>R</sub>
Train: `syn_ground_train.txt`
Validation: `syn_ground_val.txt`
Test: `real_air_test.txt`

## Target Only

### 	G<sub>S</sub> &#8594; G<sub>R</sub>
Train: `real_ground_train_tsonly.txt`
Validation: `real_ground_val_tsonly.txt`
Test: `real_ground_test.txt`

### A<sub>S</sub> &#8594; A<sub>R</sub>, G<sub>R</sub> &#8594; A<sub>R</sub>, G<sub>S</sub> &#8594; A<sub>R</sub>

Train: `real_air_train_tsonly.txt`
Validation: `real_air_val_tsonly.txt`
Test: `real_air_test.txt`

## Domain Adaptation
### 	G<sub>S</sub> &#8594; G<sub>R</sub>
Train: `syn_ground_train.txt` + `real_ground_train.txt`
Validation: `syn_ground_val.txt`
Test: `real_ground_test.txt`

### 	A<sub>S</sub> &#8594; A<sub>R</sub>
Train: `syn_air_train.txt` + `real_air_train.txt`
Validation: `syn_air_val.txt`
Test: `real_air_test.txt`

### 	G<sub>R</sub> &#8594; A<sub>R</sub>
Train: `real_ground_train_tsonly.txt` + `real_air_train.txt`
Validation: `real_ground_val_tsonly.txt`
Test: `real_air_test.txt`

### 	G<sub>S</sub> &#8594; A<sub>R</sub>
Train: `syn_ground_train.txt` + `real_air_train.txt`
Validation: `syn_ground_val.txt`
Test: `real_air_test.txt`
