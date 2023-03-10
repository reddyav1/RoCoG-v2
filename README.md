# Robot Control Gestures (RoCoG-v2)

This repository provides access to the RoCoG-v2 gesture recognition dataset introduced in the ICRA 2023 paper "Synthetic-to-Real Domain Adaptation for Action Recognition: A Dataset and Baseline Performances." 

RoCoG-v2 (Robot Control Gestures) is a dataset intended to support the study of synthetic-to-real and ground-to-air video domain adaptation. It contains over 100K synthetically-generated videos of human avatars performing gestures from seven (7) classes. It also provides videos of real humans performing the same gestures from both ground and air perspectives.

<img src="https://user-images.githubusercontent.com/72093042/194117338-880d9ff2-4c5a-4731-9742-9cb32744f841.gif" width="500" />

## Downloading the Dataset

All of the data for RoCoG-v2 can be found [here](https://www.cis.jhu.edu/~rocog/data/). Each data type is provided in a separate zip file.

You may download the data through the browser or using the following command:

```
wget https://www.cis.jhu.edu/~rocog/data/<FILENAME>
```

Replace \<FILENAME\> in the above command with a filename from the table below corresponding to the data type you wish to download.

| Filename | Description  |
| :--------:| :---: |
| syn_ground.zip | Synthetic videos rendered from the ground perspective |
| syn_air.zip | Synthetic videos rendered from the air perspective (static hover)   |
| syn_orbital.zip*      | Synthetic videos rendered from the air perspective (orbiting the subject) |
| real_ground.zip      | Real cropped videos collected from the ground perspective   |
| real_air.zip      | Real cropped videos collected from the air perspective   |
| real_uncropped.zip      | Uncropped versions of the real ground and air videos   |

\* Our paper does not provide experimental results with this data type.

## Dataset Details

### Number of Videos across Splits

| Data Type | View  | Train | Test | Total  |
| :--------:| :---: | :--:  | :---:| :----: |
| Synthetic | Ground| 53,438|   -  | 53,438 |
| Synthetic | Air   | 53,558|   -  | 53,558 |
| Real      | Ground| 204   |  100 | 304  |
| Real      | Air   | 87    |   91 | 178  |

## Dataset Directory Structure
```bash
.
├── annotations
├── real
│   ├── air
│   │   ├── Advance
│   │   ├── Attention
│   │   ├── FollowMe
│   │   ├── Halt
│   │   ├── MoveForward
│   │   ├── MoveInReverse
│   │   └── Rally
│   └── ground
│       ├── Advance
│       ├── Attention
│       ├── FollowMe
│       ├── Halt
│       ├── MoveForward
│       ├── MoveInReverse
│       └── Rally
└── syn
    ├── air
    │   ├── Advance
    │   ├── Attention
    │   ├── FollowMe
    │   ├── Halt
    │   ├── MoveForward
    │   ├── MoveInReverse
    │   └── Rally
    └── ground
        ├── Advance
        ├── Attention
        ├── FollowMe
        ├── Halt
        ├── MoveForward
        ├── MoveInReverse
        └── Rally
 ```


# Citation

If you use this dataset in your work, please cite our ICRA 2023 paper:

bibtex
```
@inproceedings{2023rocogv2,
  title={Synthetic-to-Real Domain Adaptation for Action Recognition: A Dataset and Baseline Performances},
  author={Reddy, Arun V and Shah, Ketul and Paul, William and Mocharla, Rohita and Hoffman, Judy and Katyal, Kapil D and Manocha, Dinesh and de Melo, Celso M and Chellappa, Rama},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={},
  year={2023},
  organization={IEEE}
}
```
APA
```
Reddy, A. V., Shah, K., Paul, W., Mocharla, R., Hoffman, J., Katyal, K. D., Manocha, D., de Melo, C. M., & Chellappa, R. (2023). Synthetic-to-Real Domain Adaptation for Action Recognition: A Dataset and Baseline Performances. IEEE International Conference on Robotics and Automation (ICRA).
```
