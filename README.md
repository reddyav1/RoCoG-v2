# Robot Control Gestures (RoCoG-v2)

A link to download the RoCoG-v2 dataset will be available here soon.

<img src="https://user-images.githubusercontent.com/72093042/194117338-880d9ff2-4c5a-4731-9742-9cb32744f841.gif" width="500" />

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
 
## Dataset Details

### Number of Videos across Splits

| Data Type | View  | Train | Test | Total  |
| :--------:| :---: | :--:  | :---:| :----: |
| Synthetic | Ground| 53,438|   -  | 53,438 |
| Synthetic | Air   | 53,558|   -  | 53,558 |
| Real      | Ground| 204   |  100 | 304  |
| Real      | Air   | 87    |   91 | 178  |
