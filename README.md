# Auto Speech Tech Project2

## Baseline

1. `mkdir features` in the baseline dir
2. change the data dir `pathToDatabase` to yourself in `baseline_cqcc.m` and `baseline_mfcc.m`
3. run `baseline_cqcc.m` like `/matlab_dir/matlab -nodisplay -nodesktop -nosplash -r baseline_cqcc`, then do same with the
  `baseline_mfcc.m`
4. u will get `cqcc` and `mfcc` features in dir `features`

## NNET
in the `nnet` dir, we use some deep learning algorithm to solve this problem



## Result
> the column 3 and 4 only use train data
> the column 5 use train and dev data to test the eval

### BaseLine

|  system  | feature | EER(Dev) | EER(Eval) | EER(Eval) | Frequency Range | B |       Remarks        |
| :------: | :-----: | :------: | :-------: | :-------: | :-------: | :----: | :------------------: |
| GMM | cqcc | 10.35 | 30.60 | 24.77 | 16-8000     | 96  |    Baseline !!!   |

### GMM Approach

> only one feature use in the gmm

|  system  | feature | EER(Dev) | EER(Eval) | EER(Eval) | Frequency Range | B | iter(default 100) |
| :------: | :-----: | :------: | :-------: | :-------: | :-------: | :----: | :------------------: |
| GMM | mfcc | 14.14 | 33.08 |       | 16-8000     | 256 | |
| GMM | mfcc | 36.03 | 36.17 |       | 16-2000     | 256 | |
| GMM | mfcc | 38.60 | 37.32 |       | 2000-4000   | 256 | |
| GMM | mfcc | 6.86  | 27.60 |       | 4000-8000   | 256 | |
| GMM | mfcc | **3.53**  | 25.55 |   | 6000-8000   | 256 | |
| GMM | cqcc | 13.44 | 28.50 |       | 16-8000 | 256 | |
| GMM | cqcc | 40.45 | 37.98 |       | 16-2000 | 256 | |
| GMM | cqcc | 42.04 | 39.59 |       | 2000-4000 | 256 | |
| GMM | cqcc | 7.61  | 27.49 |       | 4000-8000 | 256 | |
| GMM | cqcc | 4.82  | 20.30 |       | 6000-8000 | 256 | |
| GMM | cqcc | 7.15  | 19.99 |       | 7000-8000 | 256 | |
| GMM | cqcc | 4.99  | 18.05 |       | 6000-8000 | 512 | |
| GMM | cqcc | 6.63  | 18.58 |       | 6000-8000 | 1024 | |
| GMM | cqcc | 5.06  | 18.32 |       | 6000-8000 | 512 | 200 |
| GMM | cqcc | 6.64  | 18.58 |       | 6000-8000 | 1024 | 200 |
| GMM | cqcc | 7.56  | 18.07 |       | 7000-8000 | 512 | |
| GMM | cqcc | 7.97  | **17.24** | 17.64 | 7000-8000 | 1024 | |
| GMM | cqcc | 8.11  | 17.35 | 17.48 | 7000-8000 | 1024 | 200 |
| GMM | cqcc | 8.11  | 17.35 | 17.48 | 7000-8000 | 1024 | 300 |


> combine all features in the gmm


### NNET

|  system  | feature | EER(Dev) | EER(Eval) | EER(Eval) | Frequency Range | B  |       Remarks         |
| :------: | :-----: | :------: | :-------: | :-------: | :-------: | :----: | :------------------: |
| DNN      | cqcc    | 8.56     |  35.603   |           |                 |    |                       |
| LCNN     | cqcc    | 7.293    |  28.009   |           |                 |    | lr 5e-4 bs 8 wd 1e-2  |



