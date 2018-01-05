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

|  system  | feature | EER(Dev) | EER(Eval) | EER(Eval) |       Remarks        |
| :------: | :-----: | :------: | :-------: | :-------: | :------------------: |
| Baseline |  cqcc   |  10.35   |   30.60   |   24.77   |                      |
|   DNN    |  cqcc   |  8.560   |  35.603   |           |                      |
|   LCNN   |  cqcc   |  7.293   |  28.009   |           | lr 5e-4 bs 8 wd 1e-2 |
|   LCNN   |   fft   |          |           |           |                      |
|   LCNN   |   db4   |          |           |           |                      |



