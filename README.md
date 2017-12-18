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

** These Result are very bad!!! **

|    system    | feature | EER(Dev) | EER(Eval) | EER(Eval) | Remarks |
| :---------- | :---: |:---: | :---: | :---: | :-----: |
| Baseline | mfcc | 15.19 | 33.39 | | |
| Baseline | cqcc | 10.35 | 28.48 | | |
| DNN | mfcc | 15.028 | 45.137 | 42.080 |
| DNN | cqcc | 7.588 | 34.580 | 31.900 |
| CNN | fft |  |  |  |
| CNN | db4 |  |  |  |
| CNN | db8 |  |  |  |


 
