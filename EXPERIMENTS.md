# EXPERIMENTS


## TripletLoss Benchmark

|   Loss  | Type | Margin | Size | Epoch | Precision |  Recall  | Specificity |
|:-------:|:----:|:------:|:----:|:-----:|:---------:|:--------:|:-----------:|
| Triplet | Hard |   0.1  |   2  |   30  |  0.174101 | 0.429078 |   0.571642  |
| Triplet |  All |   0.1  |   2  |   30  |  0.772388 | 0.734043 |   0.954478  |
| Triplet |  All |   0.1  |   4  |   30  |  0.808118 | 0.776596 |   0.961194  |
| Triplet |  All |   0.1  |   8  |   30  |  0.781132 | 0.734043 |   0.956716  |
| Triplet |  All |   0.1  |  16  |   30  |  0.803846 | 0.741135 |  `0.961940` |
| Triplet |  All |   0.1  |  32  |   30  |  0.816479 | 0.773050 |   0.963433  |
| Triplet |  All |   0.1  |  64  |   30  |  0.794118 | 0.765957 |   0.958209  |
| Triplet |  All |   0.1  |  128 |   30  |  0.760714 | 0.755319 |   0.950000  |
| Triplet |  All |   0.1  |  256 |   30  |  0.798419 | 0.716312 |  `0.961940` |
| Triplet |  All |   0.1  |  512 |   30  |  0.710098 | 0.773050 |   0.933582  |


## Attention

|   Loss  | Ratio | Batch | Epoch | Precision |  Recall  | Specificity |
|:-------:|:-----:|:-----:|:-----:|:---------:|:--------:|:-----------:|
| Softmax |  0.0  |   32  |  100  |  0.863071 | 0.737589 |   0.975373  |
| Softmax |  0.0  |   16  |  100  |  0.883333 | 0.751773 |   0.979104  |
| Softmax |  0.0  |   8   |  100  |  0.919283 | 0.726950 |  `0.986567` |
|:-------:|:-----:|:-----:|:-----:|:---------:|:--------:|:-----------:|
| Softmax |  0.25 |   8   |  50   |  0.873984 | 0.762411 |   0.976866  |
| Softmax |  0.5  |   8   |  50   |  0.875000 | 0.744681 |   0.977612  |
| Softmax |  0.75 |   8   |  50   |  0.859375 | 0.780142 |   0.973134  |
| Softmax |  1.0  |   8   |  50   |  0.879310 | 0.723404 |  `0.979104` |
|:-------:|:-----:|:-----:|:-----:|:---------:|:--------:|:-----------:|
| Softmax |  0.25 |   8   |  100  |  0.871795 | 0.723404 |   0.977612  |
| Softmax |  0.5  |   8   |  100  |  0.856589 | 0.783688 |   0.972388  |
| Softmax |  0.75 |   8   |  100  |  0.871369 | 0.744681 |   0.976866  |
| Softmax |  1.0  |   8   |  100  |  0.901639 | 0.780142 |  `0.982090` |
| Softmax |  1.0  |   8   |  200  |  0.918605 | 0.840426 |  `0.984328` |


## Pooling/StridedConv

|   Loss  | Ratio | Pool | Batch | Epoch | Precision |  Recall  | Specificity |
|:-------:|:-----:|:----:|:-----:|:-----:|:---------:|:--------:|:-----------:|
| Softmax |  0.25 |   1  |   8   |  100  |  0.871324 | 0.840426 |   0.973881  |
| Softmax |  0.5  |   1  |   8   |  100  |  0.894117 | 0.808510 |  `0.979850` |
| Softmax |  0.75 |   1  |   8   |  100  |  0.855633 | 0.861702 |   0.969402  |
| Softmax |  1.0  |   1  |   8   |  100  |  0.877323 | 0.836879 |   0.975373  |
|:-------:|:-----:|:----:|:-----:|:-----:|:---------:|:--------:|:-----------:|
| Softmax |  0.25 |   0  |   8   |  100  |  0.868131 | 0.840425 |  `0.973134` |
| Softmax |  0.5  |   0  |   8   |  100  |  0.840277 | 0.858156 |   0.965671  |
| Softmax |  0.75 |   0  |   8   |  100  |  0.851064 | 0.851063 |   0.968656  |
| Softmax |  1.0  |   0  |   8   |  100  |  0.860714 | 0.854609 |   0.970895  |
