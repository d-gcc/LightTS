# EthanolLevel

This data set is part of the project with Scotch Whisky Research Institute into detecting forged spirits in a non-intrusive manner. One way of detecting forgery without sampling the wine is through inspecting ethanol level by spectrograph. 

The data set covers 20 different bottle types and four levels of alcohol: 35%, 38%, 40% and 45%. Each series is a spectrograph of 1751 observations. 

This data set is an example of when it is wrong to merge and resample, because the train/test split are constructed so that the same bottle type is never in both train and test sets. 

There are 4 classes corresponding to four alcohol levels. 

- Class 1: E35
- Class 2: E38
- Class 3: E40
- Class 4: E45

Train size: 504

Test size: 500

Missing value: No

Number of classses: 4

Time series length: 1751

For more information about this data set, see [1].

[1] Lines, Jason, Sarah Taylor, and Anthony Bagnall. "Hive-cote: The hierarchical vote collective of transformation-based ensembles for time series classification." Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.