# MixedShapes data sets

The data consist of "pseudo" time series obtained by converting two-dimensional shapes into one-dimensional time series (see [1], [2]).

There are five classes corresponding to different shapes.

- Class 1: Arrowhead
- Class 2: Butterfly
- Class 3: Fish
- Class 4: Seashell
- Class 5: Shield

We make two data sets out of these data, MixedShapesRegularTrain and MixedShapesSmallTrain. As the names suggest, the two data sets share a same test set and only differ in the number of training instances. 

## MixedShapesRegularTrain

Train size: 500

Test size: 2425

Missing value: No

Number of classses: 5

Time series length: 1024

## MixedShapesSmallTrain

Train size: 100

Test size: 2425

Missing value: No

Number of classses: 5

Time series length: 1024

There is nothing to infer from the order of examples in the train and test set.

Original data created by Eamonn Keogh (see [1], [2]). Data edited by Hoang Anh Dau.

[1] Wang, Xiaoyue, et al. "Annotating historical archives of images." Proceedings of the 8th ACM/IEEE-CS joint conference on Digital libraries. ACM, 2008.

[2] Keogh, Eamonn, et al. "LB_Keogh supports exact indexing of shapes under rotation invariance with arbitrary representations and distance measures." Proceedings of the 32nd international conference on Very large data bases. VLDB Endowment, 2006.