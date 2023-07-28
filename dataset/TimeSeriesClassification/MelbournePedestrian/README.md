# PedestrianCountingSystem data sets

The City of Melbourne, Australia has developed an automated pedestrian counting system to better understand pedestrian activity within the municipality, such as how people use different city locations at different times of the day. The data analysis can facilitate decision making and urban planning for the future. 

We extract data of 10 locations for the whole year of 2017. We make two data sets from these data.

## MelbournePedestrian

Data are pedestrian count for 12 months of the year 2017. Classes correspond to locations of sensor placement.

- Class 1: Bourke Street Mall (North)
- Class 2: Southern Cross Station
- Class 3: New Quay
- Class 4: Flinders St Station Underpass
- Class 5: QV Market-Elizabeth (West)
- Class 6: Convention/Exhibition Centre
- Class 7: Chinatown-Swanston St (North)
- Class 8: Webb Bridge
- Class 9: Tin Alley-Swanston St (West)
- Class 10: Southbank

Train size: 1194

Test size: 2439

Missing value: Yes

Number of classses: 10

Time series length: 24

## Chinatown

Data are pedestrian count in Chinatown-Swanston St (North) for 12 months of the year 2017. Classes are based on whether data are from a normal day or a weekend day.  

- Class 1: Weekend
- Class 2: Weekday

Train size: 20

Test size: 343

Missing value: No

Number of classses: 2

Time series length: 24

There is nothing to infer from the order of examples in the train and test set.

Data source: City of Melbourne (see [1]). Data edited by Hoang Anh Dau.

[1] http://www.pedestrian.melbourne.vic.gov.au/#date=11-06-2018&time=4

## Revision notes - September 2019:

### Chinatown

- As the data are pedestrian counts, -1 should be treated as missing values.
- Removes two cases from the test set, each with a single -1 value. Now the test size reduces from 345 to 343 exemplars.

### MelbournePedestrian
- As the data are pedestrian counts, -1 should be treated as missing values.
- Removes cases with all -1 values; for the other exemplars, replaces -1 with NaN. 
- For the train set, there are 62 cases with missing values, among which 6 cases with all missing values got removed. Now the train size reduces from 1200 to 1194 exemplars.
- For the test set, there are 131 cases with mising values, among which 11 cases with all missing values got removed. Now the test size reduces from 2450 to 2439 exemplars.
- MelbournePedestrian is now a data set with missing values.






