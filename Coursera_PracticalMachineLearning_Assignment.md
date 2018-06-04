<style type="text/css">
body{ /* Normal  */
   font-size: 12px;
}
h2{ /* Normal  */
   font-size: 18px;
}
h3{ /* Normal  */
   font-size: 15px;
   font-weight : bold;
}
</style>
Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Data
----

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.

Overview
--------

From the HAR (Human Activity Recognition) project, we come to know that six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

In this assignment, we were provided with the data from the subset of HAR data for building a predictive model and test the model.

Loading the Libraries
---------------------

We will be using the following libraries for this assignment.

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.4.4

    ## Warning: package 'ggplot2' was built under R version 3.4.4

``` r
library(rpart)
library(e1071)
```

    ## Warning: package 'e1071' was built under R version 3.4.4

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.4.4

Loading the Data
----------------

For this assignment, we have a training data (pml-training.csv) to build our model and a validation data (pml-testing.csv) on which our model will be applied to predict the output. It has been observed that the data contains blank strings ("") and some invalid numbers ("\#DIV/0!"). We will consider those strins as NA. After loading the data, we will apply str function on the data to see the structures.

``` r
if (!file.exists("pml-training.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" ,"pml-training.csv",method="auto")
}
if (!file.exists("pml-testing.csv")) {
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" ,"pml-testing.csv",method="auto")
}
pml_training <- read.csv("pml-training.csv", na.strings = c("#DIV/0!","","NA"))

pml_testing <- read.csv("pml-testing.csv", na.strings = c("#DIV/0!","","NA"))

str(pml_training)
```

    ## 'data.frame':    19622 obs. of  160 variables:
    ##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
    ##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_roll_arm       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
    ##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
    ##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
    ##   [list output truncated]

Cleaning the Data
-----------------

From the structure, it seems that the serial number (X), user\_ \_name, timestanp (raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, cvtd\_timestamp) and window (new\_window, num\_window) are the description fields of the data and they don't influence the data.

``` r
data_training <- subset(pml_training, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
```

We can see from the data that few of the columns have NA value. Let us look into the data to find the proportion of NA values for the columns to the valid values. If the missing values for a column are completely at random or more than 95% of the data for a column is missing value, we will remove the coulmn.

``` r
pMiss <- function(x){sum(is.na(x))/length(x) > .95}
missingVector <- apply(data_training, 2, pMiss)
missingVector
```

    ##                roll_belt               pitch_belt                 yaw_belt 
    ##                    FALSE                    FALSE                    FALSE 
    ##         total_accel_belt       kurtosis_roll_belt      kurtosis_picth_belt 
    ##                    FALSE                     TRUE                     TRUE 
    ##        kurtosis_yaw_belt       skewness_roll_belt     skewness_roll_belt.1 
    ##                     TRUE                     TRUE                     TRUE 
    ##        skewness_yaw_belt            max_roll_belt           max_picth_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##             max_yaw_belt            min_roll_belt           min_pitch_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##             min_yaw_belt      amplitude_roll_belt     amplitude_pitch_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##       amplitude_yaw_belt     var_total_accel_belt            avg_roll_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##         stddev_roll_belt            var_roll_belt           avg_pitch_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##        stddev_pitch_belt           var_pitch_belt             avg_yaw_belt 
    ##                     TRUE                     TRUE                     TRUE 
    ##          stddev_yaw_belt             var_yaw_belt             gyros_belt_x 
    ##                     TRUE                     TRUE                    FALSE 
    ##             gyros_belt_y             gyros_belt_z             accel_belt_x 
    ##                    FALSE                    FALSE                    FALSE 
    ##             accel_belt_y             accel_belt_z            magnet_belt_x 
    ##                    FALSE                    FALSE                    FALSE 
    ##            magnet_belt_y            magnet_belt_z                 roll_arm 
    ##                    FALSE                    FALSE                    FALSE 
    ##                pitch_arm                  yaw_arm          total_accel_arm 
    ##                    FALSE                    FALSE                    FALSE 
    ##            var_accel_arm             avg_roll_arm          stddev_roll_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##             var_roll_arm            avg_pitch_arm         stddev_pitch_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##            var_pitch_arm              avg_yaw_arm           stddev_yaw_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##              var_yaw_arm              gyros_arm_x              gyros_arm_y 
    ##                     TRUE                    FALSE                    FALSE 
    ##              gyros_arm_z              accel_arm_x              accel_arm_y 
    ##                    FALSE                    FALSE                    FALSE 
    ##              accel_arm_z             magnet_arm_x             magnet_arm_y 
    ##                    FALSE                    FALSE                    FALSE 
    ##             magnet_arm_z        kurtosis_roll_arm       kurtosis_picth_arm 
    ##                    FALSE                     TRUE                     TRUE 
    ##         kurtosis_yaw_arm        skewness_roll_arm       skewness_pitch_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##         skewness_yaw_arm             max_roll_arm            max_picth_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##              max_yaw_arm             min_roll_arm            min_pitch_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##              min_yaw_arm       amplitude_roll_arm      amplitude_pitch_arm 
    ##                     TRUE                     TRUE                     TRUE 
    ##        amplitude_yaw_arm            roll_dumbbell           pitch_dumbbell 
    ##                     TRUE                    FALSE                    FALSE 
    ##             yaw_dumbbell   kurtosis_roll_dumbbell  kurtosis_picth_dumbbell 
    ##                    FALSE                     TRUE                     TRUE 
    ##    kurtosis_yaw_dumbbell   skewness_roll_dumbbell  skewness_pitch_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##    skewness_yaw_dumbbell        max_roll_dumbbell       max_picth_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##         max_yaw_dumbbell        min_roll_dumbbell       min_pitch_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##         min_yaw_dumbbell  amplitude_roll_dumbbell amplitude_pitch_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##   amplitude_yaw_dumbbell     total_accel_dumbbell       var_accel_dumbbell 
    ##                     TRUE                    FALSE                     TRUE 
    ##        avg_roll_dumbbell     stddev_roll_dumbbell        var_roll_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##       avg_pitch_dumbbell    stddev_pitch_dumbbell       var_pitch_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##         avg_yaw_dumbbell      stddev_yaw_dumbbell         var_yaw_dumbbell 
    ##                     TRUE                     TRUE                     TRUE 
    ##         gyros_dumbbell_x         gyros_dumbbell_y         gyros_dumbbell_z 
    ##                    FALSE                    FALSE                    FALSE 
    ##         accel_dumbbell_x         accel_dumbbell_y         accel_dumbbell_z 
    ##                    FALSE                    FALSE                    FALSE 
    ##        magnet_dumbbell_x        magnet_dumbbell_y        magnet_dumbbell_z 
    ##                    FALSE                    FALSE                    FALSE 
    ##             roll_forearm            pitch_forearm              yaw_forearm 
    ##                    FALSE                    FALSE                    FALSE 
    ##    kurtosis_roll_forearm   kurtosis_picth_forearm     kurtosis_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##    skewness_roll_forearm   skewness_pitch_forearm     skewness_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##         max_roll_forearm        max_picth_forearm          max_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##         min_roll_forearm        min_pitch_forearm          min_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##   amplitude_roll_forearm  amplitude_pitch_forearm    amplitude_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##      total_accel_forearm        var_accel_forearm         avg_roll_forearm 
    ##                    FALSE                     TRUE                     TRUE 
    ##      stddev_roll_forearm         var_roll_forearm        avg_pitch_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##     stddev_pitch_forearm        var_pitch_forearm          avg_yaw_forearm 
    ##                     TRUE                     TRUE                     TRUE 
    ##       stddev_yaw_forearm          var_yaw_forearm          gyros_forearm_x 
    ##                     TRUE                     TRUE                    FALSE 
    ##          gyros_forearm_y          gyros_forearm_z          accel_forearm_x 
    ##                    FALSE                    FALSE                    FALSE 
    ##          accel_forearm_y          accel_forearm_z         magnet_forearm_x 
    ##                    FALSE                    FALSE                    FALSE 
    ##         magnet_forearm_y         magnet_forearm_z                   classe 
    ##                    FALSE                    FALSE                    FALSE

Exclude the columns that contains more than 95% missing value.

``` r
data_training <- data_training[,names(missingVector[missingVector == FALSE])]
```

Now let us check, how many variables are there with near zero variance and if there is any we will investigate and exclude such columns.

``` r
sum(nearZeroVar(data_training, saveMetrics=TRUE)$nzv)
```

    ## [1] 0

As the sum of all the near zero variables are zero, we don't need to exclude any columns from the data set.

Partitioning the Training Data
------------------------------

We will be pratitioning the training data into two parts. We will use 80% of the randomly partitioned data to build the model and rest of the data (20%) to test the model. We will set the seed 12345 to replicate the scenarios in future.

``` r
set.seed(12345)
inTrain <- createDataPartition(data_training$classe, p = 0.8, list = FALSE)
dt_train <- data_training[inTrain,]
dt_test <- data_training[-inTrain,]
```

Decision Tree
-------------

Let us train the decision tree model and find out the accuracy of the model for the training data.

``` r
modelDecisionTree <- rpart(classe~., data=dt_train)
```

Now we will test the model by applying it onto the testing data that was randomly partitioned from the original training data.

``` r
predictDecisionTree <- predict(modelDecisionTree, dt_test, type="class")
confusionMatrix(predictDecisionTree, dt_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1003  165    9   77   12
    ##          B   21  390   72   23   50
    ##          C   36   64  542  101   92
    ##          D   33   58   45  408   36
    ##          E   23   82   16   34  531
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7326          
    ##                  95% CI : (0.7185, 0.7464)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6604          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8987  0.51383   0.7924   0.6345   0.7365
    ## Specificity            0.9063  0.94753   0.9095   0.9476   0.9516
    ## Pos Pred Value         0.7923  0.70144   0.6491   0.7034   0.7741
    ## Neg Pred Value         0.9575  0.89041   0.9540   0.9297   0.9413
    ## Prevalence             0.2845  0.19347   0.1744   0.1639   0.1838
    ## Detection Rate         0.2557  0.09941   0.1382   0.1040   0.1354
    ## Detection Prevalence   0.3227  0.14173   0.2128   0.1478   0.1749
    ## Balanced Accuracy      0.9025  0.73068   0.8510   0.7910   0.8440

Accuracy of the decision tree model is given below.

``` r
confusionMatrix(predictDecisionTree, dt_test$classe)$overall["Accuracy"]
```

    ##  Accuracy 
    ## 0.7326026

The accuracy of the decision tree algorithm is very low. Let us try some other algorithm to fit the model.

Random Forest
-------------

Let us train the decision tree model and find out the accuracy of the model for the training data.

``` r
modelRandomForest <- randomForest(classe~., data=dt_train)
```

Now we will test the model by applying it onto the testing data that was randomly partitioned from the original training data.

``` r
predictRandomForest <- predict(modelRandomForest, dt_test)
confusionMatrix(predictRandomForest, dt_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    3    0    0    0
    ##          B    0  756   10    0    0
    ##          C    0    0  674   10    0
    ##          D    0    0    0  633    1
    ##          E    0    0    0    0  720
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9939          
    ##                  95% CI : (0.9909, 0.9961)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9923          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9960   0.9854   0.9844   0.9986
    ## Specificity            0.9989   0.9968   0.9969   0.9997   1.0000
    ## Pos Pred Value         0.9973   0.9869   0.9854   0.9984   1.0000
    ## Neg Pred Value         1.0000   0.9990   0.9969   0.9970   0.9997
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1927   0.1718   0.1614   0.1835
    ## Detection Prevalence   0.2852   0.1953   0.1744   0.1616   0.1835
    ## Balanced Accuracy      0.9995   0.9964   0.9911   0.9921   0.9993

Accuracy of the random forest model is given below.

``` r
confusionMatrix(predictRandomForest, dt_test$classe)$overall["Accuracy"]
```

    ##  Accuracy 
    ## 0.9938822

As the accuracy of the random forest model is very high, we will be using this model to predict the data from pml-testing.csv.

Predict the data
----------------

Below is the predited output of the classe variable for pml-testing.csv.

``` r
predict(modelRandomForest, pml_testing)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
