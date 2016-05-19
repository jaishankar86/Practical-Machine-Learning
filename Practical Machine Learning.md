Practical Machine Learning
========================================================

The goal of this project is to predict the manner in which people did the exercise.

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

Pre-processing Data

Several columns of the raw data set have string contaning nothing, so we delete those columns first, and we also delete the first 7 columns: X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window. These features are obviously not related to predict the outcome.


```r
setwd("C:/Users/JAI/Downloads/Cousera Data Science Specialization/8. Practical Machine Learning/week4/")
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.1.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.3
```

```r
set.seed(12463)

training <- read.csv("pml-training.csv", stringsAsFactors=FALSE)
training$classe <- as.factor(training$classe)
training <- training[,-nearZeroVar(training)]
training <- training[,-c(1,2,3,4,5,6,7)]
```

There are many NA values in the data set, so we use KnnImpute method to impute those values. Besides, we try to standardize each features and use PCA to reduce features.


```r
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
training <- training[inTrain,]
testing <- training[-inTrain,]
preObj <- preProcess(training[,-length(training)],method=c("center", "scale", "knnImpute", "pca"), thresh=0.9)
clean_data <- predict(preObj,training[,-length(training)])
```

Prediction

After getting the clean data set from the above processing, we use Knn method to build model. We use testing data to evaluate the performance of our model. The accuracy is 0.9748.


```r
modelFit <- train(training$classe ~.,data=clean_data, method="knn")
test <- predict(preObj, testing[,-length(testing)])
confusionMatrix(testing$classe, predict(modelFit,test))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1018    5    6    3    0
##          B   20  703   13    0    1
##          C    8   10  605    7    2
##          D    1    0   11  587    2
##          E    1    3    3    1  676
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9737         
##                  95% CI : (0.968, 0.9786)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9667         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9714   0.9750   0.9483   0.9816   0.9927
## Specificity            0.9947   0.9885   0.9911   0.9955   0.9973
## Pos Pred Value         0.9864   0.9539   0.9573   0.9767   0.9883
## Neg Pred Value         0.9887   0.9939   0.9892   0.9964   0.9983
## Prevalence             0.2843   0.1956   0.1731   0.1622   0.1848
## Detection Rate         0.2762   0.1907   0.1641   0.1593   0.1834
## Detection Prevalence   0.2800   0.1999   0.1715   0.1630   0.1856
## Balanced Accuracy      0.9830   0.9818   0.9697   0.9885   0.9950
```

Finally, we load the testing data file and predict the reult as the following:

```r
testing <- read.csv("pml-testing.csv", stringsAsFactors=FALSE)
testing <- testing[,names(testing) %in% names(training)]

test <- predict(preObj, testing)
predict_result <- predict(modelFit, test)
```


