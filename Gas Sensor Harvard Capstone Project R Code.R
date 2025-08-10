knitr::opts_chunk$set(echo = TRUE)
#Importing all the libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(reshape2)) install.packages("reshape2")
if(!require(brnn)) install.packages("brnn")
if(!require(gam)) install.packages("gam")

#Bringing in the dataset
#Original dataset at https://www.kaggle.com/datasets/uciml/gas-sensor-array-under-dynamic-gas-mixtures/data
zip_file_location <- "ethylene_CO.txt.zip"
dataset <- "ethylene_CO.txt"
#Unzipping the dataset from the zip file
if(!file.exists(dataset))
  unzip(zip_file_location, dataset)
#The column names in the original dataset are labeled as "Time (seconds), CO conc (ppm), Ethylene conc (ppm), sensor readings (16 channels)", so I pull in the data without the headers and then reassign them with the complete names
#Here, I pull in the data, but omit the headers
completedf <- read.csv(dataset, header = FALSE, sep = "", check.names = FALSE, skip = 1)
#Here I add the correct headers back into the dataset
colnames(completedf) <- c("Time_seconds","CO_conc_ppm","Ethylene_conc_ppm", paste0("Sensor",1:16))


#############################
##Introduction/Overview/Executive Summary
#############################
#The goal of this project is to compare 4 different machine learning models for predicting Carbon Monoxide concentration in an Ethylene and Air mixture. The dataset used is called "Gas sensor array under dynamic gas mixtures" on Kaggle and was created by the University of California San Diego. The data was gathered by creating different concentration mixtures of Ethylene and Carbon Monoxide (also known as "CO") with air and then measuring the resulting gas mixture with 16 sensors. Because Carbon Monoxide is a potential product of the combustion of Ethylene, methods for determining the concentration of CO in Ethylene and air will have real world applications. I will explore the data, prepare the data, fit it to 4 different models, then compare the RMSE (Root Mean Squared Error) results at the end. 



#############################
##Methods and Analysis
#############################

# I chose 4 popular and powerful models for this exercise: K Nearest Neighbors, Generalized Linear Model, Neural Network, and Bayesian Regularized Neural Network. Here is why I chose those specifically.
# All 4 are designed to work for regression tasks, all 4 are included in the caret package, and the first 3 are quite popular which makes them useful for model comparison. I've chosen not to tune any hyperparameters so as to compare these models as they come.
# I'm also curious to see if the Bayesian Regularized Neural Network fairs better than the standard neural network, so I've included that for comparison.
# In the end, I have 4 models in total.



#I set the seed to 500. The seed is a usually arbitrary number that determines the random events of the code. I set it manually so that my code runs the same way each time
set.seed(500)
#I select a portion of the data so that model run times are reasonable. I randomize the selection so that I still get a representative sample of the dataset.
df <- completedf[sample(nrow(completedf),50000),]

#This is where I explore the data
#First, I print the first 10 rows. This allows me to see that the header assignment worked as expected and gives me an idea of the typical values in the dataset
print(head(df,10))
#I print the structure in this line. It tells me about the shape of the datset, if there's any nested values (which there aren't in this case), the data types of the columns, and the names of the columns. Knowing this information is very helpful later on as I work with this data. For example, if the Time_seconds column were a string instead of a number, that would tell me something went wrong with the data import and prompt me to investigate further. I also notice that many of the columns have a lot of overlapping data. For example, CO_conc_ppm has a Min of 0, a 1st Quartile of 0, and a Median that's also 0. I see other columns that also have a lot of repeating or close values. These values appear to be close in feature space.
print(str(df))
#This line tells me there are no null values in the dataset. This is one data cleaning step I no longer have to do
print(colSums(is.na(df)))
#Finally, the summary gives me some statistics about the distribution of the data. Because of this line, I now know CO_conc_ppm ranges from 0-533.3. This can help inform me later on concerning whether a particular RMSE is good or bad
print(summary(df))



#I have a lot of columns here and reducing the total number of them could help my models run faster. To figure out which columns to remove, I make a correlation matrix. This will tell me how linearly related different columns' values are.

#Calculate the correlations
cor_matrix <- cor(df)
#Melt the data into the correct format
cor_melt <- melt(cor_matrix)
#Create the correlation matrix with the values made above. 
ggplot(cor_melt, aes(Var1, Var2, fill = value)) + geom_tile() + geom_text(aes(label=round(value,2)), color = "black", size = 3) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) + scale_fill_gradient(low="blue", high = "red")


#It looks like there are some columns that aren't correlated with CO_conc_ppm nearly at all. To make the model run faster, I'll remove those columns from the dataset

#Pull the correlations for CO_conc_ppm
correlation_vector <- cor(df)[, "CO_conc_ppm"]
#Create the list of columns with correlations greater than 0.15. 0.15 is an arbitrary number that I picked. Had I picked a greater number, I would have removed more columns. It's possible that these columns that I'm removing would have been useful in at least one of the models created, however, due to lack of correlation, I make the educated guess that these columns would be less useful than the ones kept. 
cols_to_keep <- names(correlation_vector)[abs(correlation_vector) >= 0.15]
#Removing the columns that are not in the cols_to_keep variable
df <- df[,cols_to_keep]



#The metric I will be using to grade each model's performance is RMSE. Because I will be using it often, I will make a function for it. 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Creating my train and test set. 20% allocated to the test set is convention. Because I have a large dataset, I could have also gotten away with using a little less towards testing if I wanted to.
test_index <- createDataPartition(y = df$`CO_conc_ppm`, times = 1,p = 0.2, list = FALSE)
train_set <- df[-test_index,]
test_set <- df[test_index,]






#Printing the RMSE just using the mean of all values. This is to give me something to compare to later on
BaselineRMSE <- RMSE(df$CO_conc_ppm, mean(df$CO_conc_ppm))
print(BaselineRMSE)
#Fitting the KNN model
knn_fit <- train(CO_conc_ppm ~ ., data = train_set, method = "knn")
#Predicting using the KNN model
y_hat_knn <- predict(knn_fit, test_set)
#Printing the RMSE
KNNRMSE <-RMSE(test_set$CO_conc_ppm, y_hat_knn)
print(KNNRMSE)





#Fitting the glm model
glm_fit <- train(CO_conc_ppm ~ ., data = train_set, method = "glm")
#Predicting using the glm model
y_hat_glm <- predict(glm_fit, test_set)
#Printing the RMSE
GLMRMSE <- RMSE(test_set$CO_conc_ppm, y_hat_glm)
print(GLMRMSE)




#Fitting the Standard Neural Network model
nnet_fit <- train(CO_conc_ppm ~ ., data = train_set, method = "nnet")
#Predicting using the nnet model
y_hat_nnet <- predict(nnet_fit, test_set)
#printing the RMSE
NNETRMSE <- RMSE(test_set$CO_conc_ppm, y_hat_nnet)
print(NNETRMSE)


#Fitting the Bayesian Regularized Neural Network model
brnn_fit <- train(CO_conc_ppm ~ ., data = train_set, method = "brnn")
#Predicting using the brnn model
y_hat_brnn <- predict(brnn_fit, test_set)
#printing the RMSE
BRNNRMSE <- RMSE(test_set$CO_conc_ppm, y_hat_brnn)
print(BRNNRMSE)


#############################
##Results
#############################


RMSEdf <- data.frame(Method = c("Predict Using Only The Mean","K Nearest Neighbors","Generalized Linear Model","Neural Network","Bayesian Regularized Neural Network"),
                     RMSE = c(BaselineRMSE,KNNRMSE,GLMRMSE,NNETRMSE,BRNNRMSE))
print(RMSEdf %>% arrange(RMSE))



#K Nearest Neighbors did the best by far with an RMSE of 39.04079. I suspect that the good performance is due to the fact that the data points had a lot of near overlapping data points. The values were close in feature space, leading to good performance on a model that draws predictions from the nearest points in feature space. The standard Neural Network performed the worst, despite the model's reputation for achieving amazing results. It even performed worse than predicting using only the mean. I suspect this is because I used the model as it comes without any tuning. It could be that Neural Networks are especially susceptible to incorrectly tuned hyperparameters. Since the introduction of Bayesian Regularization appears to have improved the model considerably, I suspect overfitting may also have a part in the standard Neural Network's poor performance.

#############################
##Conclusion
#############################
#I tried 4 different out of the box machine learning models to predict the Carbon Monoxide concentration in Ethylene and Air. Those models were K Nearest Neighbors, Generalized Linear Model, Neural Network, and Bayesian Regularized Neural Network. In the end, the K Nearest Neighbors did the best with an RMSE of 39.04079 and the standard Neural Network fared the worst with an RMSE of 203.42177. These findings could prove helpful in the chemical industry in the use of Carbon Monoxide concentration testing. A major limitation of these models is the lack of hyperparameter tuning. This tuning was omitted from this project for the sake of calculation speed and so that the base models could be compared. For future work, I would like to repeat this project, but with hyperparameter tuning. I suspect the neural networks especially would fare significantly better in that case.An ensemble of the models that perform better than just the mean could be made for even better model performance.




#############################
##References
#############################
#Ethylene combustion chemistry learned from my Chemistry Bachelors degree, so no reference is needed

#https://www.kaggle.com/datasets/uciml/gas-sensor-array-under-dynamic-gas-mixtures/data