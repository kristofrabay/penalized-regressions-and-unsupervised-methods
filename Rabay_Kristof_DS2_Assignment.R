## ---- include = F--------------------------------------------------------
library(data.table)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(magrittr)
library(tidyverse)
library(skimr)
library(GGally)
library(pls)
library(NbClust)
library(factoextra)
library(ggrepel)

options(digits = 6)
options(scipen = 999)
theme_set(theme_bw())


## ---- echo=FALSE---------------------------------------------------------
data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>% as.data.table()
data[, logTotalValue := log(TotalValue)]
data <- data[complete.cases(data)]


## ---- echo = F-----------------------------------------------------------
str(data)


## ------------------------------------------------------------------------
table(data$Council)
table(data$PolicePrct)
table(data$HealthArea)


## ---- echo = F-----------------------------------------------------------
data$Council <- as.factor(data$Council)
data$PolicePrct <- as.factor(data$PolicePrct)
data$HealthArea <- as.factor(data$HealthArea)


## ------------------------------------------------------------------------
# for PCR, I will save dt with all columns, no matter their correlation or values
data$ID <- NULL
data_for_pcr <- copy(data)


## ------------------------------------------------------------------------
lapply(data.frame(data)[, sapply(data.frame(data), is.factor)], table)


## ---- echo = F-----------------------------------------------------------
data$ZoneDist2 <- NULL
data$ZoneDist3 <- NULL
data$ZoneDist4 <- NULL
data$Extension <- NULL


## ---- fig.height = 4, fig.width = 6,fig.align='center'-------------------
ggplot(data, aes(TotalValue)) + geom_histogram(binwidth = 100000, fill = "skyblue") + 
  labs(title = "Distribution of property values",
       x = "Value ranges",
       y = "Count") +
  scale_x_continuous(labels = scales::dollar)


## ---- fig.height = 4, fig.width = 6,fig.align='center'-------------------
ggplot(data, aes(logTotalValue)) + geom_histogram(binwidth = 0.05, fill = "skyblue") + 
  labs(title = "Distribution of property values in log scale",
       x = "Value ranges",
       y = "Count") +
  scale_x_continuous(labels = scales::comma)


## ---- results='asis'-----------------------------------------------------
kable(data[logTotalValue < 10, .(under_log_10 = .N)])


## ------------------------------------------------------------------------
names(data[, sapply(data, is.numeric), with = FALSE])


## ---- fig.width = 10, fig.height=10,fig.align='center'-------------------
numerics <- names(data[, sapply(data, is.numeric), with = FALSE])
corrplot::corrplot(cor(data.frame(data)[numerics]), 
                   method = "number",
                   tl.cex = 0.66, 
                   tl.col = "black",
                   diag = T, 
                   cl.cex = 0.75, 
                   number.cex = 2/3)


## ---- echo = F-----------------------------------------------------------
data$UnitsRes <- NULL


## ------------------------------------------------------------------------
summary(data %>% select(names(data[, sapply(data, is.numeric) & !colnames(data) %like% "TotalValue", with = FALSE])))


## ---- warning=F----------------------------------------------------------
numerics <- names(data[, sapply(data, is.numeric) & !colnames(data) %like% "TotalValue", with = FALSE])
data <- data %>% mutate_at(vars(numerics), funs("log" = ifelse(. != 0, log(.), 0)))
data <- data[!names(data) %in% numerics]


## ---- fig.width = 20, fig.height=20,fig.align='center', warning=F--------
data[names(data) %like% '_log'] %>% gather() %>% 
  ggplot(aes(value, fill = key, color = key)) +
  geom_density(show.legend = F) +
  facet_wrap(~key, scales = 'free') +
  labs(title = "Numeric features distribution, log scales features",
       x = NULL,
       y = NULL)



## ------------------------------------------------------------------------
set.seed(20200214)
train_index <- createDataPartition(data[["logTotalValue"]], p = 0.3, list = F)
train <- data[train_index,]
test <- data[-train_index]


## ---- results = 'asis'---------------------------------------------------
dummy_mean <- mean(train$logTotalValue)
dummy_RMSE <- RMSE(dummy_mean, train$logTotalValue)
kable(data.frame("model" = "dummy_mean", "RMSE" = dummy_RMSE, "RMSE_to_avg" = (dummy_RMSE / mean(train$logTotalValue))))


## ---- warning=F----------------------------------------------------------
set.seed(20200214)
ols_model <- train(logTotalValue ~ . -TotalValue,
                   data = train, 
                   trControl = trainControl(method = "cv", number = 10), 
                   method = "lm",
                   preProcess = c("center", "scale"))


## ------------------------------------------------------------------------
ols_model


## ------------------------------------------------------------------------
ols_model$results$RMSE / mean(train$logTotalValue)


## ----warning=F-----------------------------------------------------------
set.seed(20200214)

lasso_fit <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = 1,
                                          lambda = seq(0.00001, 0.01, by = 0.00001)), # knowing the best hyperparams
                   trControl = trainControl(method = "cv", number = 10))



## ---- results = 'asis', echo = F-----------------------------------------
lasso_best <- lasso_fit$results[lasso_fit$results$lambda == lasso_fit$bestTune$lambda,]
kable(lasso_best)


## ---- fig.align='center', echo = F---------------------------------------
plot(lasso_fit)


## ---- fig.align='center', echo = F---------------------------------------
plot.train(lasso_fit, plotType = "scatter", metric = "Rsquared", output = "ggplot")


## ---- echo = F, fig.align='center'---------------------------------------
plot(lasso_fit$finalModel, xvar = 'lambda', label=T)


## ----warning=F-----------------------------------------------------------
set.seed(20200214)

ridge_fit <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = 0,
                                          lambda = seq(0.001, 0.2, by = 0.001)), # knowing the best hyperparams
                   trControl = trainControl(method = "cv", number = 10))



## ---- results = 'asis', echo = F-----------------------------------------
ridge_best <- ridge_fit$results[ridge_fit$results$lambda == ridge_fit$bestTune$lambda,]
kable(ridge_best)


## ---- fig.align='center', echo = F---------------------------------------
plot(ridge_fit)


## ---- fig.align='center', echo = F---------------------------------------
plot.train(ridge_fit, plotType = "scatter", metric = "Rsquared", output = "ggplot")


## ---- echo = F, fig.align='center'---------------------------------------
plot(ridge_fit$finalModel, xvar = 'lambda', label=T)


## ---- warning=F----------------------------------------------------------
set.seed(20200214)

# knowing the best hyperparams, I won't give it a grid search to reduce knitting time
elnet_fit <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = 0.8, 
                                          lambda = 0.0001),
                   trControl = trainControl(method = "cv", number = 10))



## ---- results = 'asis', echo = F-----------------------------------------
elnet_best <- elnet_fit$results[elnet_fit$results$lambda == elnet_fit$bestTune$lambda &
                                 elnet_fit$results$alpha == elnet_fit$bestTune$alpha,]
kable(elnet_best)


## ----results = 'asis', echo = F------------------------------------------
allmodels <- rbind(lasso_best, ridge_best, elnet_best)
allmodels <- allmodels %>% arrange(RMSE)
rownames(allmodels) <- c("Elastic Net", "LASSO", "RIDGE")
kable(allmodels)


## ----results = 'asis', echo = F------------------------------------------
allmodels <- plyr::rbind.fill(lasso_best, ridge_best, elnet_best, ols_model$results[2:7]) %>% arrange(RMSE)
rownames(allmodels) <- c("Elastic Net", "OLS", "LASSO", "RIDGE")
kable(allmodels)


## ---- echo = F ,warning=F------------------------------------------------
# knowing the best hyperparams, I won't give them too much hyperparams to reduce knitting time
set.seed(20200214)
lasso_fit_1se <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = 1,
                                          lambda = seq(0.00001, 0.008, by = 0.00001)),
                   trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE"))

lasso_best_1se <- lasso_fit_1se$results[lasso_fit_1se$results$lambda == lasso_fit_1se$bestTune$lambda,]

set.seed(20200214)
ridge_fit_1se <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = 0,
                                          lambda = seq(0.001, 1, by = 0.001)),
                   trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE"))
ridge_best_1se <- ridge_fit_1se$results[ridge_fit_1se$results$lambda == ridge_fit_1se$bestTune$lambda,]

set.seed(20200214)
elnet_fit_1se <- train(logTotalValue ~ . -TotalValue,
                   data = train,
                   method = "glmnet",
                   #preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(alpha = seq(0.7, 0.8, by = 0.1), # knowing best values, to reduce computation
                                          lambda = seq(0.0001, 0.001, by = 0.0001)),
                   trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE"))
elnet_best_1se <- elnet_fit_1se$results[elnet_fit_1se$results$lambda == elnet_fit_1se$bestTune$lambda &
                                 elnet_fit_1se$results$alpha == elnet_fit_1se$bestTune$alpha,]


## ----results = 'asis', echo = F------------------------------------------
allmodels_1se <- rbind(lasso_best_1se, ridge_best_1se, elnet_best_1se) %>% arrange(RMSE)
rownames(allmodels_1se) <- c("Elastic Net", "LASSO", "RIDGE")
kable(allmodels_1se)


## ---- echo = F-----------------------------------------------------------
# taking logs
numerics <- names(data_for_pcr[, sapply(data_for_pcr, is.numeric) & !colnames(data_for_pcr) %like% "TotalValue", with = FALSE])
data_for_pcr <- data_for_pcr %>% mutate_at(vars(numerics), funs("log" = ifelse(. != 0, log(.), 0)))
data_for_pcr <- data_for_pcr[!names(data_for_pcr) %in% numerics]

# train test for pcr data
set.seed(20200214)
train_index_pcr <- createDataPartition(data_for_pcr[["logTotalValue"]], p = 0.3, list = F)
train_pcr <- data_for_pcr[train_index_pcr,]
test_pcr <- data_for_pcr[-train_index_pcr,]



## ---- warning=F----------------------------------------------------------
set.seed(20200214)

pcr_fit <- train(logTotalValue ~ . -TotalValue, 
                 data = train_pcr, 
                 method = "pcr", 
                 trControl = trainControl(method = "cv", number = 10),
                 tuneGrid = data.frame(ncomp = 30:231),
                 preProcess = c("center", "scale"))


## ---- echo = F, results='asis'-------------------------------------------
pcr_best <- pcr_fit$results[pcr_fit$results$ncomp == pcr_fit$bestTune$ncomp,]
pcr_vs_ols <- plyr::rbind.fill(ols_model$results[2:7], pcr_best) %>% arrange(RMSE)
rownames(pcr_vs_ols) <- c("PCR", "OLS")
kable(pcr_vs_ols)


## ---- warning=F----------------------------------------------------------
# knowing the best hyperparams, I won't give it a grid search to reduce knitting time
set.seed(20200214)

pca_penalized <- train(logTotalValue ~ . -TotalValue, 
                       data = train_pcr, 
                       method = "glmnet", 
                       trControl = trainControl(method = "cv", number = 10,
                                                preProcOptions = list(thresh= 0.95)),
                       tuneGrid = expand.grid(alpha = 0.3, # knowing best values, to reduce computation
                                              lambda = 0.003),
                       preProcess = c("pca", "nzv"))



## ---- echo = F, results='asis'-------------------------------------------
pca_pen_best <- pca_penalized$results[pca_penalized$results$alpha == pca_penalized$bestTune$alpha & 
                                      pca_penalized$results$lambda == pca_penalized$bestTune$lambda,]

elnet_vs_elnetpca <- rbind(elnet_best, pca_pen_best)
rownames(elnet_vs_elnetpca) <- c("Simple Elastic Net", "PCA Elastic Net")
kable(elnet_vs_elnetpca)


## ------------------------------------------------------------------------
ncol(pca_penalized$preProcess$rotation)


## ---- echo = F, results='asis'-------------------------------------------
all_ex1 <- plyr::rbind.fill(allmodels, pcr_best, pca_pen_best) %>% arrange(RMSE)
all_ex1 <- all_ex1[3:4]
rownames(all_ex1) <- c("PCR", "Elastic Net", "OLS", "LASSO", "RIDGE", "Elastic Net PCA")
kable(all_ex1)


## ---- warning = F, results = 'asis', echo = F----------------------------

kable(data.frame("Model" = "PCR",
                 "RMSE_test" = RMSE(predict(pcr_fit, newdata = test_pcr), test_pcr$logTotalValue),
                 "R2_test" = caret::R2(predict(pcr_fit, newdata = test_pcr), test_pcr$logTotalValue)))



## ---- echo = F-----------------------------------------------------------
data <- USArrests


## ---- echo = F-----------------------------------------------------------
skim_with(numeric = list(hist = NULL),
          integer = list(hist = NULL))
skim(data)


## ------------------------------------------------------------------------
prcomp(data, scale. = F)


## ------------------------------------------------------------------------
prcomp(data, scale. = T)


## ---- results='hide', message=FALSE, warning=FALSE, fig.align='center'----
set.seed(20200216)
noc_opt <- NbClust(scale(data), 
                   method = "kmeans", 
                   min.nc = 2, 
                   max.nc = 7, 
                   index = "all")


## ---- fig.height = 4, fig.width = 6,fig.align='center', echo = F---------
fviz_nbclust(noc_opt)


## ---- fig.height = 4, fig.width = 6,fig.align='center'-------------------
set.seed(20200216)
fviz_nbclust(scale(data), kmeans, method = "wss") +
    labs(subtitle = "Elbow method")


## ---- fig.height = 4, fig.width = 6,fig.align='center'-------------------
set.seed(20200216)
fviz_nbclust(scale(data), kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")


## ---- fig.height = 4, fig.width = 6,fig.align='center'-------------------
set.seed(20200216)
fviz_nbclust(scale(data), kmeans, nstart = 25,  method = "gap_stat", nboot = 500)+
  labs(subtitle = "Gap statistic method")



## ---- results = 'hide'---------------------------------------------------
set.seed(20200216)
data_w_clusters <- cbind(data, 
                         data.table("cluster" = factor(kmeans(scale(data), 
                                                              centers = 2, 
                                                              nstart = 100)$cluster)))

data_w_clusters$state <- rownames(data_w_clusters)


## ---- fig.height = 8, fig.width = 12,fig.align='center', echo = F--------
ggplot(data_w_clusters, 
       aes(x = UrbanPop, 
           y = Assault, 
           color = cluster)) + 
  #geom_point(size = 2, show.legend = F) +
  geom_text_repel(aes(label = state), show.legend = F) +
  labs(title = "The 2 clusters with regards to population - assault #",
       x = "Population",
       y = "Assaults")


## ---- fig.height = 8, fig.width = 12,fig.align='center', echo = F--------
ggplot(data_w_clusters, 
       aes(x = UrbanPop, 
           y = Murder, 
           color = cluster)) + 
  #geom_point(size = 2, show.legend = F) +
  geom_text_repel(aes(label = state), show.legend = F) +
  labs(title = "The 2 clusters with regards to population - murder #",
       x = "Population",
       y = "Murders")


## ---- fig.height = 8, fig.width = 12,fig.align='center', echo = F--------
ggplot(data_w_clusters, 
       aes(x = UrbanPop, 
           y = Rape, 
           color = cluster)) + 
  #geom_point(size = 2, show.legend = F) +
  geom_text_repel(aes(label = state), show.legend = F) +
  labs(title = "The 2 clusters with regards to population - rape #",
       x = "Population",
       y = "Cases of rape")


## ---- fig.height = 8, fig.width = 12,fig.align='center', echo = F--------
set.seed(20200216)
fviz_cluster(kmeans(scale(data), centers = 4, nstart = 100), 
             data = scale(data), 
             show.clust.cent = T, 
             ellipse.type = "convex") +
  theme_minimal() +
  labs(title = "4 clusters")


## ---- fig.height = 8, fig.width = 12,fig.align='center', echo = F--------
set.seed(20200216)
fviz_cluster(kmeans(scale(data), centers = 2, nstart = 100), 
             data = scale(data), 
             show.clust.cent = T, 
             ellipse.type = "convex") +
  theme_minimal() +
  labs(title = "2 clusters")


## ------------------------------------------------------------------------
print(paste0("Dim1 is: " ,
             format(list(prcomp(data, scale. = T)$sdev^2)[[1]][1][1] / sum(prcomp(data, scale. = T)$sdev^2), digits = 2),
             " and Dim2 is: ",
             format(list(prcomp(data, scale. = T)$sdev^2)[[1]][2][1] / sum(prcomp(data, scale. = T)$sdev^2), digits = 3)))


## ------------------------------------------------------------------------
pc2 <- data.frame(prcomp(data, scale. = TRUE)$x[, 1:2])
pc2[rownames(pc2) == "Mississippi",]


## ---- echo = F-----------------------------------------------------------
data <- fread("../../machine-learning-course/data/gene_data_from_ISLR_ch_10/gene_data.csv")
data[, is_diseased := factor(is_diseased)]


## ---- echo = F-----------------------------------------------------------
data_features <- copy(data)
data_features$is_diseased <- NULL


## ------------------------------------------------------------------------
pc2 <- data.frame(prcomp(data_features, scale. = T)$x[, 1:2])

# for plotting purposes, otherwise "I do not know the label"
pc2$outcome <- as.character(data$is_diseased)


## ---- fig.height = 4, fig.width = 6,fig.align='center', echo = T---------
ggplot(pc2, aes(PC1, PC2, color = outcome, fill = outcome)) + 
  geom_point() +
  labs(title = "Clusters visualized by the first 2 Principal Components",
       subtitle = "I pretend I do not know the labels",
       x = "PC1",
       y = "PC2")


## ---- fig.height = 4, fig.width = 6,fig.align='center', echo = T---------
set.seed(20200216)
fviz_cluster(kmeans(scale(data_features), centers = 2, nstart = 100), 
             data = scale(data_features), 
             show.clust.cent = T, 
             ellipse.type = "norm") +
  theme_minimal() +
  labs(title = "2 clusters created on the dataset",
       subtitle = "(We know the red cluster contains the diseased ones, the green the healthy ones)")


## ---- fig.height = 4, fig.width = 6,fig.align='center', echo = T---------
fviz_pca_ind(prcomp(data_features, scale. = T))


## ------------------------------------------------------------------------
summary(prcomp(data_features, scale. = T))


## ---- results = 'asis'---------------------------------------------------
pc1_loadings <- data.frame("features" = rownames(prcomp(data_features, scale. = T)$rotation),
                           "loadings" = prcomp(data_features, scale. = T)$rotation[colnames(prcomp(data_features, scale. = T)$rotation) == "PC1"])

kable(pc1_loadings %>% arrange(desc(abs(loadings))) %>% head(2))


## ---- fig.height = 3, fig.width = 8, fig.align='center', echo = F--------
ggplot(data %>% select(measure_603, measure_179, is_diseased) %>% gather(key = "measure" ,value = "value", measure_603, measure_179), aes(value, is_diseased)) +
  geom_point(aes(color = is_diseased), size = 5, show.legend = F) +
  labs(title = "Outcome in dimensions of most important features of 1st PC",
       x = "Measures",
       y = "Labels") +
  facet_wrap(~measure)


## ---- fig.height = 4, fig.width = 6,fig.align='center', echo = F---------
ggplot(data, aes(measure_179, measure_603)) +
  geom_point(aes(color = is_diseased), size = 5) +
  geom_smooth(method = "loess", color = "black", size = 1, se = F) +
  labs(title = "Outcome in dimensions of most important features of 1st PC",
       x = "measure_179",
       y = "measure_603")


