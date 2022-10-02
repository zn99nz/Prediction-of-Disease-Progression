# data loading and split into train and test set
als <- read.table('C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\ALS.txt', header = T)
test <- als[which(als$testset == T), -1]
train <- als[which(als$testset == F), -1]

# random seed = 2018150457
seed <- 2018150457

# required packages
library(randomForest) # random forest
library(gbm) # boosting
library(glmnet) # adaptive lasso 
library(ncvreg) # SCAD-penalized regression
library(caret) # for variable importance

# 1. Random Forest
set.seed(seed)
cv.rf.1 <- rfcv(as.matrix(train[,-1]), as.matrix(train$dFRS),
                mtry = function(p) max(1, floor(p/3)),
                ntree = 1000, cv.fold = 10)

mtry <- rep(0,9)
for(i in 1:9){mtry[i] <- max(1, floor(cv.rf.1$n.var[i]/3))}
# mtry 1이 두번 : 둘의 mse를 평균냄

# Table 3.1
matrix(round(c(mean(cv.rf.1$error.cv[8:9]),
             cv.rf.1$error.cv[7:1])*1000,3),
       1, 8, dimnames = list(c('MSE'), mtry[8:1]), byrow = T)

set.seed(seed)
rf.fit <- randomForest(x = as.matrix(train[,-1]),
                       y = as.matrix(train$dFRS),
                       mtry = 30, ntree = 1000)

#####################################################################
#####################################################################
#####################################################################

# 2. Boosting
e.list <- seq(0.005, 0.1, length = 10)
d.list <- c(2, 4, 7)
B.mat <- matrix(0, 10, 3)
mse.mat <- matrix(0, 10, 3)
for (i in 1:10){
  for (j in 1:3){
    set.seed(seed)
    cv.gbm.1 <- gbm(dFRS ~ ., data = train, distribution = "gaussian", 
                    n.trees = 1500, shrinkage = e.list[i],            
                    interaction.depth = d.list[j],
                    train.fraction = 0.5, bag.fraction = 1) 
    
    best <- gbm.perf(cv.gbm.1, method = 'test', plot.it = F)
    B.mat[i, j] <- best
    mse.mat[i, j] <- cv.gbm.1$valid.error[best]
    print(c(i, j))
  }
}
round(e.list, 3)

boosting_cv <- NULL
for(i in 1:3){boosting_cv <- cbind(boosting_cv, 
                                   cbind(round(mse.mat*1000, 3)[,i],
                                         B.mat[,i]))}
# write.csv(boosting_cv, 'C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\boosting_cv.csv')
# table 3.2
boosting_cv

min1 <- apply(mse.mat, 2, which.min)
min1
min.list <- rep(0,3)
for (i in 1:3) {min.list[i] <- mse.mat[min1[i], i]}
best.d <- d.list[which.min(min.list)] 
best.e <- e.list[min1[which.min(min.list)]] 

best.mse <- mse.mat[min1[which.min(min.list)], which.min(min.list)]
best.B <- B.mat[min1[which.min(min.list)], which.min(min.list)]

set.seed(seed)
gbm.fit <- gbm(dFRS ~ ., data = train, distribution = "gaussian", 
               n.trees = 500, shrinkage = best.e,            
               interaction.depth = best.d,
               train.fraction = 0.5, bag.fraction = 1)

#####################################################################
#####################################################################
#####################################################################

# 3. Adaptive Lasso
ols <- lm(dFRS ~ ., data = train)$coefficient[-1]

min(abs(ols[which(!is.na(ols))]))
2.310553e-06 == 2310.553e-09
abs(2.310553e-06)^(-1)

n <- nrow(train)
penalty <- (abs(ols)^(-1))
sum(is.na(penalty)) # ols값에 NA가 존재 -> 해당 변수에는 매우 큰 패널티 적용

max.pe <- max(penalty[is.na(penalty) == F])
max.pe
penalty.1 <- ifelse(is.na(penalty), penalty <- max.pe, penalty <- penalty)
sum(is.na(penalty.1)) # NA 제거 성공

set.seed(seed)
cv.adalasso.2 <- cv.glmnet(as.matrix(train[,-1]), as.matrix(train$dFRS),
                           lambda = exp(seq(-6, -1.5, length = 100)), 
                           penalty.factor = abs(penalty.1)^(-1), 
                           type.measure = 'mse')

# fig 3.1
windows(10, 10)
plot(cv.adalasso.2, ylab = 'MSE')
abline(v = log(cv.adalasso.2$lambda.min), lwd = 1, lty = 'dotted', 
       col = 'white')
abline(v = log(cv.adalasso.2$lambda.min), lwd = 2, lty = 2, 
       col = 'blue')


# table 3.3
adalasso_cv <- round(cbind(log(cv.adalasso.2$lambda), cv.adalasso.2$cvm*1000), 3)
adalasso_cv
# write.csv(adalasso_cv, 'C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\adalasso_cv.csv')

set.seed(seed)
adalasso.fit <- glmnet(as.matrix(train[,-1]), as.matrix(train$dFRS),
                       lambda = cv.adalasso.2$lambda.min, 
                       penalty.factor = abs(penalty.1)^(-1), 
                       type.measure = 'mse')

#####################################################################
#####################################################################
#####################################################################

# 4. SCAD-penalized regression
set.seed(seed)
cv.scad.1 <- cv.ncvreg(as.matrix(train[,-1]), as.matrix(train$dFRS),
                       family = 'gaussian', penalty = 'SCAD')

# fig 3.2
windows(10, 10)
plot(cv.scad.1, ylab = 'MSE')
abline(v = log(cv.scad.1$lambda.min), lwd = 1, lty = 1, 
       col = 'white')
abline(v = log(cv.scad.1$lambda.min), lwd = 2, lty = 2, 
       col = 'blue')

log(cv.scad.1$lambda.min)

# table 3.4
scad_cv <- round(cbind(log(cv.scad.1$lambda), cv.scad.1$cve*1000), 3)
scad_cv
# write.csv(scad_cv, 'C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\scad_cv.csv')



set.seed(seed)
scad.fit <- ncvreg(as.matrix(train[,-1]), as.matrix(train$dFRS),
                   family = 'gaussian', penalty = 'SCAD',
                   lambda = cv.scad.1$lambda.min)

#####################################################################
#####################################################################
#####################################################################

# 5. Elastic net
alpha.list.1 <- seq(0.1, 0.9, length = 5)
tuning.mat.1 <- matrix(0, 5, 2, dimnames = list(as.character(alpha.list.1),
                                                c('lambda', 'mse')))
for (i in 1:5){
  set.seed(seed)
  cv.elastic.1 <- cv.glmnet(as.matrix(train[,-1]), as.matrix(train$dFRS),
                            lambda = exp(seq(-6, -1.5, length = 100)), 
                            type.measure = 'mse', 
                            alpha = alpha.list.1[i])
  
  tuning.mat.1[i, 1] <- cv.elastic.1$lambda.min
  tuning.mat.1[i, 2] <- min(cv.elastic.1$cvm)
}

which.min(tuning.mat.1[,2])
log(tuning.mat.1[,1])

best.a <- alpha.list.1[which.min(tuning.mat.1[,2])]
best.l <- tuning.mat.1[which.min(tuning.mat.1[,2]), 1]


tuning.mat.2 <- round(cbind(log(tuning.mat.1[,1]),
                            tuning.mat.1[,2]*1000), 3)
# table 3.5
tuning.mat.2

set.seed(seed)
elastic.fit <- glmnet(as.matrix(train[,-1]), as.matrix(train$dFRS),
                      lambda = best.l, alpha = best.a,
                      type.measure = 'mse')

#####################################################################
#####################################################################
#####################################################################

boot.mat <- matrix(0, 200, 5)
colnames(boot.mat) <- c('RF', 'Boosting', 'AdaLasso', 'SCAD', 'Elastic')

for (i in 1:200){
  # bootstrap
  set.seed(i)
  b.row <- sample(nrow(train), nrow(train), replace = T)
  boot <- train[b.row,]
  
  # 1. Random Forest
  rf.boot <- randomForest(x = as.matrix(boot[,-1]),
                         y = as.matrix(boot$dFRS),
                         mtry = 30, ntree = 1000)
  pre.y1 <- predict(rf.boot, as.matrix(test[,-1]))
  mse.rf <- mean((test$dFRS - pre.y1)^2)
  boot.mat[i, 1] <- mse.rf
  
  # 2. Boosting
  gbm.boot <- gbm(dFRS ~ ., data = boot, distribution = "gaussian", 
                 n.trees = 200, shrinkage = best.e,            
                 interaction.depth = best.d,
                 train.fraction = 0.5, bag.fraction = 1)
  
  pre.y2 <- predict(gbm.boot, test[,-1], n.tree = best.B)
  mse.gbm <- mean((test$dFRS - pre.y2)^2)
  boot.mat[i, 2] <- mse.gbm
  
  # 3. Adaptive Lasso
  adalasso.boot <- glmnet(as.matrix(boot[,-1]), as.matrix(boot$dFRS),
                         lambda = cv.adalasso.2$lambda.min, 
                         penalty.factor = abs(penalty.1)^(-1), 
                         type.measure = 'mse')
  pre.y3 <- predict.glmnet(adalasso.boot, as.matrix(test[,-1]))
  mse.adalasso <- mean((test$dFRS - pre.y3)^2)
  boot.mat[i, 3] <- mse.adalasso
  
  # 4. SCAD-penalized regression
  scad.boot <- ncvreg(as.matrix(boot[,-1]), as.matrix(boot$dFRS),
                     family = 'gaussian', penalty = 'SCAD',
                     lambda = cv.scad.1$lambda.min)
  pre.y4 <- predict(scad.boot, as.matrix(test[,-1]))
  mse.scad <- mean((test$dFRS - pre.y4)^2)
  boot.mat[i, 4] <- mse.scad
  
  # 5. Elastic net
  elastic.boot <- glmnet(as.matrix(boot[,-1]), as.matrix(boot$dFRS),
                        lambda = best.l, alpha = best.a,
                        type.measure = 'mse')
  pre.y5 <- predict(elastic.boot, as.matrix(test[,-1]))
  mse.elastic <- mean((test$dFRS - pre.y5)^2)
  boot.mat[i, 5] <- mse.elastic
  
  print(i)
}

mean <- apply(boot.mat, 2, mean)
se <- apply(boot.mat, 2, sd)/sqrt(200)
up <- mean+2*se
lo <- mean-2*se

# table 4.2
comparision.mat <- round(matrix(c(mean, se, lo, up)*1000, 5, 4), 3)
# write.csv(comparision.mat, 'C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\comparison_mat.csv')

# fig 4.1
plot(1, 1, xlim = c(1, 5), ylim = c(0.273, 0.303)*1000, 'n', 
     ylab = '평균 MSE', xlab = 'model')
for (i in 1:5){
  lines(rep(i, 2), c(up[i], lo[i])*1000, lty = 2, lwd = 2,col = 'gray')
  text(i, up[i]*1000, '-', cex = 2)
  text(i, lo[i]*1000, '-', cex = 2)
}
points(1:5, mean*1000, 'p', pch = 19, col = 'steelblue', cex = 1.5)


#####################################################################
#####################################################################
#####################################################################
# variable importance
# fig 4.2
rf.varimp <- as.matrix(varImp(rf.fit))
windows(40, 20)
par(mai = c(0.5, 2.7, 0, 0.5))
barplot(sort(rf.varimp[,1], decreasing = T), 
        horiz = T, las = 1,
        col = 'steelblue', border = 'steelblue', cex.names = 0.7)
abline(h = 20, lwd = 2, lty = 2, col = 'tomato')


# rf
sort(rf.varimp[,1], decreasing = T)
sum(rf.varimp[,1] != 0) # 369
rf_var_20 <- sort(rf.varimp[,1], decreasing = T)[1:20]

# table 4.3
round(as.matrix(rf_var_20),3)
# write.csv(round(as.matrix(rf_var_20),3), 'C:\\Users\\uccoo\\Desktop\\학교\\4-1\\통계계산방법론\\Final\\rf_var_20.csv')

# boosting
gbm.var <- as.matrix(varImp(gbm.fit, numTrees = best.B))
sum(gbm.var[,1] != 0) # 83
boosting.nonzero <- which(gbm.var[,1] != 0)
boosting_var <- names(gbm.var[boosting.nonzero, 1])
boosting.zero <- which(gbm.var[,1] == 0)
which(names(rf_var_20) %in% colnames(train)[boosting.zero+1])

# adalasso
sum(adalasso.fit$beta != 0) # 51
adalasso.nonzero <- which(as.matrix(adalasso.fit$beta)[,1] != 0)
adalasso_var <- names(as.matrix(adalasso.fit$beta)[adalasso.nonzero, 1])
adalasso.zero <- which(as.matrix(adalasso.fit$beta)[,1] == 0)
which(names(rf_var_20) %in% colnames(train)[adalasso.zero+1])

# scad
sum(scad.fit$beta[-1] != 0) # 37
scad.fit.beta <- as.matrix(scad.fit$beta)[-1,]
scad.nonzero <- scad.fit.beta != 0
scad_var <- names(scad.fit.beta[scad.nonzero])
which(names(rf_var_20) %in% colnames(train)[-c(F, scad.nonzero)])

# elastic
sum(elastic.fit$beta != 0) # 51
elastic.nonzero <- as.matrix(elastic.fit$beta)[,1] != 0
elastic_var <- names(as.matrix(elastic.fit$beta)[elastic.nonzero, 1])
which(names(rf_var_20) %in% colnames(train)[-c(F, elastic.nonzero)])


sum(adalasso_var %in% elastic_var) # 완전 일치
n_var <- scad_var[scad_var %in% elastic_var]
n_var1 <- n_var[n_var %in% boosting_var]
n_var2 <- n_var1[n_var1 %in% rf_var_50]

# rf_var_20와 비교 (table 4.3 진한 표시)
used <- which(colnames(train) %in% union(union(adalasso_var, boosting_var), scad_var))
which(names(rf_var_20) %in% colnames(train)[-used])
names(rf_var_20)[which(names(rf_var_20) %in% colnames(train)[-used])]

###############################################################
# Reference
toBibtex(citation(package = "randomForest", lib.loc = NULL, auto = NULL))
toBibtex(citation(package = "gbm", lib.loc = NULL, auto = NULL))
toBibtex(citation(package = "glmnet", lib.loc = NULL, auto = NULL))
toBibtex(citation(package = "ncvreg", lib.loc = NULL, auto = NULL))
toBibtex(citation(package = "caret", lib.loc = NULL, auto = NULL))
toBibtex(citation(package = "stats", lib.loc = NULL, auto = NULL))

