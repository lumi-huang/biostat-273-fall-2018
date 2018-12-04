library(tidyverse)
library(corrplot)
library(glmnet)
library(randomForest)
library(xgboost)


setwd("C:/Users/Darrick Shen/Desktop")
pubg_tr <- read_csv("train_V2.csv")
#cor(pubg_tr[,covariates], use = "pairwise.complete.obs") %>% corrplot(t1.cex = .75)

#drop single NA row
pubg_tr <- pubg_tr %>% drop_na()

#filter duo
pubg_tr <- pubg_tr %>%
  filter(matchType == 'flarefpp'| matchType == 'flaretpp' | 
           matchType == 'crashfpp' | matchType == 'crashtpp')
nrow(pubg_tr)
head(pubg_tr)

head(pubg_tr$winPoints)
head(pubg_tr$killPoints)

#modify the winpoint and kill point based on rank point
pubg_tr$winPoints = ifelse(pubg_tr$winPoints == 0 & pubg_tr$rankPoints != -1, NA, pubg_tr$winPoints)
pubg_tr$killPoints = ifelse(pubg_tr$killPoints == 0 & pubg_tr$rankPoints != -1, NA, pubg_tr$killPoints)

pubg_group <- pubg_tr %>%
  select(Id, groupId, matchId, assists, boosts, damageDealt, DBNOs, headshotKills,
         heals, killPlace, killPoints, kills, killStreaks, longestKill, maxPlace, numGroups,
         revives, rideDistance, roadKills, swimDistance, teamKills, vehicleDestroys, walkDistance,
         weaponsAcquired, winPoints, winPlacePerc, matchDuration) %>%
  group_by(groupId, matchId) %>%
  summarise(matchId = max(matchId, na.rm = TRUE), assists = sum(assists, na.rm = TRUE),
            boosts = sum(boosts, na.rm = TRUE), damageDealt = sum(damageDealt, na.rm = TRUE), 
            DBNOs = sum(DBNOs, na.rm = TRUE), headshotKills = sum(headshotKills, na.rm = TRUE),
            heals = sum(heals, na.rm = TRUE), killPlace = sum(killPlace, na.rm = TRUE), 
            killPoints = ifelse(is.finite(max(killPoints, na.rm = TRUE)), max(killPoints), NA), kills = sum(kills, na.rm = TRUE), 
            killStreaks = mean(killStreaks, na.rm = TRUE), longestKill = mean(longestKill, na.rm = TRUE),
            maxPlace = max(maxPlace, na.rm = TRUE), numGroups = mean(numGroups, na.rm = TRUE), 
            revives = sum(revives, na.rm = TRUE), rideDistance = mean(rideDistance, na.rm = TRUE), 
            roadKills = sum(roadKills, na.rm = TRUE), swimDistance = mean(swimDistance, na.rm = TRUE),
            teamKills = sum(teamKills, na.rm = TRUE), vehicleDestroys = sum(vehicleDestroys, na.rm = TRUE), 
            walkDistance = mean(walkDistance, na.rm = TRUE), 
            weaponsAcquired = sum(weaponsAcquired, na.rm = TRUE), 
            winPoints = ifelse(is.finite(max(winPoints, na.rm = TRUE)), max(winPoints), NA), winPlacePerc = mean(winPlacePerc, na.rm = TRUE),
            matchDuration = mean(matchDuration, na.rm = TRUE))

# Adding features and dropping component features
custom_group <- pubg_group %>%
  mutate(killsAssistsByDistance= (kills + assists)/(walkDistance + 1), 
         killAssistsByDuration = (kills + assists)/(matchDuration + 1),
         itemsByDistance = (boosts + heals + weaponsAcquired) / (walkDistance + 1),
         itemsByDuration = (boosts + heals + weaponsAcquired)/ (matchDuration + 1)) %>%
  select(-kills, -assists, -boosts, -heals, -weaponsAcquired, -killPoints, -winPoints)


# Seperate ID and variables
ids <- colnames(custom_group)[grepl("Id", colnames(custom_group))] #grab ID names
variables <- colnames(custom_group)[!(grepl("Id", colnames(custom_group)))] #grab variable names


#killpoints win pots drop ?
write_csv(custom_group, "C:/Users/Darrick Shen/Desktop/train_custom_group.csv")


####DONE READING IN DATA#####

#################### Prediction Helper Functions #################### 
confine_predictions = function(x){
  x = ifelse(x > 1, 1, x)
  x = ifelse(x < 0, 0, x)
}


MAE = function(yhat, y){
  return(abs(yhat - y) %>% mean)
}
pubg_group %>% count(teamsize)
pubg_group %>% count(groupId)

###################################################
#Sample the train dataset.
set.seed(1)
row.number <- sample(1:nrow(custom_group), 0.8 * nrow(custom_group))

custom_group_train <- custom_group[row.number, ]
custom_group_test <- custom_group[-row.number, ]
dim(custom_group_train) + dim(custom_group_test) # 4205 obs.
#dim(train) + dim(test)
#dim(custom_group)


#################### 1. Linear Regression #################### 

ggplot(custom_group_train, aes(winPlacePerc)) + geom_density(fill="blue")

#We will start by taking all input variables in the multiple regression.
m_linear = lm(winPlacePerc ~ ., data = custom_group_train[,variables])

summary(m_linear) #maxplace, numGroups, revives are nonsignificant. F test overall is significant
par(mfrow=c(2,2))
plot(m_linear)


p_linear <- predict(m_linear, select(custom_group_test, variables, -winPlacePerc)) %>% confine_predictions
MAE(p_linear, custom_group_test$winPlacePerc)

#################### 2. Ridge Regression #################### 
lambdas <- 10 ^ seq(3, -2, by = -.1)

# Define training variables and outcome variable
x <- select(custom_group_train, variables, -winPlacePerc) %>% data.matrix()
y <- custom_group_train$winPlacePerc

## Cross Validation to get optimal lambda for ridge regression
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min

m_ridge <- glmnet(x, y, alpha = 0, lambda = opt_lambda)
# Predict using held out test variables and outcome variable
p_ridge <-  cbind(rep(1, nrow(custom_group_test)), select(custom_group_test, variables, -winPlacePerc)) %>% as.matrix %*% 
  as.matrix(coef(m_ridge)) %>% confine_predictions()

colnames(p_ridge) = c("p_ridge")

MAE(p_ridge, custom_group_test$winPlacePerc)

#################### 3. Lasso Regression #################### 
lambdas <- 10 ^ seq(3, -2, by = -.1)

## Cross Validation to get optimal lambda for Lasso regression
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min

m_lasso <- glmnet(x, y, alpha = 1, lambda = opt_lambda)
p_lasso <- cbind(rep(1, nrow(custom_group_test)), select(custom_group_test, variables, -winPlacePerc)) %>% as.matrix %*% 
  as.matrix(coef(m_lasso)) %>% confine_predictions()

colnames(p_lasso) = c("p_lasso")

MAE(p_lasso, custom_group_test$winPlacePerc)

summary(p_lasso)

#################### 4. Random Forest ####################
m_forest_100 <- randomForest(custom_group_train[,-c(1, 2, 18)], custom_group_train$winPlacePerc, ntree = 100, do.trace = TRUE)
m_forest_250 <- randomForest(custom_group_train[,-c(1, 2, 18)], custom_group_train$winPlacePerc, ntree = 250, do.trace = TRUE)
m_forest_500 <- randomForest(custom_group_train[,-c(1, 2, 18)], custom_group_train$winPlacePerc, ntree = 500, do.trace = TRUE)

p_forest <- predict(m_forest_500, custom_group_test[,variables])
MAE(p_forest, custom_group_test$winPlacePerc)

custom_group_train[,18]

#################### 5. Boosting ####################
m_boost <- xgboost(as.matrix(select(custom_group_train, variables, -winPlacePerc)), custom_group_train$winPlacePerc, nrounds = 100)
p_boost <- predict(m_boost, as.matrix(select(custom_group_test, variables, -winPlacePerc))) %>% confine_predictions
MAE(p_boost, custom_group_test$winPlacePerc)


## Merge back with Original Dataset ##
results_table <- cbind(custom_group_test %>% select(matchId, groupId), p_boost, p_forest, p_lasso, p_ridge, p_linear)
complete_table <- left_join(pubg_tr, results_table, by = c("matchId", "groupId"))
id_prediction_table <- filter(complete_table, !is.na(p_boost)) 


MAE_final <- MAE(id_prediction_table$p_boost, id_prediction_table$winPlacePerc)
