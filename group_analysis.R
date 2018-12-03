library(tidyverse)
library(corrplot)
library(glmnet)
library(randomForest)
library(xgboost)

pudg_tr = read_csv("/Users/Wesley/Desktop/BIOSTAT273/train_V2.csv")

##### EDA #####
cor(pudg_tr[,covariates], use = "pairwise.complete.obs") %>% corrplot(t1.cex = .75)


##### Data Preprocessing #####
# Remove single NA row
pudg_tr =  pudg_tr %>% drop_na()

# Clean up winPoints and killPoints (not using in model right now hoever)
pudg_tr$winPoints = ifelse(pudg_tr$winPoints == 0 & pudg_tr$rankPoints != -1, NA, pudg_tr$winPoints)
pudg_tr$killPoints = ifelse(pudg_tr$killPoints == 0 & pudg_tr$rankPoints != -1, NA, pudg_tr$killPoints)


# Filter by matchType (squad) and perform aggregrations
squad = pudg_tr %>% filter(matchType == 'squad'| matchType == 'normal-squad-fpp' | 
                       matchType == 'normal-squad' | matchType == 'duo-squad')

squad_group = squad %>%
  select(Id, groupId, matchId, assists, boosts, damageDealt, DBNOs, headshotKills,
         heals, killPlace, killPoints, kills, killStreaks, longestKill, matchDuration, maxPlace, numGroups,
         revives, rideDistance, roadKills, swimDistance, teamKills, vehicleDestroys, walkDistance,
         weaponsAcquired, winPoints, winPlacePerc) %>%
  group_by(groupId) %>%
  summarise(teamsize = n(), matchId = max(matchId, na.rm = TRUE), assists = sum(assists, na.rm = TRUE),
          boosts = sum(boosts, na.rm = TRUE), damageDealt = sum(damageDealt, na.rm = TRUE), 
          DBNOs = sum(DBNOs, na.rm = TRUE), headshotKills = sum(headshotKills, na.rm = TRUE),
          heals = sum(heals, na.rm = TRUE), killPlace = sum(killPlace, na.rm = TRUE), 
          killPoints = ifelse(is.finite(max(killPoints, na.rm = TRUE)), max(killPoints, na.rm = TRUE), NA), kills = sum(kills, na.rm = TRUE), 
          killStreaks = mean(killStreaks, na.rm = TRUE), longestKill = mean(longestKill, na.rm = TRUE),
          matchDuration = mean(matchDuration, na.rm = TRUE), 
          maxPlace = max(maxPlace, na.rm = TRUE), numGroups = mean(numGroups, na.rm = TRUE), 
          revives = sum(revives, na.rm = TRUE), rideDistance = max(rideDistance, na.rm = TRUE), 
          roadKills = sum(roadKills, na.rm = TRUE), swimDistance = max(swimDistance, na.rm = TRUE),
          teamKills = sum(teamKills, na.rm = TRUE), vehicleDestroys = sum(vehicleDestroys, na.rm = TRUE), 
          walkDistance = max(walkDistance, na.rm = TRUE), 
          weaponsAcquired = sum(weaponsAcquired, na.rm = TRUE), 
          winPoints = ifelse(is.finite(max(winPoints, na.rm = TRUE)), max(winPoints, na.rm = TRUE), NA) , winPlacePerc = mean(winPlacePerc, na.rm = TRUE))

# Add aggregate features here
squad_group = squad_group %>% mutate(killsAssistsByDistance = (kills + assists) / (walkDistance + 1),
                                     killsAssistsByDuration = (kills + assists) / (matchDuration + 1),
                                     itemsByDistance = (boosts + heals + weaponsAcquired) / (walkDistance + 1), 
                                     itemsByDuration = (boosts + heals + weaponsAcquired) / (matchDuration + 1)) %>%
                              select(-kills, - assists, -boosts, - heals, - weaponsAcquired, -killPoints, -winPoints)


# Seperate ID and variables
ids = colnames(squad_group)[grepl("Id", colnames(squad_group))]
variables = colnames(squad_group)[!(grepl("Id", colnames(squad_group)))]



################################################################################################################################################################


##### Data Prep for Model Building #####
rows = 1:nrow(squad_group)
train_rows = sample(rows, .8*length(rows), replace = FALSE)
test_rows = setdiff(rows, train_rows)

squad_group_train = squad_group[train_rows,]
squad_group_test = squad_group[test_rows,]


#################### Prediction Helper Functions #################### 
confine_predictions = function(x){
  x = ifelse(x > 1, 1, x)
  x = ifelse(x < 0, 0, x)
}


MAE = function(yhat, y){
  return(abs(yhat - y) %>% mean)
}

#################### 1. Linear Regression #################### 
m_linear = lm(winPlacePerc ~ ., data = squad_group_train[,variables])
p_linear = predict(m_linear, select(squad_group_test, variables, -winPlacePerc)) %>% confine_predictions
MAE(p_linear, squad_group_test$winPlacePerc)


#################### 2. Ridge Regression #################### 
lambdas <- 10^seq(3, -2, by = -.1)

# Define training variables and outcome variable
x = select(squad_group_train, variables, -winPlacePerc) %>% data.matrix()
y = squad_group_train$winPlacePerc

## Cross Validation to get optimal lambda for ridge regression
cv_fit = cv.glmnet(x, y, alpha = 0, lambda = lambdas)
opt_lambda = cv_fit$lambda.min

m_ridge = glmnet(x, y, alpha = 0, lambda = opt_lambda)
# Predict using held out test variables and outcome variable
p_ridge = cbind(rep(1, nrow(squad_group_test)), select(squad_group_test, variables, -winPlacePerc)) %>% as.matrix %*% 
          as.matrix(coef(m_ridge)) %>% confine_predictions()

colnames(p_ridge) = c("p_ridge")

MAE(p_ridge, squad_group_test$winPlacePerc)


#################### 3. Lasso Regression #################### 
lambdas <- 10^seq(3, -2, by = -.1)

## Cross Validation to get optimal lambda for Lasso regression
cv_fit = cv.glmnet(x, y, alpha = 1, lambda = lambdas)
opt_lambda = cv_fit$lambda.min

m_lasso = glmnet(x, y, alpha = 1, lambda = opt_lambda)
p_lasso = cbind(rep(1, nrow(squad_group_test)), select(squad_group_test, variables, -winPlacePerc)) %>% as.matrix %*% 
              as.matrix(coef(m_lasso)) %>% confine_predictions()

colnames(p_lasso) = c("p_lasso")

MAE(p_lasso, squad_group_test$winPlacePerc)

#################### 4. Random Forest ####################
m_forest_100 = randomForest(squad_group_train[,variables], squad_group_train$winPlacePerc, ntree = 100, do.trace = TRUE)
m_forest_250 = randomForest(squad_group_train[,variables], squad_group_train$winPlacePerc, ntree = 250, do.trace = TRUE)
m_forest_500 = randomForest(squad_group_train[,variables], squad_group_train$winPlacePerc, ntree = 500, do.trace = TRUE)

p_forest = predict(m_forest_500, squad_group_test[,variables])
MAE(p_forest, squad_group_test$winPlacePerc)


#################### 4. Boosting ####################
m_boost = xgboost(as.matrix(select(squad_group_train, variables, -winPlacePerc)), squad_group_train$winPlacePerc, nrounds = 100)
p_boost = predict(m_boost, as.matrix(select(squad_group_test, variables, -winPlacePerc))) %>% confine_predictions
MAE(p_boost, squad_group_test$winPlacePerc)


## Merge back with Original Dataset ##
results_table = cbind(squad_group_test %>% select(matchId, groupId), p_boost, p_forest, p_lasso, p_ridge, p_linear)
complete_table = left_join(squad, results_table, by = c("matchId", "groupId"))
id_prediction_table = filter(complete_table, !is.na(p_boost)) 


MAE_final = MAE(id_prediction_table$p_boost, id_prediction_table$winPlacePerc)
