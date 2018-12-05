squad = read.csv("/Users/Wesley/Desktop/BIOSTAT273/biostat-273-fall-2018/predictions/squad.csv", header = FALSE)
solo = read.csv("/Users/Wesley/Desktop/BIOSTAT273/biostat-273-fall-2018/predictions/solo.csv", header = FALSE)
custom = read.csv("/Users/Wesley/Desktop/BIOSTAT273/biostat-273-fall-2018/predictions/Custom_group_predictions.csv", header = FALSE)
duo = read.csv("/Users/Wesley/Desktop/BIOSTAT273/biostat-273-fall-2018/predictions/prediction_duo.csv", header = FALSE)

final = rbind(squad, solo, custom, duo)
colnames(final) = c("Id", "winPlacePerc")
write.table(final, file = "/Users/Wesley/Desktop/BIOSTAT273/biostat-273-fall-2018/predictions/final.csv", row.names = FALSE, sep = ",")
