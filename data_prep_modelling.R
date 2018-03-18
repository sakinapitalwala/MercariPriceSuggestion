#Librarires
library('data.table') #For fread
library('tictoc') #For time elapsed for each blocl of code
library('stringr') #For str_split_fixed
library('tm') #For DTM
library('dummies') #For one hot encoding
library('xgboost') #For modelling
library('sampling') #For stratified sampling

tic()
#load data
train <- fread(input = "../input/train.tsv", header=TRUE, stringsAsFactors = FALSE, sep='\t')
test <- fread(input = "../input/test.tsv", header=TRUE, stringsAsFactors = FALSE, sep='\t')
train <- data.frame(train)
test <- data.frame(test)

train$price <- with(train, ifelse(price>0, log(price), price))

stratas <- strata(train, c("item_condition_id"), size = c(100000,100000,100000,10000,1500), method = "srswor")
train <- getdata(train, stratas)
train <- train[,-which(names(train) %in% c("ID_unit","Prob","Stratum"))]
toc()



tic()
price_train <- train[,"price"]
train <- train[, -which(names(train) %in% c("price"))]
train_num_rows <- dim(train)[1]
test_last_id <- dim(test)[1]
colnames(train)[1] <- "id"
colnames(test)[1] <- "id"
train_test <- rbind(train, test)
rm(train)
rm(test)
gc()
toc()


#Missing values
tic()
train_test$brand_name[train_test$brand_name=="" | is.na(train_test$brand_name)] <- "Missing_Brand"
train_test$item_description[train_test$item_description=="No description yet" | train_test$item_description==""] <- "NoDescYet"
toc()


#DTM for item_description
#Form corpus
tic()
corpus <- Corpus(VectorSource(train_test$item_description))
toc()
#Remove columns from train data
tic()
train_test <- train_test[, -which(names(train_test) %in% c("item_description"))]
toc()
#Clean data
tic()
corpus <- tm_map(corpus,tolower)
toc()
tic()
corpus <- tm_map(corpus, removeNumbers)
toc()
tic()
corpus <- tm_map(corpus, removePunctuation)
toc()
tic()
corpus <- tm_map(corpus, removeWords, stopwords('english'))
toc()
tic()
corpus <- tm_map(corpus, stemDocument, "english")
toc()
tic()
corpus <- tm_map(corpus, stripWhitespace)
toc()
tic("Document Term Matrix")
dtm <- DocumentTermMatrix(corpus)
toc()
rm(corpus)
gc()
tic()
dtm <- removeSparseTerms(dtm, 0.99)
toc()
tic()
train_test <- data.frame(train_test, as.matrix(dtm))
toc()
rm(dtm)
gc()




#DTM for name
#Form corpus
tic()
name_corpus <- Corpus(VectorSource(train_test$name))
toc()
tic()
train_test <- train_test[, -which(names(train_test) %in% c("name"))]
toc()
#Clean data
tic()
name_corpus <- tm_map(name_corpus, tolower)
toc()
tic()
name_corpus <- tm_map(name_corpus, removeNumbers)
toc()
tic()
name_corpus <- tm_map(name_corpus, removePunctuation)
toc()
tic()
name_corpus <- tm_map(name_corpus, removeWords, stopwords("english"))
toc()
tic()
name_corpus <- tm_map(name_corpus, stripWhitespace)
toc()
tic()
dtm_name <- DocumentTermMatrix(name_corpus)
toc()
rm(name_corpus)
gc()
tic()
dtm_name <- removeSparseTerms(dtm_name, 0.99)
toc()
tic()
train_test <- data.frame(train_test, as.matrix(dtm_name))
toc()
tic()
rm(dtm_name)
gc()
toc()




tic()
train_test$item_condition_id <- as.factor(train_test$item_condition_id)
train_test$shipping <- as.factor(train_test$shipping)
gc()
toc()


#Separate out the three levels of category
tic()
train_test$category_name[train_test$category_name == "" | is.na(train_test$category_name)] <- "Missing1/Missing2/Missing3"
train_test$level1_category <- str_split_fixed(train_test$category_name, "/" ,3)[,1]
train_test$level2_category <- str_split_fixed(train_test$category_name, "/" ,3)[,2]
train_test$level3_category <- str_split_fixed(train_test$category_name, "/" ,3)[,3]
toc()


#Convert all data to numeric
#1. Convert brand name to numeric
#select top 20 brands based on frequency
tic()
top_brands <- sort(table(train_test$brand_name), decreasing = T)[1:21]
top_brands <- data.frame(top_brands)
top_brands_list <- top_brands$Var1
train_test$brand_name <- as.factor(ifelse(train_test$brand_name %in% top_brands_list, train_test$brand_name, "other_brand"))
Brands <- dummy(train_test$brand_name)
toc()

train_test <- data.frame(train_test, Brands)
rm(top_brands)
rm(top_brands_list)
rm(Brands)
gc()


#Convert level1_category to numeric
tic()
train_test$level1_category <- as.factor(train_test$level1_category)
Level1_category <- dummy(train_test$level1_category)
train_test <- data.frame(train_test, Level1_category)
rm(Level1_category)
gc()
toc()


#Convert level2_category to numeric
tic()
#Select top 20 level 2 categories based on frequency
top_l2_cat <- sort(table(train_test$level2_category), decreasing = T)[1:21]
top_l2_cat <- data.frame(top_l2_cat)
top_l2_cat_list <- top_l2_cat$Var1
train_test$level2_category <- as.factor(ifelse(train_test$level2_category %in% top_l2_cat_list, train_test$level2_category, "other_l2_cat"))
Level2_category <- dummy(train_test$level2_category)
train_test <- data.frame(train_test, Level2_category)
rm(Level2_category)
rm(top_l2_cat)
rm(top_l2_cat_list)
gc()
toc()


#Convert level3_category to numeric
tic()
#Select top 20 level 3 categories based on frequency
top_l3_cat <- sort(table(train_test$level3_category), decreasing = T)[1:21]
top_l3_cat <- data.frame(top_l3_cat)
top_l3_cat_list <- top_l3_cat$Var1
train_test$level3_category <- as.factor(ifelse(train_test$level3_category %in% top_l3_cat_list, train_test$level3_category, "other_l3_cat"))
Level3_category <- dummy(train_test$level3_category)
train_test <- data.frame(train_test, Level3_category)
rm(Level3_category)
rm(top_l3_cat)
rm(top_l3_cat_list)
gc()
toc()


#Convert item_condition_id to one hot encoding
tic()
Item_condition_id <- dummy(train_test$item_condition_id)
train_test <- data.frame(train_test, Item_condition_id)
rm(Item_condition_id)
gc()
toc()


#Remove columns not required
train_test <- train_test[, -which(names(train_test) %in% c("brand_name", "level1_category", "level2_category", "level3_category", "category_name", "item_condition_id"))]
#train_test <- subset(train_test, select = -c(brand_name, level1_category, level2_category, level3_category, category_name))
gc()


tic()
#Split train and test again
train <- train_test[1:train_num_rows,]
test <- train_test[train_num_rows + 1:test_last_id,]
dim(train)
dim(test)
rm(train_test)
gc()
toc()


#Build model
tic()
train$shipping <- as.numeric(train$shipping)
toc()
tic()
model <- xgboost(data = as.matrix(train[,-c(1)]), label = price_train, nrounds = 70, objective = "reg:linear")
toc()
tic()
test$shipping <- as.numeric(test$shipping)
toc()
tic()
predicted_price <- predict(model, as.matrix(test[,-c(1)]))
toc()


#Write results into csv
predicted_price <- data.frame(exp(predicted_price))
csv_output <- data.frame(test$id, predicted_price)
colnames(csv_output) <- c("test_id", "price")
write.csv(csv_output, "output.csv", row.names = F)
