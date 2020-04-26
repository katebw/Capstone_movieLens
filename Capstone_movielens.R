################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

edx %>% group_by(userId) %>% summarize(count = n())%>% summarize( avg = mean(count), median = median(count), st.dev. = sd(count))%>% print.data.frame()

edx %>% group_by(userId) %>% summarize(count = n())%>% filter(count <= 500)%>% ggplot(aes(count)) + geom_histogram(binwidth = 10) + labs(x= "No. of ratings", y = "No. of Users")+ theme_light()


edx %>% group_by(movieId) %>% summarize(count = n())%>% summarize( avg = mean(count), median = median(count), st.dev. = sd(count))%>% print.AsIs()

edx %>% group_by(movieId) %>% summarize(count = n())%>% filter( count <= 1000) %>% ggplot(aes(count)) + geom_histogram(binwidth = 10) + labs(x= "No. of ratings", y = "No. of Movies")+ theme_light()

library(recommenderlab)

temp <- edx %>% select(userId, movieId, rating)

m<- as(temp, "realRatingMatrix")

scheme <- evaluationScheme(m[1:200], method = "split", train = 0.9, k=1, given = 10)

algorithms <- list(
  "random items" = list(name="RANDOM", param=NULL),
  "popular items" = list(name="POPULAR", param=NULL),
  "user-based CF" = list(name="UBCF", param=list(nn=50)), 
  "SVD approximation" = list(name="SVD", param=list(k = 50))
)

results <- evaluate(scheme, algorithms, type = "ratings")
plot(results, ylim= c(0,2))

library(recosystem)

set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# remove excess information that clogs up your system
rm(test_index, temp, removed)

train_data <-  with(train_set, data_memory(user_index = userId,item_index = movieId, rating = rating))
test_data  <-  with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))

r <- recosystem::Reco()

set.seed(123, sample.kind = "Rounding")

opts <- r$tune(train_data, opts = list(nthread = 4))

r$train(train_data, opts = opts$min)

y_hat <- r$predict(test_data, out_memory())

RMSE <- function(observed, predicted){
  sqrt(mean((observed - predicted)^2))
}

RMSE(test_set$rating, y_hat)

edx_final <-  with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
valid_final <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))

r$train(edx_final, opts = opts$min)

y_hat <- r$predict(valid_final, out_memory())

print("Final RMSE:")
RMSE(validation$rating, y_hat)

