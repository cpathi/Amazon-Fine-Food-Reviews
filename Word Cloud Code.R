install.packages(tm)
install.packages("RColorBrewer")
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
setwd("C:/Users/pchak/Desktop/Projects/amazon-fine-food-reviews")
df <- read.csv('Reviews.csv')
df$Score[df$Score<=3]=0
df$Score[df$Score>=4]=1
df1=df[df$Score == '1',]
df2=df[df$Score == '0',]


#Positive Reviews cloud code
df_sai <- VCorpus(VectorSource(df1$Text))
df_sai <- tm_map(df_sai, PlainTextDocument)
df_sai <- tm_map(df_sai, removePunctuation)
df_sai <- tm_map(df_sai, removeWords, stopwords('english'))
#df_sai <- tm_map(df_sai, stemDocument)
df_sai <- tm_map(df_sai, removeWords, c(stopwords('english')))
wordcloud(df_sai,max.words = 100,random.order = FALSE, colors=brewer.pal(8, "Dark2"))

#Negative Reviews cloud code
df_sai <- VCorpus(VectorSource(df2$Text))
df_sai <- tm_map(df_sai, PlainTextDocument)
df_sai <- tm_map(df_sai, removePunctuation)
df_sai <- tm_map(df_sai, removeWords, stopwords('english'))
#df_sai <- tm_map(df_sai, stemDocument)
df_sai <- tm_map(df_sai, removeWords, c(,'the','like','love','good','well','great',stopwords('english')))
wordcloud(df_sai,max.words = 100,random.order = FALSE, colors=brewer.pal(8, "Dark2"))