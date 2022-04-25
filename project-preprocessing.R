library("caret")
library("ggrepel")
library("scales")
library("dplyr")
library("tidyverse")
library("glmnet")
library("fastDummies")
library("lubridate")
library("parallel")
library("tibble")

CLEAN <- function(df){
print(names(df))
  
df$age <- 2015 - as.integer(df$year)
df <- select(df, -year)

df$mmr <- as.integer(df$mmr)

df$saledate <- as.POSIXlt(df$saledate, format="%a %b %d %Y %I:%M:%S GMT%z")
df$gmtoff <- unclass(df$saledate)$gmtoff
df$mon <- unclass(df$saledate)$mon + 1
df$weekday <- weekdays(df$saledate)

df$gmtoff <- as.factor(df$gmtoff)
df$mon <- as.factor(df$mon)
df$weekday <- as.factor(df$weekday)
df$saledate <- as.numeric(df$saledate)

df <- df[-which(is.na(df$saledate)),]

# drop a not-needed unique identifier
df <- select(df, -vin)
df <- df[-which(df$condition %in% c("","oh","ms","co","fl","sc","ca","pa","in","va","ga","wi","tx")),]
df$condition <- as.double(df$condition)

# drop rows with NA values
df <- na.omit(df)
df <- df[-which(df$sellingprice <= 1 | df$sellingprice > 175000)]


df$state = toupper(df$state)

df$make <- toupper(df$make)
df$make <- replace(df$make, df$make == "VW", "VOLKSWAGEN")
df$make <- replace(df$make, df$make == "LAND ROVER", "LANDROVER")
df$make <- replace(df$make, df$make %in% c("FORD TK","FORD TRUCK"), "FORD")
df$make <- replace(df$make, df$make %in% c("MERCEDES-B","MERCEDES-BENZ"), "MERCEDES")
df$make <- replace(df$make, df$make == "DODGE TK", "DODGE")
df$make <- replace(df$make, df$make == "GMC TRUCK", "GMC") 
df <- df[-which(df$make==""),]

df <- df[-which(df$transmission==""),]

df$model <- toupper(df$model)
df$trim <- toupper(df$trim)

df$hybrid <- ifelse(grepl('HYBRID',
                          str_replace_all(df$model,
                                          "[^[:alnum:]]", 
                                          "")) | 
                      grepl('PRIUS',df$model) |
                      grepl('HYBRID',df$trim) |
                      grepl('HEV',df$trim) |
                      grepl('PHEV',df$trim) |
                      df$model == 'GS 450H' |
                      df$model == 'ES 300H' |
                      df$model == 'LS 600H L' |
                      df$model == 'RX 400H' |
                      df$model == 'RX 450H' |
                      df$model == 'CT 200H' |
                      (df$make == 'SMART' & !(grepl('ELECTRIC',df$trim))), 
                    yes=1, 
                    no=0)

df$low.emission <- ifelse(grepl('PZEV',df$trim) |
                            grepl('PZEV',df$model) |
                            grepl('SULEV',df$model) |
                            grepl('SULEV',df$trim),
                          yes=1,
                          no=0)

df$electric <- ifelse(df$model %in% c('LEAF', 
                                      'B-CLASS ELECTRIC DRIVE',
                                      'IQ',
                                      'SPARK EV') |
                        grepl('BEV',df$trim) | 
                        grepl('ELECTRIC',df$trim),
                      yes=1,
                      no=0)

df$body <- toupper(str_replace_all(df$body, "[^[:alnum:]]", ""))
df$body <- replace(df$body, grepl("CAB", df$body) , "CAB")
df$body <- replace(df$body, grepl("CONVERTIBLE", df$body) , "CONVERTIBLE")
df$body <- replace(df$body, grepl("VAN", df$body) , "VAN")
df$body <- replace(df$body, grepl("COUPE", df$body) | df$body == "KOUP" , "COUPE")
df$body <- replace(df$body, grepl("SEDAN", df$body) , "SEDAN")
df$body <- replace(df$body, grepl("WAGON", df$body) , "WAGON")
df <- df[-which(df$body==""),]

df$seller <- toupper(str_replace_all(df$seller, "[^[:alnum:]]", ""))
df$seller <- replace(df$seller, grepl("FORD", df$seller), "FORD")
df$seller <- replace(df$seller, grepl("NISSAN", df$seller), "NISSAN")
df$seller <- replace(df$seller, grepl("HERTZ", df$seller), "HERTZ")
df$seller <- replace(df$seller, grepl("AVIS", df$seller), "AVIS")
df$seller <- replace(df$seller, grepl("TDA", df$seller), "TDA")
df$seller <- replace(df$seller, grepl("ENTERPRISE", df$seller), "ENTERPRISE")
df$seller <- replace(df$seller, grepl("HONDA", df$seller), "HONDA")
df$seller <- replace(df$seller, grepl("MERCEDES", df$seller), "MERCEDES")
df$seller <- replace(df$seller, grepl("CHRYSLER", df$seller), "CHRYSLER")
df$seller <- replace(df$seller, grepl("KIA", df$seller), "KIA")
df$seller <- replace(df$seller, grepl("TOYOTA", df$seller), "TOYOTA")
df$seller <- replace(df$seller, grepl("LEXUS", df$seller), "LEXUS")
df$seller <- replace(df$seller, grepl("GMF", df$seller), "GENERALMOTORS")
df$seller <- replace(df$seller, grepl("GMR", df$seller), "GENERALMOTORS")
df[-which(df$seller %in% c("FORD","NISSAN","HERTZ", "AVIS", "TDA", "ENTERPRISE",
                           "HONDA", "MERCEDES", "CHRYSLER", "KIA", "TOYOTA", 
                           "GENERALMOTORS")),"seller"] <- "OTHER"

df$interior <- iconv(df$interior, "UTF-8","UTF-8",sub='')
df$interior <- replace(df$interior, grepl("off-white", df$interior), "white")
df$interior <- str_replace_all(df$interior,"[^([:alnum:]|_)]","")

df$color <- iconv(df$color, "UTF-8","UTF-8",sub='')
df$color <- replace(df$color, grepl("off-white", df$color), "white")
df$color <- str_replace_all(df$color,"[^([:alnum:]|_)]","")


df <- select(df, -c(model,trim))

baseline.df <- df
# baseline.df$year <- as.factor(baseline.df$year)
baseline.df$make <- as.factor(baseline.df$make)
baseline.df$body <- as.factor(baseline.df$body)
baseline.df$transmission <- as.factor(baseline.df$transmission)
baseline.df$state <- as.factor(baseline.df$state)
baseline.df$color <- as.factor(str_replace_all(baseline.df$color,"[^([:alnum:]|_)]",""))
baseline.df$interior <- as.factor(str_replace_all(baseline.df$interior,"[^([:alnum:]|_)]",""))
baseline.df$seller <- as.factor(baseline.df$seller)
baseline.df$hybrid <- as.factor(baseline.df$hybrid)
baseline.df$low.emission <- as.factor(baseline.df$low.emission)
baseline.df$electric <- as.factor(baseline.df$electric)

library(fastDummies)
df <- dummy_cols(df, 
                 select_columns = colnames(df %>% select_if(is.character)),
                 remove_selected_columns = TRUE)

df <- dummy_cols(df, 
                 select_columns = colnames(df %>% select_if(is.factor)),
                 remove_selected_columns = TRUE)




return(list("df"=df, "baseline.df"=baseline.df))
}

