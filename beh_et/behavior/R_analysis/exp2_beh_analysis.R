library(ordinal)
library(tidyr)
library(dplyr)
library(lme4)
library(bayestestR)
library(emmeans)
library(ggplot2)
library(ggthemes) 
library(Bolstad)
library(BayesFactor)
library(rstatix)
library(ggpubr)
library(afex)
library(lsmeans)
library(psych)

# modalities
modalities <- c("MEEG", "fMRI")


"
OPTIMIZATION PARTICIPANTS
MEG: SA111, SA148, SB040, SB069, SB081
fMRI: SC109, SC143, SC160, SD107, SD165
"
optimization_subs <- c("SA111", "SA148", "SB040", "SB069", "SB081", 
                       "SC109", "SC143", "SC160", "SD107", "SD165")

#### CHANGE THIS TO "TRUE" IF YOU WANT TO WORK ONLY ON OPTIMIZATION SUBS, OTHERWISE IT'S ALL OF THEM
optimization_mode <- TRUE 


if (isTRUE(optimization_mode)){
  print("####### OPTIMOZATION MODE: results just for the 10 optimizariton subs #######")
}






####### =================== GENERAL SUMMARY STATISTICS =================== 


## Data ---------------------

data <- read.csv("quality_checks_general_stats.csv")
data$sub_code <- factor(data$sub_code)  # participant code is not a number

# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_summ_stats.png"
} else{  # don't filter the data, change the saved file name
  save_name <- "summ_stats.png"
}


print("=================== GENERAL SUMMARY STATISTICS =================== ")

# Prepare results saving
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_beh_analysis_stats.txt", append=TRUE)
}else{
  sink(file="exp2_beh_analysis_stats.txt", append=TRUE)  # txt file instead of a console
  
}


## Summary ------------------------
print("Summary")
print(summary(data))

print("Description")
print(describe(data))






####### =================== ANALYSIS 1 =================== 
"
In this analysis, we will compare the video game difficulty beteween 
seen (hits) and unseen (misses) trials in the video game phase (dAT).
"


# Prepare results saving
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_beh_analysis1.txt", append=TRUE)
}else{
  sink(file="exp2_beh_analysis1.txt", append=TRUE)  # txt file instead of a console
  
}

## Data ---------------------

data <- read.csv("vg_difficulty_per_vis.csv")  # 3 columns: sub_code, vgdiff, visibility
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$visibility <- factor(data$visibility)  # visibility: seen, unseen
data$modality <- factor(data$modality) 

# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_1.png"
} else{  # don't filter the data, change the saved file name
  save_name <- "analysis_1.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 1: Difficulty by Visibility =================== ")

## Summary ------------------------
print("Summary of difficulty for seen and unseen trials")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = describe))


## Per Modality ------------------------
for (mod in modalities){
  d <- data[data$modality == mod, ]
  print("                         ")
  print(paste0("------ summary per modality: ", mod, " ------", "N=", length(unique(d$sub_code))))
  print(by(data = d, 
           INDICES = d[,"visibility"], 
           FUN = summary))
  
  print("Description")
  print(by(data = d, 
           INDICES = d[,"visibility"], 
           FUN = describe))
  print("                         ")
}



## Model ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(vgdiff ~ modality * visibility + (1|sub_code), data=data)
result_summary <- summary(result)
print(result_summary)


if (isSingular(result)==TRUE){
  print("isSingular warning: The effects of sub_code are extremely small - there's no really systematic effect coming from sub_code (can be seen if running ranef(result))")
}

print("                ")

## Effect Significance ------------------------
print("------Effect Significance------")
result_effects <- anova(result, type = 2)  
print(result_effects)
# correct p-values with Bonferroni
result_effects[["p_adjust"]] <- p.adjust(result_effects[["Pr(>F)"]], method = "bonf")
print(result_effects)  


print("                ")
## Post Hoc ------------------------
print("------Post Hoc------")
for (i in 1:nrow(result_effects)) {  # for each effect in the effects table
  print(rownames(result_effects)[i])
  if (result_effects[i, "p_adjust"] < 0.05) {  # if it's significant
    if (!grepl(":", rownames(result_effects)[i])) {  
      # if it's a single fixed effect (not an interaction)
      fixed <- rownames(result_effects)[i]
      # post hoc
      em <- emmeans(result, fixed, lmer.df = "S") 
      em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
      print(em_contrast)  # write to file
      
    } else {  # else, this is a significant interaction effect
      terms <- strsplit(rownames(result_effects)[i], ":")[[1]]
      fixed1 <- terms[1]
      fixed2 <- terms[2]
      em <- emmeans(result, c(fixed1,fixed2), lmer.df = "S")  
      em_contrast <- contrast(em, method='pairwise', by=fixed2, infer=TRUE, adjust="bonf")
      print(em_contrast)  
    }
  }
} 


print("                ")
## Bayes Factors ------------------------
print("------Bayes Factors------")
# BIC hypothesis
bic_result <- BIC(result)
print(paste("MODEL 1 BIC: ", bic_result))

# BIC nulls
print(paste("against NULL MODELS: "))
null1 <-lmer(vgdiff ~ 1 + (1|sub_code), data=data)
null2 <-lmer(vgdiff ~ modality  + (1|sub_code), data=data)
null3 <-lmer(vgdiff ~ visibility + (1|sub_code), data=data)
null4 <-lmer(vgdiff ~ modality + visibility + (1|sub_code), data=data)
nulls <- c(null1, null2, null3, null4)

print("                           ")
print("------* comparison to nulls *------")

for (null in nulls){
  model_null <- null
  print("                           ")
  print("------ model summary ------")
  print(summary(model_null))
  print(paste("----- comparison to alternative -----"))
  bic_model_null <- BIC(model_null)
  print(paste("BIC: ", bic_model_null))
  # comparison: which model is preferred
  print(bayesfactor_models(result, denominator = model_null))
  print("                           ")
  print("------ model END ------")
  print("                           ")
}

## Close the file ---------------------
sink()








####### =================== ANALYSIS 2 =================== 
"
In this analysis, we will compare the hit rate (proportion of seen trials) 
between the two phases of exp.2 (game//dAT, replay//AT) and between the two 
stimulus category types (faces, objects). 
"

# Prepare results saving
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_beh_analysis2.txt", append=TRUE)
}else{
  sink(file="exp2_beh_analysis2.txt", append=TRUE)  # txt file instead of a console
  
}

## Data ---------------------

data <- read.csv("hits_fas_per_cond_category.csv")  #  columns: sub_code, 
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$stim_category <- factor(data$stim_category)  # face, obj
data$condition <- factor(data$condition)  # GAME, REPLAY
data$modality <- factor(data$modality)

# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_2.png"
} else{  # don't filter the data, change the saved file name
  save_name <- "analysis_2.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 2: Hit Rate by Category x Condition =================== ")

## Summary ------------------------
print("Summary statistics of hit rate per category and condition")
print(data %>%
        group_by(condition, stim_category) %>%
        get_summary_stats(hitrate, type = "mean_sd"))


## Per Modality ------------------------
for (mod in modalities){
  d <- data[data$modality == mod, ]
  print("                         ")
  print(paste0("------ summary per modality: ", mod, " ------", "N=", length(unique(d$sub_code))))
 print(d %>%
         group_by(condition, stim_category) %>%
         get_summary_stats(hitrate, type = "mean_sd"))
  print("                         ")
}



## Model ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(hitrate ~ modality * condition * stim_category + (1|sub_code), data=data)
result_summary <- summary(result)
print(result_summary)


if (isSingular(result)==TRUE){
  print("isSingular warning: The effects of sub_code are extremely small - there's no really systematic effect coming from sub_code (can be seen if running ranef(result))")
}


print("                ")
## Effect Significance ------------------------
print("------Effect Significance------")
result_effects <- anova(result, type = 2)  
print(result_effects)
# correct p-values with Bonferroni
result_effects[["p_adjust"]] <- p.adjust(result_effects[["Pr(>F)"]], method = "bonf")
print(result_effects)  


print("                ")
## Post Hoc ------------------------
print("------Post Hoc------")
for (i in 1:nrow(result_effects)) {  # for each effect in the effects table
  print(rownames(result_effects)[i])
  if (result_effects[i, "p_adjust"] < 0.05) {  # if it's significant
    if (!grepl(":", rownames(result_effects)[i])) {  
      # if it's a single fixed effect (not an interaction)
      fixed <- rownames(result_effects)[i]
      # post hoc
      em <- emmeans(result, fixed, lmer.df = "S") 
      em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
      print(em_contrast)  # write to file
      
    } else {  # else, this is a significant interaction effect
      terms <- strsplit(rownames(result_effects)[i], ":")[[1]]
      fixed1 <- terms[1]
      fixed2 <- terms[2]
      em <- emmeans(result, c(fixed1,fixed2), lmer.df = "S")  
      em_contrast <- contrast(em, method='pairwise', by=fixed2, infer=TRUE, adjust="bonf")
      print(em_contrast)  
    }
  }
} 


print("                ")
## Bayes Factors ------------------------
print("------Bayes Factors------")
# BIC hypothesis
bic_result <- BIC(result)
print(paste("MODEL 1 BIC: ", bic_result))

# BIC nulls
print(paste("against NULL MODELS: "))
null1 <-lmer(hitrate ~ 1 + (1|sub_code), data=data)
null2 <-lmer(hitrate ~ modality  + (1|sub_code), data=data)
null3 <-lmer(hitrate ~ condition + (1|sub_code), data=data)
null4 <-lmer(hitrate ~ stim_category + (1|sub_code), data=data)
null5 <-lmer(hitrate ~ modality * condition + (1|sub_code), data=data)
null6 <-lmer(hitrate ~ modality * stim_category + (1|sub_code), data=data)
null7 <-lmer(hitrate ~ condition * stim_category + (1|sub_code), data=data)
nulls <- c(null1, null2, null3, null4, null5, null6, null7)

print("                           ")
print("------* comparison to nulls *------")


for (null in nulls){
  model_null <- null
  print("                           ")
  print("------ model summary ------")
  print(summary(model_null))
  print(paste("----- comparison to alternative -----"))
  bic_model_null <- BIC(model_null)
  print(paste("BIC: ", bic_model_null))
  # comparison: which model is preferred
  print(bayesfactor_models(result, denominator = model_null))
  print("                           ")
  print("------ model END ------")
  print("                           ")
}

## Close the file ---------------------
sink()









####### =================== ANALYSIS 3 =================== 
"
In this analysis, we will compare the false alarm rate 
(proportion of seen responses for non-stimuli: BOTH BLANKS AND FILLERS), in 
the REPLAY ONLY (AT condition), between replay levels where the target was 
faces, and ones where the target was objects. 
"

# Prepare results saving
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_beh_analysis3.txt", append=TRUE)
}else{
  sink(file="exp2_beh_analysis3.txt", append=TRUE)  # txt file instead of a console
  
}


## Data ---------------------

data <- read.csv("replay_fa_wfillers_per_cat.csv")  # 3 columns: sub_code, replay_farate_wfillers, category
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$category <- factor(data$category)  # category: face, obj

# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_3.png"
} else{  # don't filter the data, change the saved file name
  save_name <- "analysis_3.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 3: FA rate WITH FILLERS by TARGET CATEGORY =================== ")

## Summary ------------------------
print("Summary of fa rate for face and object levels")
print(by(data = data, 
         INDICES = data[,"category"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"category"], 
         FUN = describe))


## Per Modality ------------------------
for (mod in modalities){
  d <- data[data$modality == mod, ]
  print("                         ")
  print(paste0("------ summary per modality: ", mod, " ------", "N=", length(unique(d$sub_code))))
  print(by(data = d, 
           INDICES = d[,"category"], 
           FUN = summary))
  
  print("Description")
  print(by(data = d, 
           INDICES = d[,"category"], 
           FUN = describe))
  print("                         ")
}



## Model ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(replay_farate_wfillers ~ modality * category + (1|sub_code), data=data)
result_summary <- summary(result)
print(result_summary)


if (isSingular(result)==TRUE){
  print("isSingular warning: The effects of sub_code are extremely small - there's no really systematic effect coming from sub_code (can be seen if running ranef(result))")
}

print("                ")
## Effect Significance ------------------------
print("------Effect Significance------")
result_effects <- anova(result, type = 2)  
print(result_effects)
# correct p-values with Bonferroni
result_effects[["p_adjust"]] <- p.adjust(result_effects[["Pr(>F)"]], method = "bonf")
print(result_effects)  

print("                ")
## Post Hoc ------------------------
print("------Post Hoc------")
for (i in 1:nrow(result_effects)) {  # for each effect in the effects table
  print(rownames(result_effects)[i])
  if (result_effects[i, "p_adjust"] < 0.05) {  # if it's significant
    if (!grepl(":", rownames(result_effects)[i])) {  
      # if it's a single fixed effect (not an interaction)
      fixed <- rownames(result_effects)[i]
      # post hoc
      em <- emmeans(result, fixed, lmer.df = "S") 
      em_contrast <- contrast(em, method='pairwise',infer=TRUE, adjust="bonf")
      print(em_contrast)  # write to file
      
    } else {  # else, this is a significant interaction effect
      terms <- strsplit(rownames(result_effects)[i], ":")[[1]]
      fixed1 <- terms[1]
      fixed2 <- terms[2]
      em <- emmeans(result, c(fixed1,fixed2), lmer.df = "S")  
      em_contrast <- contrast(em, method='pairwise', by=fixed2, infer=TRUE, adjust="bonf")
      print(em_contrast)  
    }
  }
} 


print("                ")
## Bayes Factors ------------------------
print("------Bayes Factors------")
# BIC hypothesis
bic_result <- BIC(result)
print(paste("MODEL 1 BIC: ", bic_result))

# BIC nulls
print(paste("against NULL MODELS: "))
null1 <-lmer(replay_farate_wfillers ~ 1 + (1|sub_code), data=data)
null2 <-lmer(replay_farate_wfillers ~ modality  + (1|sub_code), data=data)
null3 <-lmer(replay_farate_wfillers ~ category + (1|sub_code), data=data)
null4 <-lmer(replay_farate_wfillers ~ modality + category + (1|sub_code), data=data)
nulls <- c(null1, null2, null3, null4)


print("                           ")
print("------* comparison to nulls *------")

for (null in nulls){
  model_null <- null
  print("                           ")
  print("------ model summary ------")
  print(summary(model_null))
  print(paste("----- comparison to alternative -----"))
  bic_model_null <- BIC(model_null)
  print(paste("BIC: ", bic_model_null))
  # comparison: which model is preferred
  print(bayesfactor_models(result, denominator = model_null))
  print("                           ")
  print("------ model END ------")
  print("                           ")
}

## Close the file ---------------------
sink()
