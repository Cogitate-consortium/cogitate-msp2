library(ordinal)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(bayestestR)
library(emmeans)
library(ggplot2)
library(Bolstad)
library(BayesFactor)
library(rstatix)
library(ggpubr)
library(afex)
library(lsmeans)
library(psych)

# modalities
modalities <- c("MEG", "fMRI")


"
OPTIMIZATION PARTICIPANTS
MEG: SA111, SA148, SB040, SB069, SB081
fMRI: SC109, SC143, SC160, SD107, SD165
"
optimization_subs <- c("SA111", "SA148", "SB040", "SB069", "SB081", 
                       "SC109", "SC143", "SC160", "SD107", "SD165")

#### CHANGE THIS TO "TRUE" IF YOU WANT TO WORK ONLY ON OPTIMIZATION SUBS, OTHERWISE IT'S ALL OF THEM
optimization_mode <- TRUE 






####### =================== ANALYSIS 1 =================== 
"
In this analysis, we will compare fixation distance from the stimulus location
within the time window of [-500ms, 0ms] in the dAT (game), 
between seen and unseen trials.
"


## Prepare results saving ---------------------
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_et_analysis1.txt", append=TRUE)
}else{
  sink(file="exp2_et_analysis1.txt", append=TRUE)  # txt file instead of a console
}


## Data ---------------------

data <- read.csv("vg_prestim_fixdist_per_vis.csv")  # 3 columns: sub_code, vg_fixdiststim, visibility
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$visibility <- factor(data$visibility)  # visibility: seen, unseen



# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_1.png"
} else{  # don't filter the data, change the saved file name
  data <- data[!(data$sub_code %in% optimization_subs), ]  #  take out the "optimization_subs"
  save_name <- "analysis_1.png"
}

print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 1: Fixation Dist from Stim in Prestim Window by Visibility =================== ")

## Summary ------------------------
print(paste0("N = ", length(unique(data$sub_code)), " participants in data"))
print("Summary of fixation distance from the stimulus in the prestim window for seen and unseen trials")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = describe))


print("                                            ")
print("Summary for MEG and fMRI")
print(by(data = data, 
         INDICES = data[,"mod"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"mod"], 
         FUN = describe))


print("                                            ")
print("Summary per modality")
# per modality
for (modality in modalities){
  d <- data[data$mod == modality, ]
  print("                         ")
  print(paste0("------ summary per modality: ", modality, " ------", "N=", length(unique(d$sub_code))))
  print(by(data = d, 
           INDICES = d[,"visibility"], 
           FUN = summary))
  
  print("Description")
  print(by(data = d, 
           INDICES = d[,"visibility"], 
           FUN = describe))
  print("                         ")
}



## Models ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(vg_fixdiststim ~ mod * visibility + (1|sub_code), data=data)
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
null1 <-lmer(vg_fixdiststim ~ 1 + (1|sub_code), data=data)
null2 <-lmer(vg_fixdiststim ~ mod  + (1|sub_code), data=data)
null3 <-lmer(vg_fixdiststim ~ visibility + (1|sub_code), data=data)
null4 <-lmer(vg_fixdiststim ~ mod + visibility + (1|sub_code), data=data)
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
In this analysis, we will compare the NUMBER OF SACCADES (saccade rate)
within the time window of [-500ms, 0ms] in the dAT (game), 
between seen and unseen trials.
"

## Prepare results saving ---------------------
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_et_analysis2.txt", append=TRUE)
}else{
  sink(file="exp2_et_analysis2.txt", append=TRUE)  # txt file instead of a console
}


## Data ---------------------

data <- read.csv("vg_prestim_sacc_per_vis.csv")  # 3 columns: sub_code, vg_numsaccs, visibility
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$visibility <- factor(data$visibility)  # visibility: seen, unseen



# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_2.png"
} else{  # don't filter the data, change the saved file name
  data <- data[!(data$sub_code %in% optimization_subs), ]  #  discard only "optimization_subs"
  save_name <- "analysis_2.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 2: Saccade Rate in Prestim Window by Visibility =================== ")

## Summary ------------------------
print(paste0("N = ", length(unique(data$sub_code)), " participants in data"))
print("Summary of saccade rate in the prestim window for seen and unseen trials")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"visibility"], 
         FUN = describe))


print("                                            ")
print("Summary for MEG and fMRI")
print(by(data = data, 
         INDICES = data[,"mod"], 
         FUN = summary))

print("Description")
print(by(data = data, 
         INDICES = data[,"mod"], 
         FUN = describe))


print("                                            ")
print("Summary per modality")
# per modality
for (modality in modalities){
  d <- data[data$mod == modality, ]
  print("                         ")
  print(paste0("------ summary per modality: ", modality, " ------", "N=", length(unique(d$sub_code))))
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
result <-lmer(vg_numsaccs ~ mod * visibility + (1|sub_code), data=data)
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
null1 <-lmer(vg_numsaccs ~ 1 + (1|sub_code), data=data)
null2 <-lmer(vg_numsaccs ~ mod  + (1|sub_code), data=data)
null3 <-lmer(vg_numsaccs ~ visibility + (1|sub_code), data=data)
null4 <-lmer(vg_numsaccs ~ mod + visibility + (1|sub_code), data=data)
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





####### =================== ANALYSIS 3 =================== 
"
In this analysis, we will compare the DURATION of the FIRST fixation post 
stimulus onset that was located on the stimulus (within a window [0, 500] - 
after that, the stimulus is gone and it doesn't count as 'on it'), between 
face stimuli and object stimuli, for seen stimuli and for 
unseen stimuli. 
"

## Prepare results saving ---------------------
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_et_analysis3.txt", append=TRUE)
}else{
  sink(file="exp2_et_analysis3.txt", append=TRUE)  # txt file instead of a console
}


## Data ---------------------

data <- read.csv("vg_poststim_fixdur_per_vis.csv")  # 4 columns: sub_code, vg_fixdur, stim_category, visibility
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$visibility <- factor(data$visibility)  # visibility: seen, unseen
data$category <- factor(data$category)  # visibility: face, obj
data$mod <- factor(data$mod)


# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_3.png"
} else{  # don't filter the data, change the saved file name
  data <- data[!(data$sub_code %in% optimization_subs), ]  #  discard only "optimization_subs"
  save_name <- "analysis_3.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 3: First Stim Fixation Duration in Stim Window by Cagetory x Visibility =================== ")

## Summary ------------------------
print(paste0("N = ", length(unique(data$sub_code)), " participants in data"))
print("Summary of the duration of the first fixation towards the stimulus per category and visibility")
print(data %>%
        group_by(category, visibility) %>%
        get_summary_stats(vg_fixdur, type = "mean_sd"))

print("                                 ")
print("summary for fMRI and MEG")
print(data %>%
        group_by(mod) %>%
        get_summary_stats(vg_fixdur, type = "mean_sd"))


print("                                 ")
print("summary per modality")
# per modality
for (modality in modalities){
  d <- data[data$mod == modality, ]
  print("                         ")
  print(paste0("------ summary per modality: ", modality, " ------", "N=", length(unique(d$sub_code))))
  print(d %>%
          group_by(category, visibility) %>%
          get_summary_stats(vg_fixdur, type = "mean_sd"))
  print("                         ")
}




## Model ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(vg_fixdur ~ mod * visibility * category + (1|sub_code), data=data)
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
null1 <-lmer(vg_fixdur ~ 1 + (1|sub_code), data=data)
null2 <-lmer(vg_fixdur ~ mod  + (1|sub_code), data=data)
null3 <-lmer(vg_fixdur ~ visibility + (1|sub_code), data=data)
null4 <-lmer(vg_fixdur ~ category + (1|sub_code), data=data)
null5 <-lmer(vg_fixdur ~ mod * visibility + (1|sub_code), data=data)
null6 <-lmer(vg_fixdur ~ mod * category + (1|sub_code), data=data)
null7 <-lmer(vg_fixdur ~ visibility * category + (1|sub_code), data=data)
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






####### =================== ANALYSIS 4 =================== 
"
In this analysis, we will compare the ONSET of the FIRST saccade post 
stimulus onset  (within a window [0, 500] - after that, the stimulus is gone and 
it doesn't count as caused by it), between face stimuli and object stimuli, 
for seen stimuli and for unseen stimuli. 
"

## Prepare results saving ---------------------
if (isTRUE(optimization_mode)){
  sink(file="optimization_exp2_et_analysis4.txt", append=TRUE)
}else{
  sink(file="exp2_et_analysis4.txt", append=TRUE)  # txt file instead of a console
}

## Data ---------------------

data <- read.csv("vg_poststim_sacc_per_vis.csv")  # 4 columns: sub_code, vg_sacconset, stim_category, visibility
data$sub_code <- factor(data$sub_code)  # participant code is not a number
data$visibility <- factor(data$visibility)  # visibility: seen, unseen
data$category <- factor(data$category)  # visibility: face, obj


# If optimization: filter 
if (isTRUE(optimization_mode)){
  data <- data[data$sub_code %in% optimization_subs, ]  #  keep only the "optimization_subs"
  save_name <- "optimization_analysis_4.png"
} else{  # don't filter the data, change the saved file name
  data <- data[!(data$sub_code %in% optimization_subs), ]  #  discard only "optimization_subs"
  save_name <- "analysis_4.png"
}


print("                                                    ")
print("****************************************************")
print("=================== ANALYSIS 4: First Saccade Onset in Stim Window by Cagetory x Visibility =================== ")

## Summary ------------------------
print(paste0("N = ", length(unique(data$sub_code)), " participants in data"))
print("Summary of the onset of the first saccade per category and visibility")
print(data %>%
        group_by(category, visibility) %>%
        get_summary_stats(vg_sacconset, type = "mean_sd"))


print("                                 ")
print("summary for fMRI and MEG")
print(data %>%
        group_by(mod) %>%
        get_summary_stats(vg_sacconset, type = "mean_sd"))


print("                                 ")
# per modality
for (modality in modalities){
  d <- data[data$mod == modality, ]
  print("                         ")
  print(paste0("------ summary per modality: ", modality, " ------", "N=", length(unique(d$sub_code))))
  print(d %>%
          group_by(category, visibility) %>%
          get_summary_stats(vg_sacconset, type = "mean_sd"))
  print("                         ")
}




## Model ------------------------
print("                                                    ")
print("------------------------ Linear Mixed-Effects Model ------------------------")
result <-lmer(vg_sacconset ~ mod * visibility * category + (1|sub_code), data=data)
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
null1 <-lmer(vg_sacconset ~ 1 + (1|sub_code), data=data)
null2 <-lmer(vg_sacconset ~ mod  + (1|sub_code), data=data)
null3 <-lmer(vg_sacconset ~ visibility + (1|sub_code), data=data)
null4 <-lmer(vg_sacconset ~ category + (1|sub_code), data=data)
null5 <-lmer(vg_sacconset ~ mod * visibility + (1|sub_code), data=data)
null6 <-lmer(vg_sacconset ~ mod * category + (1|sub_code), data=data)
null7 <-lmer(vg_sacconset ~ visibility * category + (1|sub_code), data=data)
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

