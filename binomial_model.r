setwd("~/programming/scriptie/Dolf/prototype_v3")
data = read.csv("dataset.csv")
roadtypes = unique(data$roadtype)

for (roadtype in roadtypes) {
  data[[roadtype]] = ifelse(data$roadtype == roadtype,1,0)
}
# take unclassified as reference category
roadtypes = roadtypes[roadtypes != "unclassified"]

data$log_edge_length = log(data$edge_length)


# estimate probit model
formula = as.formula(paste("in_pieterpad ~", paste(roadtypes, collapse = " + "), "+ forest_distance + forest_area + edge_length + log_edge_length"))
model = glm(formula, data = data, family = binomial)
summary(model)

library(MASS)
aic_step = stepAIC(model, direction = "both")

summary(aic_step)

