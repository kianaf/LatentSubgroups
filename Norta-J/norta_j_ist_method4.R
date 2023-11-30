# https://mac.r-project.org/tools/


## gcipdr_IST_analysis.R contains R commands to execute 'gcipdr application to multi-center IST data' (from omonimous repository). 
## Copyright (C) 2019 Federico Bonofiglio

## This Program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This Program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with This Program.  If not, see <https://www.gnu.org/licenses/>. 

if (!require("devtools")) {
  install.packages("devtools")
  library(devtools)
}


## Install 'JohnsonDistribution' dependency (only available on CRAN archives)

url <- "https://cran.r-project.org/src/contrib/Archive/JohnsonDistribution/JohnsonDistribution_0.24.tar.gz"
pkgFile <- "JohnsonDistribution_0.24.tar.gz"
download.file(url = url, destfile = pkgFile)

install.packages(pkgs=pkgFile, type="source", repos=NULL)

unlink(pkgFile)


### INSTALL package (install other dependencies manually if needed)


R_REMOTES_NO_ERRORS_FROM_WARNINGS="true"
install_github("bonorico/gcipdr")


# load libraries
# trace("Simulate.many.datasets", edit = TRUE)


library(gcipdr)

if (!require("cowplot")) {
  install.packages("cowplot")
  library(cowplot)
}

if (!require("meta")) {
  install.packages("meta")
  library(meta)
}

if (!require("metafor")) {
  install.packages("metafor")
  library(metafor)
}

if (!require("lme4")) {
  install.packages("lme4")
  library(lme4)
}


## do not run !
## options("mc.cores") <- 3 ## set up your global 'mclapply()' forking options other than default (2 cores). Beware: changing default options might not reproduce results as shown in paper.


# loading the data
# url2 <- "./data/simulation.csv"
# url2 <- "./data/ist.csv"

setwd("/Users/farhadyar/Documents")
url2 <- "Project_PTVAE/progs/github_repo/PTVAE/data/ist_randomization_data_smaller_no_west_no_south_aug5.csv"
x_simulation <-as.data.frame(read.csv(url2 , header = TRUE))[, 1:15] 

   #######################  COMMENT (Bono): current example runs in approx 3 minutes but ...
options(mc.cores = 8L)  ## .. change accordingly to your Nr of cores (> 3) minus one, to speed up calculations  
######################



# sim_min_array = c(0,0)
# sim_max_array = c(0,0)

# for(i in 1:dim(x_simulation)[2]){
#   sim_min_array[i] <- min(x_simulation[,i])
#   sim_max_array[i] <- max(x_simulation[,i])
#   x_simulation[,i] <- (x_simulation[,i] - sim_min_array[i] )/ (sim_max_array[i] - sim_min_array[i])
# }

# Initialize an empty list
list_x <- list()

# Use a loop to create the strings and add them to the list
for(i in 1:dim(x_simulation)[2]){
  list_x[i] <- paste0("x", i)
}


colnames(x_simulation)<-list_x


stats <- Return.key.IPD.summaries(Return.IPD.design.matrix(x_simulation), 
                                  corrtype = "moment.corr")


set.seed( 65553, "L'Ecuyer")
system.time(
  
  pseudodata <- DataRebuild(
    H = 100,
    n= stats$sample.size,
    correlation.matrix =  stats$correlation.matrix,
    moments = stats$first.four.moments,
    x.mode = stats$is.binary.variable,
    corrtype = "moment.corr",
    marg.model = "johnson", #"gamma",  # or johnson
    variable.names = stats$variable.names, 
    checkdata = TRUE,
    tabulate.similar.data = TRUE
  )
  
)



syn <- pseudodata$Xspace

syn_fed <- syn[[1]]



# for(i in 1:dim(x_simulation)[2]){
#   syn_fed[,i] <- (sim_max_array[i] - sim_min_array[i]) * syn_fed[,i] + sim_min_array[i]
# }


write.csv(syn_fed, "Project_PTVAE/progs/github_repo/PTVAE/Norta-J/202311_IST_two_regions_method4.csv")