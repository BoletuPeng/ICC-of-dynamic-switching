### Dynamic switching between brain networks predicts creative ability across cultures
##  CCRP code in R (coded by cqllogic@gmail.com)
## add pacakages
library(xlsx)
library(lattice)
library(ggplot2)
library(ggridges) 
library(grid)
library(gridExtra)                   #install.packages("gridExtra")
library(devtools)                    #install.packages("devtools")  #if(!require(devtools)) install.packages("devtools") devtools::install_github("kassambara/ggcorrplot")

library(easyGgplot2)                 #install_github("kassambara/easyGgplot2")
library(mgcv)                        #Generalized additive models if not install.packages("mgcv")
library(psych)                       #for EAP
library(lavaan)                      #for CFA  install.packages('lavaan')
library(ggpubr)
library(ggcorrplot)                  #install.packages("stringi",type="win.binary") # install.packages("stringi")
library(plyr)
library(WRS2)                        #install.packages("stringi",type="win.binary") # install.packages("WRS2")

### load data
data <- ""
df <- read.xlsx(data,1)

## check data
fix(df)
str(df)
describe(df)
colnames(df)

## mean age in each group
aggregate(df$age,by = list(ID = df$dataID),mean)

##
describe(df[which(df$dataID =="Queen"),])

### Figure 1: Age distribution across datasets
f1 <- ggplot(data = df) + geom_jitter(aes(x = dataID,y = age, color = dataID))+
      geom_pointrange(mapping = aes(x = dataID,y = age),
                     stat = "summary", fun.ymin = min, fun.ymax = max,fun.y = mean)+
      theme(axis.title = element_text(size = 16),axis.text = element_text(size = 12))+
      labs(x="Datasets", y = "Age",color = "Datasets")+ theme_bw()+
      coord_flip()+theme(legend.position = c(.8,.30))

tiff(filename = "Fig1.tif",width = 14, height = 15,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(f1,ncol=1,nrow=1)
dev.off() 

### Figure 2: Creative score distribution across datasets
f2 <- ggplot(df, aes(x =Zcrea , y = dataID)) +
      geom_density_ridges(aes(fill = dataID)) +
      labs(x="Z-score of creative performance",y = "") +
      guides(fill=guide_legend(title="Datasets"))+
      theme(axis.title = element_text(size = 16),axis.text = element_text(size = 12))+
      theme_bw()

tiff(filename = "Fig2.tif",width = 14, height = 15,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(f2,ncol=1,nrow=1)
dev.off()    

tiff(filename = "Fig1-2.tif",width = 28, height = 15,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(f1,f2,ncol=2,nrow=1)
dev.off()    

## Zcrea 
dfn <- df[which(df$dataID =="GBB_S1"),]

shapiro.test(dfn$Zcrea)
ks.test(dfn$Zcrea,'pnorm')

### Figure 3: The correlation between age and Zcrea
f3 <- ggplot(df, aes(x=age, y=Zcrea, color=dataID)) + 
  geom_point()+geom_smooth(method=lm,se=FALSE)+
  labs(x="Age",y = "Z-score of creative performance",color = "Datasets") +
  guides(fill=guide_legend(title="Datasets"))+
  theme(axis.title = element_text(size = 16),axis.text = element_text(size = 12))+
  theme_bw()


queen <- df[which(df$dataID =="Queen"),]
cor.test(queen$Zcrea,queen$age,method = "pearson")

### Gender difference in Zcrea
f4 <- ggplot(df, aes(x = dataID, y = Zcrea, fill = gender))+
      geom_bar(stat = "identity", position = "dodge")+
      geom_hline(yintercept = 0,colour = "grey90")+
      stat_summary(fun.y="mean", geom="point", size=5,
                   position=position_dodge(width=0.75), color="white")+
      labs(x="Datasets",y = "Z-score of creative performance")+
      guides(fill=guide_legend(title="Datasets"))+
      theme_bw()+
      coord_flip()

f5 <- ggplot(df, aes(x=IQ, y=Zcrea, color=dataID)) + 
  geom_point()+geom_smooth(method=lm,se=FALSE)+
  labs(x="Intelligence",y = "Z-score of creative performance") +
  theme(axis.title = element_text(size = 16),axis.text = element_text(size = 12))+
  theme_bw()+theme(legend.position = "none")


tiff(filename = "Fig3.tif",width = 42, height = 15,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(f3,f4,f5,ncol=3,nrow=1)
dev.off() 

###########################################################
#### Meta analyses ########################################
###########################################################
library(metafor)

data <- ""
dat <- read.xlsx(data,1)

dat <- escalc(measure="ZCOR", ri=rvalue, ni=size, data=dat, vtype="AV")
#dat <- escalc(measure="ZCOR", ri=rvalue, ni=size, data=dat, vtype="LS")
## Random-effects model
res <- rma(yi, vi, weights=size, data=dat, method="FE")
res


tiff(filename = "Fig4.tif",width = 16,height =12,units ="cm",compression="lzw",bg="white",res=600)
forest(res,refline=0,
       mlab="",
       xlim=c(-2, 3), 
       slab= dat$dataset,
       xlab="Observed output",showweights = T)
       text(-0.5,10:1,pos=2.5,dat$Location)
       text(c(-1.7,-0.8,1.55,2.35),12.5,pos=c(1,1,1,1),
            c("Datesets", "Location","Weight","r-value [95%CI]"),cex=1,font=1.5)
       text(-1.1,0,pos=1,cex=0.8,
            c("RE model for all datasets:"),font=1.5)
       text(-1.1,-0.8,pos=1,cex=0.8,
            bquote(paste( "Q = ", .(formatC(res$QE, digits=2, format="f")),
                         ", df = ", .(res$k - res$p),
                         ", p ", .(metafor:::.pval(res$QEp, digits=3, showeq=TRUE, sep=" ")), 
                         )))
dev.off()

## Random-effects model
resrm <- rma.mv(yi, vi, random=~1|Location, data=dat,method="REML")
resrm

tiff(filename = "Fig3.tif",width = 16,height =12,units ="cm",compression="lzw",bg="white",res=600)
forest(resrm,refline=0,
       mlab="",
       xlim=c(-2, 3), 
       slab= dat$dataset,
       xlab="Observed output",showweights = T)
text(-0.5,10:1,pos=2.5,dat$Location)
text(c(-1.7,-0.8,1.55,2.35),12.5,pos=c(1,1,1,1),
     c("Datesets", "Location","Weight","r-value [95%CI]"),cex=1,font=1.5)
text(-1.1,0,pos=1,cex=0.8,
     c("Two-level RE model for all datasets:"),font=1.5)
text(-1.1,-0.8,pos=1,cex=0.8,
     bquote(paste( "Q = ", .(formatC(resrm$QE, digits=2, format="f")),
                   ", df = ", .(resrm$k - resrm$p),
                   ", p ", .(metafor:::.pval(resrm$QEp, digits=3, showeq=TRUE, sep=" ")), 
     )))
dev.off()

########## moderation analysis ##########################################
resrm1 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=sqrt(vi))
summary(resrm1)

##Location
resrm2 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=~factor(Location)-1)
summary(resrm2)

data.Austria <- subset(dat,Location=="Austria")
data.Canada <- subset(dat,Location=="Canada")
data.China <- subset(dat,Location=="China")
data.Japan <- subset(dat,Location=="Japan")
data.USA <- subset(dat,Location=="USA")

resrm.Austria <- rma.mv(yi, vi, random=~1|Location, data=data.Austria)
resrm.China <- rma.mv(yi, vi, random=~1|Location, data=data.China)

## sample size
resrm3 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=size)
summary(resrm3)

## GenderRatio
resrm4 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=GenderRatio)
summary(resrm4)

## mean age
resrm6 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=age)
summary(resrm6)


## MRI
resrm5 <- rma.mv(yi, vi, random=~1|Location,data=dat,method="REML",mods=~factor(MRI)-1)
summary(resrm5)

data.Magnetom <- subset(dat,MRI=="Magnetom")
data.Trio <- subset(dat,MRI=="Trio")
resrm.Magnetom <- rma.mv(yi, vi, random=~1|Location, data=data.Magnetom)
resrm.Trio <- rma.mv(yi, vi, random=~1|Location, data=data.Trio)


## West vs Asia
resrm7 <- rma.mv(yi, vi, random=~1|cluture,data=dat,method="REML",mods=~factor(cluture)-1)
summary(resrm7)

data.Western <- subset(dat,cluture=="Western")
data.EastAsian <- subset(dat,cluture=="EastAsian")

resrm.Western <- rma.mv(yi, vi, random=~1|Location, data=data.Western )
resrm.EastAsian <- rma.mv(yi, vi, random=~1|Location, data=data.EastAsian)

#### Meta-Analysis via Linear (Mixed-Effects) Models
### compute the semi-partial correlation coefficients and their variances
dat <- read.csv("D:/CCRP/regmeta.csv")
dat1 <- escalc(measure="SPCOR", ti=tval2, ni=Sample, mi=preds, r2i=R2, data=dat)
dat1

### random-effects model
res <- rma(yi, vi, data=dat1, method = "FE")
res <- rma(yi, vi, data=dat1, method = "REML")
res

### random-effects model
res <- rma.mv(yi, vi, data=dat1, method = "FE")
res <- rma.mv(yi, vi, data=dat1, method = "REML")
res

###  Meta-Analysis via Multivariate/Multilevel Linear (Mixed-Effects) Models
res <- rma.mv(yi, vi,  random = ~ 1 | Datasets, data=dat1, method = "REML")
res

#########################################################
## correlation between switch and creative performance ##
## correlation between balance and creative performance #
##################################### UNCG dataset ######
## add data
data <- ""
dfs <- read.xlsx(data,1)

## check data
str(dfs)
describe(dfs)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

write.csv(dfs, file = "UNCG_dfs.csv")
## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")

#Figure
uncg1 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                   add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                   add.params = list(color = "black", fill = "lightgray",size = 2),
                   cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                   xlab = "", ylab = "",title = "UNCG")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic balance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
uncg2 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "UNCG") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()

## Figure 
tiff(filename = "Fig1_UNCG.tif",width = 24,height =12,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(uncg1,uncg2,ncol=2,nrow=1)
dev.off()

## Figure 5s

fig5s <- ggplot(dfs, aes(dytrad, switch)) + 
               geom_smooth(method = "lm", formula = y ~ x + I(x^2), 
               color = "black", se = FALSE) +
               geom_point(size = 3, col = "black") + 
               labs(x ="Balance", y = "Switching frequency") + 
               theme_classic(base_size = 16)+border()

tiff(filename = "Fig5s.tif",width = 8,height =8,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(fig5s ,ncol=1,nrow=1)
dev.off()

fig5s1 <- ggplot(dfs, aes(switch, crea)) + 
  geom_smooth(method = "lm", formula = y ~ x, 
              color = "black", se = FALSE) +
  geom_point(size = 3, col = "black") + 
  labs(x ="Switching frequency", y = "Creative performance") + 
  theme_classic(base_size = 16)+border()

tiff(filename = "Fig5s1.tif",width = 16,height =16,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(fig5s1 ,ncol=1,nrow=1)
dev.off()

fig5s2 <- ggplot(dfs, aes(dytrad, crea)) + 
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), 
              color = "black", se = FALSE) +
  geom_point(size = 3, col = "black") + 
  labs(x ="Balance", y = "Creative performance") + 
  theme_classic(base_size = 16)+border()

tiff(filename = "Fig5s2.tif",width = 8,height =8,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(fig5s2,ncol=1,nrow=1)
dev.off()

############################# UG_S1 ########################
dfs <- read.xlsx(data,2)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)
ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")


ugs1 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                   add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                   add.params = list(color = "black", fill = "lightgray",size = 2),
                   cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                  xlab = "", ylab = "",title = "UG_S1")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic balance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
ugs12 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "UG_S1") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()


############################# UG_S2 ########################
dfs <- read.xlsx(data,3)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ugs2 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                  add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                  add.params = list(color = "black", fill = "lightgray",size = 2),
                  cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                  xlab = "", ylab = "",title = "UG_S2")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
ugs22 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "UG_S2") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()

############################# UG_S3 ########################
dfs <- read.xlsx(data,4)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")


ugs3 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                  add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                  add.params = list(color = "black", fill = "lightgray",size = 2),
                  cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                  xlab = "", ylab = "",title = "UG_S3")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))
## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
ugs32 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "UG_S3") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()


############################# TKU ########################
dfs <- read.xlsx(data,7)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

tku <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                  add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                  add.params = list(color = "black", fill = "lightgray",size = 2),
                  cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                 xlab = "", ylab = "",title = "TKU")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
tku2 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "TKU") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()


############################# SLIM_S1 ########################
dfs <- read.xlsx(data,5)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")


slims1 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                 add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                 add.params = list(color = "black", fill = "lightgray",size = 2),
                 cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                 xlab = "", ylab = "",title = "SLIM_S1")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
slims12 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "SLIM_S1") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()

############################# SLIM_S2 ########################
dfs <- read.xlsx(data,6)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

slims2 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                 add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                 add.params = list(color = "black", fill = "lightgray",size = 2),
                 cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                 xlab = "", ylab = "",title = "SLIM_S2")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))


## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
slims22 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "SLIM_S2") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()
############################# GBB_S1 ########################
dfs <- read.xlsx(data,8)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")


gbbs1 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                    add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                    add.params = list(color = "black", fill = "lightgray",size = 2),
                    cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                   xlab = "", ylab = "",title = "GBB_S1")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))



## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
gbbs12 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "GBB_S1") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()
############################# GBB_S2 ########################
dfs <- read.xlsx(data,9)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

ctsIQ <- cor.test(dfs$resswitch,dfs$IQ,use = "complete.obs")


gbbs2 <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                   add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                   add.params = list(color = "black", fill = "lightgray",size = 2),
                   cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                   xlab = "", ylab = "",title = "GBB_S2")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
gbbs22 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "GBB_S2")+theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()
############################# queen ########################
dfs <- read.xlsx(data,10)
colnames(dfs)

## remove confounding variables by the Multivariate regression models to Standardized Residual for crea
fit_crea <-lm(crea ~ gender+age,data=dfs)
dfs$rescrea <- rstandard(fit_crea)

fit_switch <-lm(switch~ gender+age+FD+GMS,data=dfs)
dfs$resswitch <- rstandard(fit_switch)

fit_dytrad <-lm(dytrad~ gender+age+FD+GMS,data=dfs)
dfs$resdytrad <- rstandard(fit_dytrad)

## correlation between switch and creative performance
cts1 <- cor.test(dfs$resswitch,dfs$rescrea,use = "complete.obs")

cts2 <- pbcor(dfs$resswitch,dfs$rescrea, beta = 0.1, ci = TRUE, nboot = 1000)

queen <- ggscatter(dfs, x = "resswitch", y = "rescrea", 
                   add = "reg.line", conf.int = TRUE, point = T,color = "gray", size = 3,
                   add.params = list(color = "black", fill = "lightgray",size = 2),
                   cor.coef = F, cor.method = "pearson", cor.coef.size = 6,
                   xlab = "", ylab = "",title = "Queen")+border()+theme(plot.title = element_text(hjust = 0.5,size = 16))

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(rescrea ~ resdytrad,data=dfs)
summary(linear.model)

#Model2
dfs$resdytrad2 <- dfs$resdytrad^2
quadratic.model <-lm(rescrea ~ resdytrad+resdytrad2,data=dfs)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Figure 2
queen2 <- ggplot(dfs, aes(resdytrad,rescrea)) +
  geom_point(size=3,color = "gray") + 
  stat_smooth(method = "lm", formula = y ~ x, size = 2, se = F, color = "black") + 
  stat_smooth(method = "lm", formula = y ~ x + I(x^2), size = 2, se = F, color = "blue")+
  theme_classic(base_size = 12)+geom_vline(xintercept = 0,colour="gray")+
  labs(x = "",y="",title = "Queen") +theme(plot.title = element_text(hjust = 0.5,size = 16))+
  #annotate('text', x = 0.6, y = 3, label = "R^{2}==0.01~~P==0.49",parse = TRUE,size=4)
  border()



#### Figure 3
tiff(filename = "Fig3.tif",width = 60,height = 24,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(uncg1,ugs1,ugs2,ugs3,tku,
             slims1,slims2,gbbs1,gbbs2,queen,
             ncol=5,nrow=2)
dev.off()

#### Figure 4
tiff(filename = "Fig4.tif",width = 60,height = 24,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(uncg2,ugs12,ugs22,ugs32,tku2,
             slims12,slims22,gbbs12,gbbs22,queen2,
             ncol=5,nrow=2)
dev.off()

##############################################################
################### Reproducibility ##########################
data <- ""

dfsall <- read.xlsx(data,11)

dfsall <- read.csv('alldata.csv')
colnames(dfsall)
describe(dfsall)
str(dfsall)

dfsall$site <- as.factor(dfsall$site)
dfsall$location <- as.factor(dfsall$location)

cor.test(dfsall$switchr,dfsall$Zcrear,use = "complete.obs")
pbcor(dfsall$switchr,dfsall$Zcrear, beta = 0.1, ci = TRUE, nboot = 1000)

## correlation between dynamic banlance and creative performance
#Model1
linear.model <-lm(Zcrear ~ resdytrad,data=dfsall)
summary(linear.model)

#Model2
dfsall$resdytrad2 <- dfsall$resdytrad^2
quadratic.model <-lm(Zcrear ~ resdytrad+resdytrad2,data=dfsall)
summary(quadratic.model)

# Model comparison
anova(linear.model,quadratic.model)
AIC(linear.model,quadratic.model)

# Linear Mixed-Effects Models
#library (lme4)
#fit <- lmer(RT ~ prime + (prime|item) + (prime|participant), data = data)
#summary(fit)
##There is one fixed effect (the effect of prime) and four random effects:
##The intercept per participant (capturing the fact that some participants are faster than others). The intercept per item (capturing the fact that some items are easier than others). The slope per participant (capturing the possibility that the priming effect is not the same for all participants). The slope per item (capturing the possibility that the priming effect is not the same for all items).

library(lme4)
library(lmerTest)
library(EMAtools)        # install.packages("EMAtools")
library(ggpointdensity)  # install.packages("ggpointdensity")

prem <-lmer(Zcrear ~ switchr+(1|site)+(1|location), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

summary(prem) 
anova(prem)

#This will calculate Cohen's D for each effect in an lme4 object.
lme.dscore(prem, data = dfsall, type="lme4")

## Generalized non-linear mixed models
prem1 <-lmer(Zcrear ~ resdytrad + (1|site)+(1|location), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))
prem2 <-lmer(Zcrear ~ resdytrad + I(resdytrad^2) + (1|site)+(1|location), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

summary(prem) 
anova(prem1,prem2)

#This will calculate Cohen's D for each effect in an lme4 object.
lme.dscore(prem2, data = dfsall, type="lme4")


##### Fig.5
f5 <- ggplot(dfsall,aes(x=switchr, y=Zcrear,color =location)) +
  geom_point(alpha = 0.2,size =3) + 
  geom_smooth(method=lm,formula = y ~ x,se=T,linewidth = 1,colour = "grey2") +
  geom_smooth(method = 'lm',formula = y ~ x,se = FALSE, aes(group = location)) +
  theme_classic(base_size = 10) +
  theme(legend.position = "none",panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 12),axis.text.y = element_text(size = 12)) +
  labs(x="The frequency of switch between two states", y= "Creative performance",color = "Country") 
  

f6 <- ggplot(dfsall,aes(x=resdytrad, y=Zcrear,color =location)) +
  geom_point(alpha = 0.2,size =3) + 
  geom_smooth(method=lm,formula = y ~ x + I(x^2),se=T,linewidth = 1,colour = "grey2") +
  geom_smooth(method = 'lm',formula = y ~ x + I(x^2), se = FALSE, aes(group = location)) +
  theme_classic(base_size = 10) +
  theme(legend.position = c(.9, .8),panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 12),axis.text.y = element_text(size = 12)) +
  labs(x="The degree of balance between two states", y= "Creative performance",color = "Country") 



tiff(filename = "Fig5.tif",width = 26,height = 13,units ="cm",compression="lzw",bg="white",res=600)
  grid.arrange(f5,f6,ncol=2,nrow=1)
dev.off()

############################ External validation using task-based fMRI data ###############################
dfsall <- read.csv('task-based-fMRIdata.csv')

prem1 <-lmer(NovAUT ~ autSwP+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))
prem2 <-lmer(NovOCT ~ octSwP+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

summary(prem1) 
summary(prem2) 
anova(prem1,prem2)

#This will calculate Cohen's D for each effect in an lme4 object.
lme.dscore(prem, data = dfsall, type="lme4")

## Generalized non-linear mixed models
prem1 <-lmer(NovAUT ~ autBlanA+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))
prem2 <-lmer(NovAUT ~ autBlanA+I(autBlanA^2)+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

prem1 <-lmer(NovOCT ~ octBlanA+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))
prem2 <-lmer(NovOCT ~ octBlanA+I(octBlanA^2)+(1|subID)+(1|runID), data = dfsall,control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4)))

summary(prem1) 
summary(prem2) 
anova(prem1,prem2)

#This will calculate Cohen's D for each effect in an lme4 object.
lme.dscore(prem2, data = dfsall, type="lme4")

## Fig.6CDEF
fc <- ggplot(dfsall,aes(x=autSwP, y=NovAUT)) +
  geom_point(alpha = 0.2,size =3) + 
  geom_smooth(method=lm,formula = y ~ x,se=T,linewidth = 1,colour = "grey2") +
  #geom_smooth(method = 'lm',formula = y ~ x,se = FALSE, aes(group = location)) +
  theme_classic(base_size = 10) +
  theme(legend.position = "none",panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 14),axis.text.y = element_text(size = 14)) +
  labs(x="", y= "") 

fd <- ggplot(dfsall,aes(x=octSwP, y=NovOCT)) +
  geom_point(alpha = 0.2,size =3) + 
  geom_smooth(method=lm,formula = y ~ x,se=T,linewidth = 1,colour = "grey2") +
  #geom_smooth(method = 'lm',formula = y ~ x,se = FALSE, aes(group = location)) +
  theme_classic(base_size = 10) +
  theme(legend.position = "none",panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 14),axis.text.y = element_text(size = 14)) +
  labs(x="", y= "") 

fe <- ggplot(dfsall,aes(x=autBlanA,y=NovAUT)) +
  geom_point(alpha = 0.2,size =3) + 
  stat_smooth(method = "lm", formula = y ~ x, linewidth = 1, se = F, color = "black") + 
  geom_smooth(method=lm,formula = y ~ x + I(x^2),se=T,linewidth = 1,colour = "blue") +
  #geom_smooth(method = 'lm',formula = y ~ x + I(x^2), se = FALSE, aes(group = runID)) +
  theme_classic(base_size = 10) +
  theme(legend.position = c(.9, .8),panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 14),axis.text.y = element_text(size = 14)) +
  labs(x="", y= "") 

ff <- ggplot(dfsall,aes(x=octBlanA,y=NovOCT)) +
  geom_point(alpha = 0.2,size =3) + 
  stat_smooth(method = "lm", formula = y ~ x, linewidth = 1, se = F, color = "black") + 
  geom_smooth(method=lm,formula = y ~ x + I(x^2),se=T,linewidth = 1,colour = "blue") +
  #geom_smooth(method = 'lm',formula = y ~ x + I(x^2), se = FALSE, aes(group = runID)) +
  theme_classic(base_size = 10) +
  theme(legend.position = c(.9, .8),panel.border = element_rect(fill=NA,color="grey", size=1, linetype="solid"),
        axis.title.x = element_text(size=16),axis.title.y = element_text(size=16),
        axis.text.x= element_text(size = 14),axis.text.y = element_text(size = 14)) +
  labs(x="", y= "") 


tiff(filename = "Fig6.tif",width = 36,height = 9,units ="cm",compression="lzw",bg="white",res=600)
grid.arrange(fc,fd,fe,ff,ncol=4,nrow=1)
dev.off()


