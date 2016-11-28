setwd("C:/Users/flori/Desktop/Apprentissage statistique/Projet Carte VP")


# 1. Construction des échantillons apprentissage -----------------------------
source("visa_lec.R")
source("visa_trans.R")
source("visa_code.R")

summary(vispremv)
# variable contenant la liste des noms de variables
var=names(vispremv)
# liste des variables explicatives quantitatives
varquant=var[1:30]
# liste des variables explicatives qualitatives
varqual=var[31:55]


set.seed(13) # modifier 111
npop=nrow(vispremv)
# tirage de 200 indices sans remise
testi=sample(1:npop,200)
#Liste des indices restant qui n'ont pas été tirés
appri=setdiff(1:npop,testi)
# Extraction échantillon d'apprentissage
visappt=vispremv[appri,]
# Extraction échantillon de test
vistest=vispremv[testi,]
summary(visappt) # vérifications
summary(vistest)

# Pour SAS et Python
write.table(visappt,"visappt.dat",row.names=FALSE)
write.table(vistest,"vistest.dat",row.names=FALSE)


write.table(visappt,"visappt.txt")
write.table(vistest,"vistest.txt")

# 2. Régression logistique avec SAS et R -----------------------------------

## 2.1 Estimation
# sélection des prédicteurs qualitatifs
visapptq=visappt[,c("CARVP", varqual)]
# pour l'échantillon test
vistestq=vistest[,c("CARVP", varqual)]
# Estimation du modèle complet sans interaction
visa.logit=glm(CARVP~.,data=visapptq,
               family=binomial,na.action=na.omit)
# tests de nullité des coefficients
anova(visa.logit,test="Chisq")

## 2.2 Choix de modèle
visa.logit=glm(CARVP~.,data=visapptq,
               family=binomial,na.action=na.omit)
visa.step<-step(visa.logit)
# variables du modèles
anova(visa.step,test="Chisq")

library(boot)
# premier modele
visa1.logit=glm(CARVP ~ SEXEQ + PCSPQ + kvunbq +
                  uemnbq + nptagq + endetq + gagetq + facanq +
                  havefq + relatq + qsmoyq + opgnbq + moyrvq +
                  dmvtpq + boppnq + jnbjdq + itavcq,
                data=visapptq,family=binomial,na.action=na.omit)
cv.glm(visapptq,visa1.logit,K=10)$delta[1]
# deuxieme modele
visa2.logit=glm(CARVP ~ SEXEQ + PCSPQ + kvunbq +
                  uemnbq + nptagq + endetq + gagetq + facanq +
                  havefq + relatq + qsmoyq + opgnbq + moyrvq +
                  dmvtpq + jnbjdq + itavcq, data=visapptq,
                family=binomial,na.action=na.omit)
cv.glm(visapptq,visa2.logit,K=10)$delta[1]

visa.logit=glm(CARVP ~ SEXEQ + PCSPQ + kvunbq +
                 uemnbq + nptagq + endetq + gagetq + facanq +
                 havefq + relatq + qsmoyq + opgnbq + moyrvq +
                 dmvtpq + boppnq + jnbjdq + itavcq,
               data=visapptq,family=binomial,na.action=na.omit)

## 2.3 Prévision de l'échantillon test
pred.vistest=predict(visa.logit,
                     newdata=vistestq)>0.5
table(pred.vistest,vistestq$CARVP=="Coui")
# pred.vistest FALSE TRUE
#      FALSE   127   22   
#      TRUE     10   41

## 2.4 Courbe ROC
library(ROCR)
roclogit=predict(visa.logit, newdata=vistestq,
                 type="response")
predlogistic=prediction(roclogit,vistestq$CARVP)
perflogistic=performance(predlogistic, "tpr","fpr")
plot(perflogistic,col=1)
legend("bottom",legend=c("logit"),col=1,lty=1)

# 3. Analyse discriminante ------------------------------------------------

## 3.1 Estimation
library(MASS) # chargement des librairies
library(class)
visapptr=visappt[,c("CARVP", varquant)]
vistestr=vistest[,c("CARVP", varquant)]
# analyse discriminante linéaire
visa.disl=lda(CARVP~.,data=visapptr)
# analyse discriminante quadratique
visa.disq=qda(CARVP~.,data=visapptr)
# k plus proches voisins
visa.knn=knn(visapptr[,-1],vistestr[,-1],
             visapptr$CARVP,k=10) 

library(e1071)
plot(tune.knn(visapptr[,-1],visapptr$CARVP,
              k=seq(2,30, by=2)))

visa.knn=knn(visapptr[,-1],vistestr[,-1],
             visapptr$CARVP,k=16)

## 3.2 Prévision de l'échantillon test
table(vispremv[testi,"CARVP"],
      predict(visa.disl,vistestr)$class)
#Cnon Coui
#Cnon  123   14
#Coui   21   42

table(vispremv[testi,"CARVP"],
      predict(visa.disq,vistestr)$class)
#Cnon Coui
#Cnon  114   23
#Coui   27   36

table(visa.knn,vistestq$CARVP)
#Cnon Coui
#Cnon  114   38
#Coui   23   25

## 3.3 Courbe ROC
ROClda=predict(visa.disl,vistestr)$posterior[,2]
predlda=prediction(ROClda,vistestq$CARVP)
perflda=performance(predlda,"tpr","fpr")

ROCqda=predict(visa.disq,vistestr)$posterior[,2]
predqda=prediction(ROCqda,vistestq$CARVP)
perfqda=performance(predqda,"tpr","fpr")

ROCknn=predict(visa.knn,vistestr)$posterior[,2] #PB courbe ROC knn
predknn=prediction(ROCknn,vistestq$CARVP)
perfknn=performance(predknn,"tpr","fpr")
roclogit=predict(visa.logit, newdata=vistestq,
                 type="response")
predlogistic=prediction(roclogit,vistestq$CARVP)
perflogistic=performance(predlogistic, "tpr","fpr")

plot(perflogistic,col=1)
plot(perflda,col=2,add=TRUE)
plot(perfqda,col=3,add=TRUE)
plot(perfknn,col=4,add=TRUE)
legend("bottom",legend=c("logit","lda","qda","knn"),col=1:4,lty=1)


# 4. Arbres de décisions binaires -----------------------------------------


