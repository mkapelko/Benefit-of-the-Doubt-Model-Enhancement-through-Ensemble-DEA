#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:20:30 2024

@author: monge
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.stats import moment
from docplex.mp.model import Model
from docplex.mp.environment import Environment
import random
import pandas as pd
import seaborn as sns

env = Environment()
env.print_information()
import cplex
from cplex.callbacks import BranchCallback
import time


Caso="2019"
archivo = 'Dataset2019.xlsx'
datos = pd.read_excel(archivo, sheet_name="Hoja1")
datos_2019=datos
datos_2019["Country"]=datos["Country"]



N_run=100  # Número of runs, index by run
n_DMU=len(datos)  # number of DMU, index by i
n_outputs = 70 # number of outputs, index by j
N_samples=500 # number of samples, index by s


data=np.zeros((n_DMU,n_outputs))

scores_sampling=-1*np.ones((n_DMU,N_samples))
scores_sim_sampling=-1*np.ones((n_DMU,N_run))
scores_sim=-1*np.ones((n_DMU,N_run))
scores_sampling_mean=-1*np.ones(n_DMU)



print("DMU=",n_DMU)
print("Outputs=",n_outputs)

print(datos_2019)


DMU_sim_lambdas=np.ones((n_DMU,N_samples,n_DMU))
DMU_sim_lambdas_count=np.ones((n_DMU,N_samples,n_DMU))
DMU_benchmarking_count_Approach_1=np.zeros(n_DMU)
DMU_benchmarking_count_i_j_Approach_1=np.zeros((n_DMU,n_DMU))
DMU_sampling_count_Approach_1=np.zeros(n_DMU)
DMU_j_visited=np.zeros(n_DMU)
DMU_i_visited=np.zeros(n_DMU)



count_ll=0
for run in range(N_run):
    
    print("--------------------------------------------")
    print("sim = ",run+1,"/", N_run)
    temp=0
    for i in range(n_DMU):
        for j in range(n_outputs):
            if Caso=="2019":
                data[i,j]=datos_2019.iloc[i,j+1]
    #################################################################3
    #  DEA Model 
    #################################################################3
    for i in range(n_DMU):
        mdl = Model()
        ll_k = [mdl.continuous_var(name="ll_k%d"%(k))  for k in  range(n_DMU) ]  
        phi_i = mdl.continuous_var(name="phi_i%d" )
        mdl.maximize(phi_i )
        mdl.add_constraint( mdl.sum( ll_k[k]  for k in   range(n_DMU) )  == 1  )
        for k in range(n_outputs) :
            mdl.add_constraint( mdl.sum( ll_k[j] * data[j][k] for j in range(n_DMU)  )  >= phi_i * data[i][k] )

        solution = mdl.solve(log_output=False)
        scores_sim[i][run]=phi_i.solution_value
        
    #################################################################3
    ## SAMPLING
    #################################################################3
   
    print("Sampling: ", N_samples)
    for s in range(N_samples):
        if s%10 == 0: 
            print(s,end='/')
        sample_DMU=random.choices(range(n_DMU), k=n_DMU)  # genera muestra de 0 a n-1, con repetición. Muestra tamaño n
        size_DMU=len(sample_DMU)
        size_outputs=8 #np.int16(n_outputs/3)
        sample_outputs=random.choices(range(n_outputs), k=size_outputs)  # genera muestra de 0 a n-1, con repetición. Muestra tamaño n
        data2=data[sample_DMU]
        data2=data2[:,sample_outputs]
    
        contado_i=np.zeros(n_DMU)
        for i in range(n_DMU):
            DMU_i_visited[i]=0
        for i in range(n_DMU):
            #print(i,' -',end ='')
            mdl = Model()
            ll_k = [mdl.continuous_var(name="ll_k%d"%(k))  for k in  range(n_DMU) ]  
            phi_i = mdl.continuous_var(name="phi_i%d" )
            mdl.maximize(phi_i )
            mdl.add_constraint( mdl.sum( ll_k[k]  for k in   range(n_DMU) )  == 1  )
            for k in range(size_outputs) :
                mdl.add_constraint( mdl.sum( ll_k[j] * data2[j][k] for j in range(n_DMU)  )  >= phi_i * data2[i][k] )
        
            solution = mdl.solve(log_output=False)
            ii = sample_DMU[i]
            scores_sampling[ii][s]=phi_i.solution_value

            if DMU_i_visited[ii]==0:
                DMU_sampling_count_Approach_1[ii]=DMU_sampling_count_Approach_1[ii]+1
                for j in range(n_DMU):
                    DMU_j_visited[j]=0 
                count=0
                for j in range(n_DMU):
                    if ll_k[j].solution_value > 0.05: 
                        if DMU_j_visited[sample_DMU[j]] <= 0:
                            DMU_benchmarking_count_i_j_Approach_1[ii][sample_DMU[j]]=DMU_benchmarking_count_i_j_Approach_1[ii][sample_DMU[j]]+1
                            DMU_benchmarking_count_Approach_1[sample_DMU[j]]=DMU_benchmarking_count_Approach_1[sample_DMU[j]]+1
                            DMU_j_visited[sample_DMU[j]]=1
                    count = count+1
            DMU_i_visited[ii]=1
             
            for j in range(n_DMU):
                DMU_j_visited[j]=0
                
            





            for j in range(n_DMU):
                DMU_sim_lambdas[ii][s][j]= ll_k[j].solution_value
                if ll_k[j].solution_value > 0.05:
                    DMU_sim_lambdas_count[ii][s][j]=1
                    count_ll=count_ll+1
        
        
        
    for i in range(n_DMU):
        temp=list(scores_sampling[i])
        temp= [i for i in temp if i  != -1]
        scores_sim_sampling[i][run]=np.mean(temp)
        
        
    dataframe_solucion1 = pd.DataFrame(scores_sim_sampling)
    dataframe_solucion2 = pd.DataFrame(scores_sim)
    with pd.ExcelWriter('output_partial.xlsx') as writer:  
         dataframe_solucion1.to_excel(writer, sheet_name='Sim')
         dataframe_solucion2.to_excel(writer, sheet_name='DEA')
        
        
        
        

dataframe_solucion1 = pd.DataFrame(scores_sim_sampling)
dataframe_solucion2 = pd.DataFrame(scores_sim)
with pd.ExcelWriter('output_final.xlsx') as writer:  
     dataframe_solucion1.to_excel(writer, sheet_name='Sim')
     dataframe_solucion2.to_excel(writer, sheet_name='DEA')
        
Confidece_levels = datos.iloc[:,0:8]


for i in range(0,n_DMU):
    temp=list(scores_sampling[i])
    temp= [i for i in temp if i  != - 1]
    
    scores_sampling_mean[i]= np.mean(temp)
    p05 = np.quantile(temp,0.05)
    p95 = np.quantile(temp,0.95)
    Confidece_levels.iloc[i,1]=round(p05,3)
    Confidece_levels.iloc[i,2]=round(p95,3)

    print(datos.iloc[i,0],"\t t  IC_90" ,round(p05,3),round(p95,3))
    p025 = np.quantile(temp,0.025)
    p975 = np.quantile(temp,0.975)
    Confidece_levels.iloc[i,3]=round(p025,3)
    Confidece_levels.iloc[i,4]=round(p975,3)


    Confidece_levels.iloc[i,5]=round(scores_sim[i][0],3)

    Confidece_levels.iloc[i,6]=round(scores_sampling_mean[i],3)

    print(datos.iloc[i,0],"\t t  IC_95" ,round(p025,3),round(p975,3))

    Confidece_levels.iloc[i,7]=DMU_benchmarking_count_Approach_1[i]
       
nombre_fic_IC='./'+'_Confidence_levels'+'_'+Caso+".txt"


Confidece_levels.columns=['Country','lower 90%','upper 90%','lower 95%','upper 95%','DEA score', 'mean Score sampling', 'Frequency']
Confidece_levels.to_csv(nombre_fic_IC, sep=" ",  escapechar=" ")

# Draw Confidence  Intervals  

for lower,upper,y in zip(Confidece_levels['lower 90%'],Confidece_levels['upper 90%'],range(len(Confidece_levels))):
        plt.plot((lower,upper),(y,y),'ro-',color='blue')
        plt.yticks(range(len(Confidece_levels)),list(Confidece_levels['Country']))

nombre_fig_IC_90='./'+'Intervalos_90'+'_'+Caso+".pdf"

plt.savefig(nombre_fig_IC_90, format="pdf", bbox_inches="tight")        

plt.show()        

plt.show()        

for lower,upper,y in zip(Confidece_levels['lower 95%'],Confidece_levels['upper 95%'],range(len(Confidece_levels))):
        plt.plot((lower,upper),(y,y),'ro-',color='blue')
        plt.yticks(range(len(Confidece_levels)),list(Confidece_levels['Country']))

nombre_fig_IC_95='./'+'Intervalos_95'+'_'+Caso+".pdf"

plt.savefig(nombre_fig_IC_95, format="pdf", bbox_inches="tight")        
    
plt.show()        


for i in range(0,n_DMU):
    temp=list(scores_sampling[i])
    temp= [i for i in temp if i  != -1]
    print(temp)  
    plt.xlabel('Score')
    plt.title(Caso)
    aaaa = sns.kdeplot(temp, bw_adjust=1.0, linewidth=3) 

    aaaa.set_xlim(1, max(temp))
    plt.legend(prop={'size': 16}, title = datos.iloc[i,0])        
        
    nombre='./'+Caso+'_'+datos.iloc[i,0]+".pdf"
    plt.savefig(nombre, format="pdf", bbox_inches="tight")        
    plt.show()        
    
    DMU_benchmarking_count_i_j_Approach_1=pd.DataFrame(DMU_benchmarking_count_i_j_Approach_1)
       
for i in range(0,n_DMU):
    for j in range(0,n_DMU):
        DMU_benchmarking_count_i_j_Approach_1[i][j]=round(100*DMU_benchmarking_count_i_j_Approach_1[i][j]/DMU_sampling_count_Approach_1[j],2)
    
DMU_benchmarking_count_i_j_Approach_1.to_csv("benchmaking_approach1.txt", sep=" ",  escapechar=" ")


