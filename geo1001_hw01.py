#-- GEO1001.2020--hw01
#-- Lars Langhorst
#-- 4299922

import pandas as pd
import numpy as np
from statistics import mean
from scipy import stats
import matplotlib.pyplot as plt
import sys

aaa = list(range(4,5))
Heat_A = pd.read_excel (r'C:\Users\neder\Documents\Geomatics\1001\Assignment 1\hw01\HEAT - A_final.xls', header=3,skiprows=[4,2479,2480])    #import excel file
Heat_B = pd.read_excel (r'C:\Users\neder\Documents\Geomatics\1001\Assignment 1\hw01\HEAT - B_final.xls', header=3,skiprows=[4,2479,2480])
Heat_C = pd.read_excel (r'C:\Users\neder\Documents\Geomatics\1001\Assignment 1\hw01\HEAT - C_final.xls', header=3,skiprows=[4])
Heat_D = pd.read_excel (r'C:\Users\neder\Documents\Geomatics\1001\Assignment 1\hw01\HEAT - D_final.xls', header=3,skiprows=[4])
Heat_E = pd.read_excel (r'C:\Users\neder\Documents\Geomatics\1001\Assignment 1\hw01\HEAT - E_final.xls', header=3,skiprows=[4,2479])

wind_speeds_A = Heat_A.iloc[:,2]                                                                          #create series 
wind_speeds_B = Heat_B.iloc[:,2]
wind_speeds_C = Heat_C.iloc[:,2]
wind_speeds_D = Heat_D.iloc[:,2]
wind_speeds_E = Heat_E.iloc[:,2]


# =============================================================================
#   PRINTING THE MEAN VARIANCE AND STANDARD DEVIATION
# =============================================================================
print("Heat A means")
print(Heat_A.mean(axis = 0, numeric_only=True))

print("Heat A variance")
print(Heat_A.var(axis = 0, numeric_only=True))

print("Heat A standard deviation")
print(Heat_A.std(axis = 0, numeric_only=True))

print("Heat B means")
print(Heat_B.mean(axis = 0, numeric_only=True))

print("Heat B variance")
print(Heat_B.var(axis = 0, numeric_only=True))

print("Heat B standard deviation")
print(Heat_B.std(axis = 0, numeric_only=True))

print("Heat C means")
print(Heat_C.mean(axis = 0, numeric_only=True))

print("Heat C variance")
print(Heat_C.var(axis = 0, numeric_only=True))

print("Heat C standard deviation")
print(Heat_C.std(axis = 0, numeric_only=True))

print("Heat D means")
print(Heat_D.mean(axis = 0, numeric_only=True))

print("Heat D variance")
print(Heat_D.var(axis = 0, numeric_only=True))

print("Heat D standard deviation")
print(Heat_D.std(axis = 0, numeric_only=True))

print("Heat E means")
print(Heat_E.mean(axis = 0, numeric_only=True))

print("Heat E variance")
print(Heat_E.var(axis = 0, numeric_only=True))

print("Heat E standard deviation")
print(Heat_E.std(axis = 0, numeric_only=True))
all_means = [(Heat_A.mean(axis = 0, numeric_only=True)), (Heat_B.mean(axis = 0, numeric_only=True)), (Heat_C.mean(axis = 0, numeric_only=True)), (Heat_D.mean(axis = 0, numeric_only=True)), (Heat_E.mean(axis = 0, numeric_only=True))]
all_variance= [(Heat_A.var(axis = 0, numeric_only=True)), (Heat_B.var(axis = 0, numeric_only=True)), (Heat_C.var(axis = 0, numeric_only=True)), (Heat_D.var(axis = 0, numeric_only=True)), (Heat_E.var(axis = 0, numeric_only=True))]
all_std= [(Heat_A.std(axis = 0, numeric_only=True)), (Heat_B.std(axis = 0, numeric_only=True)), (Heat_C.std(axis = 0, numeric_only=True)), (Heat_D.std(axis = 0, numeric_only=True)), (Heat_E.std(axis = 0, numeric_only=True))]
# =============================================================================
# PLOTS
# =============================================================================
Temp = [Heat_A.iloc[:,5], Heat_B.iloc[:,5], Heat_C.iloc[:,5], Heat_D.iloc[:,5], Heat_E.iloc[:,5]]
Windspeed = [Heat_A.iloc[:,2], Heat_B.iloc[:,2], Heat_C.iloc[:,2], Heat_D.iloc[:,2], Heat_E.iloc[:,2]]
Wind_direction = [Heat_A.iloc[:,1], Heat_B.iloc[:,1], Heat_C.iloc[:,1], Heat_D.iloc[:,1], Heat_E.iloc[:,1]]

colors = ['r','b','c','g','y']
fig = plt.figure(figsize=(8,6))
plt.hist(x=Temp, bins=10, density=True, histtype='bar', color=colors, label=['A','B','C','D','E'])
plt.xlabel('Temperature [Celcius]',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Temperature Histogam')
plt.legend()
plt.show()


# =============================================================================
# FREQUENCY POLIGONS
# =============================================================================
bins1 = 20
temp_cuma = stats.cumfreq(Temp[0], numbins = bins1)
temp_cumb = stats.cumfreq(Temp[1], numbins = bins1)
temp_cumc = stats.cumfreq(Temp[2], numbins = bins1)
temp_cumd = stats.cumfreq(Temp[3], numbins = bins1)
temp_cume = stats.cumfreq(Temp[4], numbins = bins1)

temp_gxs = [temp_cuma.lowerlimit + np.linspace(0, temp_cuma.binsize*temp_cuma.cumcount.size,
                                 temp_cuma.cumcount.size),temp_cumb.lowerlimit + np.linspace(0, temp_cumb.binsize*temp_cumb.cumcount.size,
                                 temp_cumb.cumcount.size),temp_cumc.lowerlimit + np.linspace(0, temp_cumc.binsize*temp_cumc.cumcount.size,
                                 temp_cumc.cumcount.size),temp_cumd.lowerlimit + np.linspace(0, temp_cumd.binsize*temp_cumd.cumcount.size,
                                 temp_cumd.cumcount.size),temp_cume.lowerlimit + np.linspace(0, temp_cume.binsize*temp_cume.cumcount.size,
                                 temp_cume.cumcount.size)]
temp_gxa = temp_cuma.lowerlimit + np.linspace(0, temp_cuma.binsize*temp_cuma.cumcount.size,
                                 temp_cuma.cumcount.size)
temp_gxb = temp_cumb.lowerlimit + np.linspace(0, temp_cumb.binsize*temp_cumb.cumcount.size,
                                 temp_cumb.cumcount.size)
temp_gxc = temp_cumc.lowerlimit + np.linspace(0, temp_cumc.binsize*temp_cumc.cumcount.size,
                                 temp_cumc.cumcount.size)
temp_gxd = temp_cumd.lowerlimit + np.linspace(0, temp_cumd.binsize*temp_cumd.cumcount.size,
                                 temp_cumd.cumcount.size)
temp_gxe = temp_cume.lowerlimit + np.linspace(0, temp_cume.binsize*temp_cume.cumcount.size,
                                 temp_cume.cumcount.size)
cums = [temp_cuma.cumcount,temp_cumb.cumcount,temp_cumc.cumcount,temp_cumd.cumcount,temp_cume.cumcount]
cuma = temp_cuma.cumcount
cumb = temp_cumb.cumcount
cumc = temp_cumc.cumcount
cumd = temp_cumd.cumcount
cume = temp_cume.cumcount

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(temp_gxa, cuma, label='A');
ax.plot(temp_gxb, cumb, label='B');
ax.plot(temp_gxc, cumc, label='C');
ax.plot(temp_gxd, cumd, label='D');
ax.plot(temp_gxe, cume, label='E');
ax.legend()
ax.set_title('Frequency Polygon temperature')
ax.set_xlabel('Temperature [Celcius]')



# =============================================================================
# BOXPLOTS
# =============================================================================

color = 'tab:red'
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=False)
axs[0].boxplot(x=Temp)
axs[0].set_title('temperature')
axs[0].set_ylabel('Temperature [Celcius]')
axs[0].tick_params(axis = 'y', labelcolor=color)
axs[1].boxplot(x=Windspeed)
axs[1].set_title('windspeed')
axs[1].set_ylabel('Windspeed [m/s]')
axs[1].tick_params(axis = 'y', labelcolor=color)
axs[2].boxplot(x=Wind_direction)
axs[2].set_title('wind direction')
axs[2].set_ylabel('Wind direction [deg]')
axs[2].tick_params(axis = 'y', labelcolor=color)
plt.sca(axs[0])
plt.xticks(range(6), [' ','A','B','C','D','E'])
plt.sca(axs[1])
plt.xticks(range(6), [' ','A','B','C','D','E'])
plt.sca(axs[2])
plt.xticks(range(6), [' ','A','B','C','D','E'])
plt.tight_layout()



# =============================================================================
# PMF
# =============================================================================

PMFa = Temp[0].value_counts().sort_index() / len(Temp[0])
PMFb = Temp[1].value_counts().sort_index() / len(Temp[1])
PMFc = Temp[2].value_counts().sort_index() / len(Temp[2])
PMFd = Temp[3].value_counts().sort_index() / len(Temp[3])
PMFe = Temp[4].value_counts().sort_index() / len(Temp[4])
PMF = [PMFa, PMFb, PMFc, PMFd, PMFe]
                                                            #database plot

def pmf(sample):                                    
    c = sample.value_counts()
    p = c/len(sample)
    return p                                
df44 = pmf(Heat_A['Temperature'])       
df444 = df44.to_frame()
df444.reset_index(inplace=True)
df444.columns = ['T', 'PMF']
df444 = df444.astype(float)
c = df444.sort_values(by=['T'])
fig = plt.figure(figsize=(16,6))
ax4 = fig.add_subplot(111)
ax4.bar(x=c['T'], height=c['PMF'], width=0.00005, edgecolor='k')
plt.show()

pmfl = pmf(Temp[0])
pmflsort = pmfl.sort_index()
fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.bar(pmflsort.index,pmflsort, edgecolor='k')
ax1.set_title('pmf a')
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()

pmflb = pmf(Temp[1])
pmflbsort = pmflb.sort_index()
fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.bar(pmflbsort.index,pmflbsort, edgecolor='k')
ax1.set_title('pmf b')
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()

pmflc = pmf(Temp[2])
pmflcsort = pmflc.sort_index()
fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.bar(pmflcsort.index,pmflcsort, edgecolor='k')
ax1.set_title('pmf c')
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()

pmfld = pmf(Temp[3])
pmfldsort = pmfld.sort_index()
fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.bar(pmfldsort.index,pmfldsort, edgecolor='k')
ax1.set_title('pmf d')
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()

pmfle = pmf(Temp[4])
pmflesort = pmfle.sort_index()
fig = plt.figure(figsize=(17,6))
ax1 = fig.add_subplot(111)
ax1.bar(pmflesort.index,pmflesort, edgecolor='k')
ax1.set_title('pmf e')
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()



# =============================================================================
# PDF
# =============================================================================
binspdf = np.linspace(0, 1, 50) 
histogramA, binspdfA = np.histogram(Temp[0], density=True)
histogramB, binspdfB = np.histogram(Temp[1], density=True)
histogramC, binspdfC = np.histogram(Temp[2], density=True)
histogramD, binspdfD = np.histogram(Temp[3], density=True)
histogramE, binspdfE = np.histogram(Temp[4], density=True)
binspdf_centerA = 0.5*(binspdfA[1:] + binspdfA[:-1])
binspdf_centerB = 0.5*(binspdfB[1:] + binspdfB[:-1])
binspdf_centerC = 0.5*(binspdfC[1:] + binspdfC[:-1])
binspdf_centerD = 0.5*(binspdfD[1:] + binspdfD[:-1])
binspdf_centerE = 0.5*(binspdfE[1:] + binspdfE[:-1])



fig = plt.figure(figsize=(15,15))                                                  
ax2 = fig.add_subplot(111)
ax2.set_title('PDF A')
ax2.hist(x=Temp[0].astype(float), density=True, cumulative=False, bins=40, alpha=0.7, rwidth=0.85)
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()
fig = plt.figure(figsize=(15,15))                                                  
ax2 = fig.add_subplot(111)
ax2.set_title('PDF B')
ax2.hist(x=Temp[1].astype(float), density=True, cumulative=False, bins=40, alpha=0.7, rwidth=0.85)
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()
fig = plt.figure(figsize=(15,15))                                                 
ax2 = fig.add_subplot(111)
ax2.set_title('PDF C')
ax2.hist(x=Temp[2].astype(float), density=True, cumulative=False, bins=40, alpha=0.7, rwidth=0.85)
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()
fig = plt.figure(figsize=(15,15))                                                  
ax2 = fig.add_subplot(111)
ax2.set_title('PDF D')
ax2.hist(x=Temp[3].astype(float), density=True, cumulative=False, bins=40, alpha=0.7, rwidth=0.85)
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()
fig = plt.figure(figsize=(15,15))                                                  
ax2 = fig.add_subplot(111)
ax2.set_title('PDF E')
ax2.hist(x=Temp[4].astype(float), density=True, cumulative=False, bins=40, alpha=0.7, rwidth=0.85)
plt.xlabel('Temperature [Celcius]')
plt.ylabel('probability')
plt.show()


# =============================================================================
# CDF
# =============================================================================
 ########################## SEE FIGURE UNDER CONFIDENCE INTERVAL#############################

# =============================================================================
# pdf and KERNEL DENSITY for wind speed
# =============================================================================

plt.figure()
s = [pd.Series(Windspeed[0]), pd.Series(Windspeed[1]), pd.Series(Windspeed[2]), pd.Series(Windspeed[3]), pd.Series(Windspeed[4])]
ax6 = s[0].plot.kde()
ax6.hist(x=Windspeed[0].astype(float), density=True, cumulative=False, bins=30, alpha=0.7, rwidth=0.85)
plt.xlabel('windspeeds [m/s]')
plt.title('kernel A')

plt.figure()
ax6 = s[1].plot.kde()
ax6.hist(x=Windspeed[1].astype(float), density=True, cumulative=False, bins=30, alpha=0.7, rwidth=0.85)
plt.xlabel('windspeeds [m/s]')
plt.title('kernel B')
plt.show()

plt.figure()
ax6 = s[2].plot.kde()
ax6.hist(x=Windspeed[2].astype(float), density=True, cumulative=False, bins=30, alpha=0.7, rwidth=0.85)
plt.xlabel('windspeeds [m/s]')
plt.title('kernel C')
plt.show()

plt.figure()
ax6 = s[3].plot.kde()
ax6.hist(x=Windspeed[3].astype(float), density=True, cumulative=False, bins=30, alpha=0.7, rwidth=0.85)
plt.xlabel('windspeeds [m/s]')
plt.title('kernel D')
plt.show()

plt.figure()
ax6 = s[4].plot.kde()
ax6.hist(x=Windspeed[4].astype(float), density=True, cumulative=False, bins=30, alpha=0.7, rwidth=0.85)
plt.xlabel('windspeeds [m/s]')
plt.title('kernel E')
plt.show()



# =============================================================================
# CORRELATIONS
# =============================================================================

def correlations(A, B):
    spearman = stats.spearmanr(A, B).correlation
    corr, _ = stats.pearsonr(A, B)
    
    return spearman, corr
    
def corr_matrix(row):
    corrs = [[' ', 'A', 'B', 'C', 'D', 'E'], 
         ['A',  correlations(Heat_A.iloc[:,row], Heat_B.iloc[:,row]), correlations(Heat_A.iloc[:,row], Heat_C.iloc[:,row]), correlations(Heat_A.iloc[:,row], Heat_D.iloc[:,row]), correlations(Heat_A.iloc[:,row], Heat_E.iloc[:,row])],
         ['B',  correlations(Heat_B.iloc[:,row], Heat_A.iloc[:,row]), correlations(Heat_B.iloc[:,row], Heat_C.iloc[:,row]), correlations(Heat_B.iloc[:,row], Heat_D.iloc[:,row]), correlations(Heat_B.iloc[:,row], Heat_E.iloc[:,row])],
         ['C',  correlations(Heat_C.iloc[:,row], Heat_A.iloc[:,row]),  correlations(Heat_C.iloc[:,row], Heat_B.iloc[:,row]), correlations(Heat_C.iloc[:,row], Heat_D.iloc[:,row]), correlations(Heat_C.iloc[:,row], Heat_E.iloc[:,row])],
         ['D',  correlations(Heat_D.iloc[:,row], Heat_A.iloc[:,row]),  correlations(Heat_D.iloc[:,row], Heat_B.iloc[:,row]), correlations(Heat_D.iloc[:,row], Heat_C.iloc[:,row]), correlations(Heat_D.iloc[:,row], Heat_E.iloc[:,row])],
         ['E',  correlations(Heat_E.iloc[:,row], Heat_A.iloc[:,row]),  correlations(Heat_E.iloc[:,row], Heat_B.iloc[:,row]), correlations(Heat_E.iloc[:,row], Heat_C.iloc[:,row]), correlations(Heat_E.iloc[:,row], Heat_D.iloc[:,row])]]
    return corrs
Temp_corrs = corr_matrix(5)
WBG_corrs = corr_matrix(16)
CWS_corrs = corr_matrix(3)
def getcolumn(corrs, A, spearman):              #(list, 1= 2 2=b etc, 0=spearman 1=pearson)
    column = [corrs[A][1][spearman], corrs[A][2][spearman], corrs[A][3][spearman], corrs[A][4][spearman]]
    return column

T_spearmancorrs = [getcolumn(Temp_corrs, 1, 0), getcolumn(Temp_corrs, 2, 0), getcolumn(Temp_corrs, 3, 0), getcolumn(Temp_corrs, 4, 0), getcolumn(Temp_corrs, 5, 0)]
T_pearsoncorrs = [getcolumn(Temp_corrs, 1, 1), getcolumn(Temp_corrs, 2, 1), getcolumn(Temp_corrs, 3, 1), getcolumn(Temp_corrs, 4, 1), getcolumn(Temp_corrs, 5, 1)]
WBG_spearmancorrs = [getcolumn(WBG_corrs, 1, 0), getcolumn(WBG_corrs, 2, 0), getcolumn(WBG_corrs, 3, 0), getcolumn(WBG_corrs, 4, 0), getcolumn(WBG_corrs, 5, 0)]
WBG_pearsoncorrs = [getcolumn(WBG_corrs, 1, 1), getcolumn(WBG_corrs, 2, 1), getcolumn(WBG_corrs, 3, 1), getcolumn(WBG_corrs, 4, 1), getcolumn(WBG_corrs, 5, 1)]
CWS_spearmancorrs = [getcolumn(CWS_corrs, 1, 0), getcolumn(CWS_corrs, 2, 0), getcolumn(CWS_corrs, 3, 0), getcolumn(CWS_corrs, 4, 0), getcolumn(CWS_corrs, 5, 0)]
CWS_pearsoncorrs = [getcolumn(CWS_corrs, 1, 1), getcolumn(CWS_corrs, 2, 1), getcolumn(CWS_corrs, 3, 1), getcolumn(CWS_corrs, 4, 1), getcolumn(CWS_corrs, 5, 1)]

plt.figure(figsize=(15,10))
legend_x = -0.16
legend_y = 0.5

plt.scatter(('A', 'C', 'D', 'E'),T_spearmancorrs[1], label='B spearman')
plt.scatter(( 'B', 'C', 'D', 'E'),T_spearmancorrs[0], label='A spearman')
plt.scatter(('A', 'B', 'D', 'E'),T_spearmancorrs[2], label='C spearman')
plt.scatter(('A', 'B', 'C', 'E'),T_spearmancorrs[3], label='D spearman')
plt.scatter(('A', 'B', 'C', 'D'),T_spearmancorrs[4], label='E spearman')
plt.scatter(('B', 'C', 'D', 'E'),T_pearsoncorrs[0], marker='*', label='A pearson')
plt.scatter(('A', 'C', 'D', 'E'),T_pearsoncorrs[1], marker='*', label='B pearson')
plt.scatter(('A', 'B', 'D', 'E'),T_pearsoncorrs[2], marker='*', label='C pearson')
plt.scatter(('A', 'B', 'C', 'E'),T_pearsoncorrs[3], marker='*', label='D pearson')
plt.scatter(('A', 'B', 'C', 'D'),T_pearsoncorrs[4], marker='*', label='E pearson')
plt.title('Temp corrs')
plt.legend( loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show

plt.figure(figsize=(15,10))
plt.scatter(( 'B', 'C', 'D', 'E'),WBG_spearmancorrs[0], label='A spearman')
plt.scatter(('A',  'C', 'D', 'E'),WBG_spearmancorrs[1], label='B spearman')
plt.scatter(('A', 'B', 'D', 'E'),WBG_spearmancorrs[2], label='C spearman')
plt.scatter(('A', 'B', 'C', 'E'),WBG_spearmancorrs[3], label='D spearman')
plt.scatter(('A', 'B', 'C', 'D'),WBG_spearmancorrs[4], label='E spearman')
plt.scatter(( 'B', 'C', 'D', 'E'),WBG_pearsoncorrs[0], marker='*', label='A pearson')
plt.scatter(('A', 'C', 'D', 'E'),WBG_pearsoncorrs[1], marker='*', label='B pearson')
plt.scatter(('A', 'B', 'D', 'E'),WBG_pearsoncorrs[2], marker='*', label='C pearson')
plt.scatter(('A', 'B', 'C', 'E'),WBG_pearsoncorrs[3], marker='*', label='D pearson')
plt.scatter(('A', 'B', 'C', 'D'),WBG_pearsoncorrs[4], marker='*', label='E pearson')
plt.title('WBGT corrs')
plt.legend( loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()

plt.figure(figsize=(15,10))
plt.scatter(( 'B', 'C', 'D', 'E'),CWS_spearmancorrs[0], label='A spearman')
plt.scatter(('A', 'C', 'D', 'E'),CWS_spearmancorrs[1], label='B spearman')
plt.scatter(('A', 'B', 'D', 'E'),CWS_spearmancorrs[2], label='C spearman')
plt.scatter(('A', 'B', 'C',  'E'),CWS_spearmancorrs[3], label='D spearman')
plt.scatter(('A', 'B', 'C', 'D'),CWS_spearmancorrs[4], label='E spearman')
plt.scatter(( 'B', 'C', 'D', 'E'),CWS_pearsoncorrs[0], marker='*', label='A pearson')
plt.scatter(('A', 'C', 'D', 'E'),CWS_pearsoncorrs[1], marker='*', label='B pearson')
plt.scatter(('A', 'B',  'D', 'E'),CWS_pearsoncorrs[2], marker='*', label='C pearson')
plt.scatter(('A', 'B', 'C', 'E'),CWS_pearsoncorrs[3], marker='*', label='D pearson')
plt.scatter(('A', 'B', 'C', 'D'),CWS_pearsoncorrs[4], marker='*', label='E pearson')
plt.title('CWS corrs')
plt.legend( loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()


# =============================================================================
# CDFs & confidence intervals
# =============================================================================

color = 'black'
fig, axs = plt.subplots(1, 5, figsize=(30, 20), sharey=False)
a9=axs[0].hist(x=Temp[0].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[0].set_title('CDF temperature A',fontsize=15)
axs[0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[0].tick_params(axis = 'y', labelcolor=color)

a9=axs[1].hist(x=Temp[1].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[1].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[1].set_title('CDF temperature B',fontsize=15)
axs[1].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[1].tick_params(axis = 'y', labelcolor=color)

a9=axs[2].hist(x=Temp[2].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[2].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[2].set_title('CDF temperature C',fontsize=15)
axs[2].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[2].tick_params(axis = 'y', labelcolor=color)

a9=axs[3].hist(x=Temp[3].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[3].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[3].set_title('CDF temperature D',fontsize=15)
axs[3].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[3].tick_params(axis = 'y', labelcolor=color)

a9=axs[4].hist(x=Temp[4].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[4].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[4].set_title('CDF temperature E',fontsize=15)
axs[4].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[4].tick_params(axis = 'y', labelcolor=color)

plt.tight_layout(pad=10, w_pad=5, h_pad=20)




def interval_95(data):
    interv= stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    return interv
Temp_intervals_95 = [interval_95(Heat_A.iloc[:,5]), interval_95(Heat_B.iloc[:,5]), interval_95(Heat_C.iloc[:,5]), interval_95(Heat_D.iloc[:,5]), interval_95(Heat_E.iloc[:,5])]
Windspeed_intervals_95 = [interval_95(Heat_A.iloc[:,2]), interval_95(Heat_B.iloc[:,2]), interval_95(Heat_C.iloc[:,2]), interval_95(Heat_D.iloc[:,2]), interval_95(Heat_E.iloc[:,2])]
intervals_95 = [interval_95(Heat_A.iloc[:,5]), interval_95(Heat_B.iloc[:,5]), interval_95(Heat_C.iloc[:,5]), interval_95(Heat_D.iloc[:,5]), interval_95(Heat_E.iloc[:,5]), interval_95(Heat_A.iloc[:,2]), interval_95(Heat_B.iloc[:,2]), interval_95(Heat_C.iloc[:,2]), interval_95(Heat_D.iloc[:,2]), interval_95(Heat_E.iloc[:,2])]

matrix_ints = np.matrix(intervals_95)               ######## creating txt file ########
with open('conf95int.txt','wb') as x:
    for values in matrix_ints:
        np.savetxt(x, values, fmt='%.2f')


color = 'black'
fig, axs = plt.subplots(5, 2, figsize=(15, 45), sharey=False)
a9=axs[0][0].hist(x=Temp[0].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[0][0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[0][0].axvline(x=Temp_intervals_95[0][0], ymin=0, ymax=1, color = 'r')
axs[0][0].axvline(x=Temp_intervals_95[0][1], ymin=0, ymax=1, color = 'r')
axs[0][0].set_title('CDF temperature A',fontsize=15)
axs[0][0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[0][0].tick_params(axis = 'y', labelcolor=color)
a10=axs[0][1].hist(x=Windspeed[0].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[0][1].plot(a10[1][1:]-(a10[1][1:]-a10[1][:-1])/2,a10[0], color='k')
axs[0][1].axvline(x=Windspeed_intervals_95[0][0], ymin=0, ymax=1, color = 'r')
axs[0][1].axvline(x=Windspeed_intervals_95[0][1], ymin=0, ymax=1, color = 'r')
axs[0][1].set_title('CDF windspeed A',fontsize=15)
axs[0][1].set_xlabel('Windspeed [m/s]',fontsize=8)
a9=axs[1][0].hist(x=Temp[1].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[1][0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[1][0].axvline(x=Temp_intervals_95[1][0], ymin=0, ymax=1, color = 'r')
axs[1][0].axvline(x=Temp_intervals_95[1][1], ymin=0, ymax=1, color = 'r')
axs[1][0].set_title('CDF temperature B',fontsize=15)
axs[1][0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[1][0].tick_params(axis = 'y', labelcolor=color)
a10=axs[1][1].hist(x=Windspeed[1].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[1][1].plot(a10[1][1:]-(a10[1][1:]-a10[1][:-1])/2,a10[0], color='k')
axs[1][1].axvline(x=Windspeed_intervals_95[1][0], ymin=0, ymax=1, color = 'r')
axs[1][1].axvline(x=Windspeed_intervals_95[1][1], ymin=0, ymax=1, color = 'r')
axs[1][1].set_title('CDF windspeed B',fontsize=15)
axs[1][1].set_xlabel('Windspeed [m/s]',fontsize=8)
a9=axs[2][0].hist(x=Temp[2].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[2][0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[2][0].axvline(x=Temp_intervals_95[2][0], ymin=0, ymax=1, color = 'r')
axs[2][0].axvline(x=Temp_intervals_95[2][1], ymin=0, ymax=1, color = 'r')
axs[2][0].set_title('CDF temperature C',fontsize=15)
axs[2][0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[2][0].tick_params(axis = 'y', labelcolor=color)
a10=axs[2][1].hist(x=Windspeed[2].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[2][1].plot(a10[1][1:]-(a10[1][1:]-a10[1][:-1])/2,a10[0], color='k')
axs[2][1].axvline(x=Windspeed_intervals_95[2][0], ymin=0, ymax=1, color = 'r')
axs[2][1].axvline(x=Windspeed_intervals_95[2][1], ymin=0, ymax=1, color = 'r')
axs[2][1].set_title('CDF windspeed C',fontsize=15)
axs[2][1].set_xlabel('Windspeed [m/s]',fontsize=8)
a9=axs[3][0].hist(x=Temp[3].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[3][0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[3][0].axvline(x=Temp_intervals_95[3][0], ymin=0, ymax=1, color = 'r')
axs[3][0].axvline(x=Temp_intervals_95[3][1], ymin=0, ymax=1, color = 'r')
axs[3][0].set_title('CDF temperature D',fontsize=15)
axs[3][0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[3][0].tick_params(axis = 'y', labelcolor=color)
a10=axs[3][1].hist(x=Windspeed[3].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[3][1].plot(a10[1][1:]-(a10[1][1:]-a10[1][:-1])/2,a10[0], color='k')
axs[3][1].axvline(x=Windspeed_intervals_95[3][0], ymin=0, ymax=1, color = 'r')
axs[3][1].axvline(x=Windspeed_intervals_95[3][1], ymin=0, ymax=1, color = 'r')
axs[3][1].set_title('CDF windspeed D',fontsize=15)
axs[3][1].set_xlabel('Windspeed [m/s]',fontsize=8)
a9=axs[4][0].hist(x=Temp[4].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[4][0].plot(a9[1][1:]-(a9[1][1:]-a9[1][:-1])/2,a9[0], color='k')
axs[4][0].axvline(x=Temp_intervals_95[4][0], ymin=0, ymax=1, color = 'r')
axs[4][0].axvline(x=Temp_intervals_95[4][1], ymin=0, ymax=1, color = 'r')
axs[4][0].set_title('CDF temperature E',fontsize=15)
axs[4][0].set_xlabel('Temperature [Celcius]',fontsize=8)
axs[4][0].tick_params(axis = 'y', labelcolor=color)
a10=axs[4][1].hist(x=Windspeed[4].astype(float), bins=30, cumulative=True, color='b',alpha=0.7, rwidth=0.85, normed=True)
axs[4][1].plot(a10[1][1:]-(a10[1][1:]-a10[1][:-1])/2,a10[0], color='k')
axs[4][1].axvline(x=Windspeed_intervals_95[4][0], ymin=0, ymax=1, color = 'r')
axs[4][1].axvline(x=Windspeed_intervals_95[4][1], ymin=0, ymax=1, color = 'r')
axs[4][1].set_title('CDF windspeed E',fontsize=15)
axs[4][1].set_xlabel('Windspeed [m/s]',fontsize=8)
plt.tight_layout(pad=10, w_pad=5, h_pad=20)


# =============================================================================
# HYPOTHESIS TEST
# =============================================================================


t, p = stats.ttest_ind(Temp[4],Temp[3])
print("Temp E-D p = " + str(p))
t, p = stats.ttest_ind(Temp[3],Temp[2])
print("Temp C-D p = " + str(p))
t, p = stats.ttest_ind(Temp[2],Temp[1])
print("Temp B-C p = " + str(p))
t, p = stats.ttest_ind(Temp[1],Temp[0])
print("Temp A-B p = " + str(p))

t, p = stats.ttest_ind(Windspeed[4],Windspeed[3])
print("Windspeed E-D p = " + str(p))
t, p = stats.ttest_ind(Windspeed[3],Windspeed[2])
print("Windspeed C-D p = " + str(p))
t, p = stats.ttest_ind(Windspeed[2],Windspeed[1])
print("Windspeed B-C p = " + str(p))
t, p = stats.ttest_ind(Windspeed[1],Windspeed[0])
print("Windspeed A-B p = " + str(p))


# =============================================================================
# BONUS QUESTION
# =============================================================================

Ahot = (i for i, n in enumerate(Temp[0]) if n >= 31) 
print('sensor A',  list(Ahot))

Bhot = (i for i, n in enumerate(Temp[1]) if n >= 30) 
print('sensor b',  list(Bhot))
Chot = (i for i, n in enumerate(Temp[2]) if n >= 30) 
print('sensor c',  list(Chot))
Dhot = (i for i, n in enumerate(Temp[3]) if n >= 30) 
print('sensor d',  list(Dhot))
Ehot = (i for i, n in enumerate(Temp[4]) if n >= 30) 
print('sensor e',  list(Ehot))





