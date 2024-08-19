#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:35:16 2022

@author: randong
"""

import csv
import numpy as np
from utils.MEMD_all import memd
from utils.MEMD_all import avgF, avgA
import utils.ht as hs
import matplotlib.pyplot as plt
import argparse
from scipy.fftpack import fft
from scipy.fftpack import ifft

from utils.CurvatureNumeric import calc_curvature_range_kutta, calc_curvature_2_derivative 


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result


# Set up argument parser
parser = argparse.ArgumentParser(description='Process MEMD options.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output_opt', type=str, required=True, help='Path to the output directory')
parser.add_argument('--slow', type=float, default=1.0, help='Slow parameter (default: 1.0)')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter (default: 0.5)')

args = parser.parse_args()

output_csv = args.input_csv
output_opt = args.output_opt


slow = args.slow
alpha = args.alpha

####################################################################################################
 
#Import NAO6 motion data for decomposition using MEMD
with open(output_csv) as f:
    reader = csv.reader(f)
    l = [row for row in reader]

motors = l[0][1:]
data = np.array(l[1:], dtype = np.float64)[:,1:]
time = np.array(l[1:], dtype = np.float64)[:,0]
dt = time[1] - time[0]

#Orignal DOFs plot

#selected one motion in the first 400 frames
data = np.rad2deg(data[:400])
t = np.linspace(0, data.shape[0] * dt,data.shape[0])

#important joints
immotors = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "RShoulderPitch", "RShoulderRoll", "RHipYawPitch", "RHipRoll", "RHipPitch", "LHipYawPitch", "LHipRoll", "LHipPitch"]
#immotors = ["HeadYaw", "HeadPitch", "RShoulderPitch", "RShoulderRoll", "RHipYawPitch", "RHipRoll", "RHipPitch"]

plt.figure(figsize=(15,5), dpi = 150)
plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams["font.size"] = 20
plt.rcParams['axes.linewidth'] = 3     #軸の太さ
plt.rcParams['legend.fontsize'] = 20

plt.plot(t, data[:,12], label = "RHipYawPitch" , linewidth=3)   
for i in range(data.shape[1]):
    if motors[i][1:] in immotors:
        plt.plot(t, data[:,i], label = motors[i][1:], linewidth=3)

plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0)
plt.title("Original motion capture data")
plt.xlabel("Time [$s$]")
plt.ylabel("Angle [$deg$]")
plt.xlim(t[0], t[-1])
plt.savefig("orignal.pdf", bbox_inches='tight')
plt.show()

########################################################################################################

#decompose NAO6 motiondata using MEMD 
imf = memd(data)#imfs are the decomposed nonlinear frequency components
nIMFs = imf.shape[0]

plt.figure(figsize=(18,25), dpi = 150)
lineint = 2
for n in range(nIMFs - 1):
    plt.subplot(nIMFs+1, 3, n+1)
    
    plt.plot(t, imf[n,12, :], label = "RHipYawPitch" , linewidth=lineint)   
    for i in range(data.shape[1]):
        if motors[i][1:] in immotors:
            plt.plot(t, imf[n,i], label = motors[i][1:], linewidth=lineint)
    
    # plt.legend()
    plt.xlabel("Time [$s$]")
    plt.ylabel("Angle [$deg$]")
    plt.xlim(t[0], t[-1])
    plt.title("Original")
    plt.title("IMF$_{%i}$" %(n+1))
    

plt.subplot(nIMFs+1, 3, nIMFs)
plt.plot(t, imf[nIMFs-1,12,:], label = "RHipYawPitch" , linewidth=lineint)   
for i in range(data.shape[1]):
    if motors[i][1:] in immotors:
        plt.plot(t, imf[nIMFs-1,i ,:], label = motors[i][1:], linewidth=lineint)

# plt.legend()
plt.xlabel("Time [$s$]")
plt.ylabel("Angle [$deg$]")
plt.xlim(t[0], t[-1])
plt.title("Trend")
plt.tight_layout()
plt.savefig("MEMDdecomposed.pdf", bbox_inches='tight')
plt.show()

########################################################################################################

#Calculate instantaneous frequencies and amplitudes using Hilbert transform

allfreq = []
allamp = []
for i in range(data.shape[1]):
    freq, amp = hs.FAhilbert(imf[:,i], dt)
    allfreq.append(freq)
    allamp.append(amp)

allfreq = np.array(allfreq)
allamp = np.array(allamp)

freq = np.mean(allfreq,axis=0)
amp = np.linalg.norm(allamp, ord=2, axis=0)

n = freq.shape[0]
m = amp.shape[1]



########################################################################################################

#plot motion spectrum
t2=np.zeros((n,m))
for i in range(m):
    t2[:,i] = t

plt.figure(dpi=200, figsize=(15,9))

plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams["font.size"] = 40

plt.scatter(t2, freq, s=100, c=np.log(amp[:-1]), cmap='jet')
ax = plt.gca()
ax.set_facecolor([0.0,0.0,0.5])
plt.ylim(0, 50)
plt.xlim(0, t[-1])
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]', labelpad=20) 
plt.colorbar(label = "Amplitude [deg] (log scale)")
plt.show()



numplot = 20

amplog = np.log(amp[:-1])
maxv = np.max(amplog)
minv = np.min(amplog)


plt.rcParams["font.size"] = 25
plt.rcParams['axes.linewidth'] = 2
plt.figure(dpi=200, figsize=(10,15))

plt.subplot(3, 1, 1)
    
plt.scatter(t2[:,:2], freq[:,:2], s=numplot, c=amplog[:,:2], cmap='jet')
ax = plt.gca()
ax.set_facecolor([0.0,0.0,0.5])
plt.ylim(0, 50)
plt.xlim(0, t[-1])
plt.clim(minv, maxv)
plt.title("IMF$_{1-2}$")    #タイトルを付ける
plt.xlabel('Time [$s$]')        #x軸に名前をつける
plt.ylabel('Frequency [Hz]') #y軸に名前をつける
plt.colorbar(label = "Amplitude [$deg$] (log scale)")
#plt.show()


plt.subplot(3, 1, 2)

plt.scatter(t2[:,2:5], freq[:,2:5], s=numplot, c=amplog[:,2:5], cmap='jet')
ax = plt.gca()
ax.set_facecolor([0.0,0.0,0.5])
plt.ylim(0, 20)
plt.xlim(0, t[-1])
plt.clim(minv, maxv)
plt.title("IMF$_{3-5}$")  
plt.xlabel('Time [$s$]')       
plt.ylabel('Frequency [Hz]') 
plt.colorbar(label = "Amplitude [$deg$] (log scale)")
#plt.show()


plt.subplot(3, 1, 3)
    
plt.scatter(t2[:,5:], freq[:,5:], s=numplot, c=amplog[:,5:], cmap='jet')
ax = plt.gca()
ax.set_facecolor([0.0,0.0,0.5])
plt.ylim(0, 3)
plt.xlim(0, t[-1])
plt.clim(minv, maxv)
plt.title("IMF$_{6-8}$")    
plt.xlabel('Time [$s$]')      
plt.ylabel('Frequency [Hz]') 
plt.colorbar(label = "Amplitude [$deg$] (log scale)")

plt.tight_layout()
plt.savefig("spectrum.pdf", bbox_inches='tight')
plt.show()



########################################################################################################

#evaluation for all motors

#calculate average frequencies and amplitudes
avgfreq = avgF(imf,dt)
avgamp = avgA(imf,dt)
freqSD = np.std(avgfreq,axis = 1)
ampSD = np.std(avgamp,axis =1)
allavgfreq = np.mean(avgF(imf,dt), axis=1)
allavgamp = np.mean(avgA(imf,dt), axis=1)

sd = []
mse = []
maxspeed = []
maxaccelerate = []
maxcurvature = []

avgspeed = []
avgaccelerate = []
avgcurvature = []

# cut IMF1 to IMFn from original motion
for i in range (imf.shape[0]-1):
    tmpdata = np.sum(imf[i:],axis = 0)   
    sd.append(np.std(tmpdata,axis =1))
    mse.append(np.mean((data.T - tmpdata) ** 2,axis = 1))
    
    maxspeed.append(np.max(np.abs(np.diff(tmpdata))))
    maxaccelerate.append(np.max(np.abs(np.diff(np.diff(tmpdata)))))
    
    # curvature_circle_fitting = []
    # for j in range(tmpdata.shape[0]):
    #     curvature_circle_fitting.append(calc_curvature_circle_fitting(t, tmpdata[j]))  

    curvature_range_kutta = []
    for j in range(tmpdata.shape[0]):
        curvature_range_kutta.append(calc_curvature_range_kutta(t, tmpdata[j]))  
        
 
    maxcurvature.append(np.max(np.abs(curvature_range_kutta)))
    avgcurvature.append(np.mean(np.abs(curvature_range_kutta)))
    
    avgspeed.append(np.mean(np.abs(np.diff(tmpdata))))
    avgaccelerate.append(np.mean(np.abs(np.diff(np.diff(tmpdata)))))



sd.append(np.std(imf[-1], axis =1))
sd = np.array(sd)

mse.append(np.mean((data.T - imf[-1]) ** 2,axis = 1))
mse = np.array(mse)


maxspeed.append(np.max(np.abs(np.diff(imf[-1]))))
maxspeed = np.array(maxspeed)
maxaccelerate.append(np.max(np.abs(np.diff(np.diff(imf[-1])))))
maxaccelerate = np.array(maxaccelerate)


avgspeed.append(np.mean(np.abs(np.diff(imf[-1]))))
avgspeed = np.array(avgspeed)
avgaccelerate.append(np.mean(np.abs(np.diff(np.diff(imf[-1])))))
avgaccelerate = np.array(maxaccelerate)


curvature_range_kutta = []
for j in range(tmpdata.shape[0]):
    curvature_range_kutta.append(calc_curvature_range_kutta(t, imf[-1][j]))  
    
 
maxcurvature.append(np.max(np.abs(curvature_range_kutta)))
avgcurvature.append(np.mean(np.abs(curvature_range_kutta)))
    
meansd = np.mean(sd,axis = 1)
meanmse = np.mean(mse,axis = 1)

sumsd = np.sum(sd,axis = 1)
summse = np.sum(mse,axis = 1)


########################################################################################################
#compared to FT

np.random.seed(0) #
avgfreqFT = np.zeros([avgfreq.shape[0] + 1, avgfreq.shape[1]]) 

avgfreqFT[1:] = avgfreq # cut same frequency components as HHT

ftdata = np.zeros([nIMFs, sd.shape[1],imf.shape[2]])
#FT
for i in range(nIMFs):
    for j in range(sd.shape[1]):
        
        fc = avgfreqFT[i,j] # 
        
        N = data.shape[0] # 
        f = data[:,j] # 

        meanData = np.mean(f)
        f = f - meanData

        F = fft(f)
        
        F_abs = np.abs(F)
       
        F_abs_amp = F_abs / N * 2 # 
        F_abs_amp[0] = F_abs_amp[0] / 2 #
        
        fq = np.linspace(0, 1.0/dt, N) 
        
        F_ifft = ifft(F) # IFFT
        F_ifft_real = F_ifft.real 
        
        F2 = np.copy(F) 

        if fc != 0: 
            F2[(fq >= fc)] = 0 
        

        F2_abs = np.abs(F2)

        F2_abs_amp = F2_abs / N * 2 
        F2_abs_amp[0] = F2_abs_amp[0] / 2 
        

        F2_ifft = np.fft.ifft(F2) # IFFT
        
        
        F2_ifft_real = F2_ifft.real * 2 
        
        tmpft = F2_ifft_real + meanData
    
        ftdata[i,j] = tmpft
            
sdFT = []
mseFT = []
maxspeedFT = []
maxaccelerateFT = []
maxcurvatureFT = []

avgspeedFT = []
avgaccelerateFT = []
avgcurvatureFT = []

for i in range(ftdata.shape[0]):
    
    tmpdata = ftdata[i]
    sdFT.append(np.std(ftdata[i], axis =1))
    mseFT.append(np.mean((data.T - ftdata[i]) ** 2,axis = 1))
    
    maxspeedFT.append(np.max(np.abs(np.diff(ftdata[i]))))
    maxaccelerateFT.append(np.max(np.abs(np.diff(np.diff(ftdata[i])))))

    curvature_range_kutta = []
    for j in range(tmpdata.shape[0]):
        # curvature_range_kutta.append(calc_curvature_range_kutta(t, tmpdata[j]))  
        curvature_range_kutta.append(calc_curvature_2_derivative(t, tmpdata[j]))  
        

    maxcurvatureFT.append(np.max(np.abs(curvature_range_kutta)))
    avgcurvatureFT.append(np.mean(np.abs(curvature_range_kutta)))
    
    avgspeedFT.append(np.mean(np.abs(np.diff(ftdata[i]))))
    avgaccelerateFT.append(np.mean(np.abs(np.diff(np.diff(ftdata[i])))))
    
mseFT = np.array(mseFT)
sdFT = np.array(sdFT)
maxspeedFT = np.array(maxspeedFT)
maxaccelerateFT = np.array(maxaccelerateFT)
    
avgspeedFT = np.array(avgspeedFT)
avgaccelerateFT = np.array(avgaccelerateFT)

maxcurvatureFT = np.array(maxcurvatureFT)
avgcurvatureFT = np.array(avgcurvatureFT)

meansdFT = np.mean(sdFT,axis = 1)
meanmseFT = np.mean(mseFT,axis = 1)

sumsdFT = np.sum(sdFT,axis = 1)
summseFT = np.sum(mseFT,axis = 1)

########################################################################################################


#plot
x = list(range(0, 7))
# values = ["IMF$_{1}$","IMF1-2","IMF1-3","IMF1-4","IMF1-5","IMF1-6","IMF1-7","IMF1-8"]
# valuesEMD = ["IMF$_{1}$","IMF$_{1-2}$","IMF$_{1-3}$","IMF$_{1-4}$","IMF$_{1-5}$","IMF$_{1-6}$","IMF$_{1-7}$"]

values = []
for i in range(1,len(allavgfreq)):
    values.append(str(round(allavgfreq[i],2)))


plt.rcParams["font.size"] = 40
plt.rc('legend', fontsize=40)
plt.rcParams['axes.linewidth'] = 3 

plt.figure(dpi=200, figsize=(15,15))
plt.subplot(2, 1, 1)

plt.plot(maxspeed[1:-1] / dt, marker = "o", markersize=15, linestyle = "-", linewidth=5, label='MEMD')
plt.plot(maxspeedFT[1:-1] / dt, marker = "^", markersize=15, linestyle = "-", linewidth=5, label='FT')

plt.xticks(x,values)
# plt.xticks(rotation=90)
#plt.title("Average SD of reconstructed motion")
plt.title("Maximum motor speed")
plt.xlabel("Low-pass filter [Hz]")
plt.ylabel("Angle velocity [deg/sec]")
plt.legend()
#plt.show()

plt.subplot(2, 1, 2)
plt.plot(meanmse[1:-1], marker = "o", markersize=15,linestyle = "-", linewidth=5, label='MEMD')
plt.plot(meanmseFT[1:-1], marker = "^", markersize=15,linestyle = "-", linewidth=5, label='FT')
plt.xticks(x,values)
# plt.xticks(rotation=90)
plt.xlabel("Low-pass filter [Hz]")
plt.ylabel("Angle [deg]")
plt.title("Average MSE ")
plt.legend()
plt.tight_layout()
plt.show()


# ########################################################################################################
# #An examplem: remove whole IMF directly output to NAO 

# nimf = 5 
# out = np.deg2rad(np.sum(imf[nimf:],axis = 0).T)

# listout = []
# listout.append(l[0])

# for i in range(out.shape[0]):
#     listout.append([t[i] + 3] + list(out[i]))

# with open('out_hht.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(listout)
    
    
# ########################################################################################################  
#evaluation for each motor

#calculate average frequencies and amplitudes
avgfreq = avgF(imf,dt)
avgamp = avgA(imf,dt)
# freqSD = np.std(avgfreq,axis = 1)
# ampSD = np.std(avgamp,axis =1)
freqSD = np.std(freq,axis = 0)
ampSD = np.std(amp[1:],axis =0)

allavgfreq = np.mean(avgF(imf,dt), axis=1)
allavgamp = np.mean(avgA(imf,dt), axis=1)

sd = []
mse = []
maxspeed = []
maxaccelerate = []

avgcurvature = []

# cut IMF1 to IMFn from original motion
for i in range (imf.shape[0]-1):
    tmpdata = np.sum(imf[i:],axis = 0)   
    sd.append(np.std(tmpdata,axis =1))
    mse.append(np.mean((data.T - tmpdata) ** 2,axis = 1))
    maxspeed.append(np.max(np.abs(np.diff(tmpdata)), axis = 1))
    maxaccelerate.append(np.max(np.abs(np.diff(np.diff(tmpdata))), axis = 1))
    
    curvature_range_kutta = []
    for j in range(tmpdata.shape[0]):
        # curvature_range_kutta.append(calc_curvature_range_kutta(t, tmpdata[j]))  
        curvature_range_kutta.append(calc_curvature_2_derivative(t, tmpdata[j]))  
    
    avgcurvature.append(np.mean(np.abs(curvature_range_kutta), axis = 1))
    
sd.append(np.std(imf[-1], axis =1))
sd = np.array(sd)

mse.append(np.mean((data.T - imf[-1]) ** 2,axis = 1))
mse = np.array(mse)


maxspeed.append(np.max(np.abs(np.diff(imf[-1])), axis = 1))
maxspeed = np.array(maxspeed)

maxaccelerate.append(np.max(np.abs(np.diff(np.diff(imf[-1]))), axis = 1))
maxaccelerate = np.array(maxaccelerate)

avgcurvature = np.array(avgcurvature)
########################################################################################################
#compared to FT

np.random.seed(0) #
avgfreqFT = np.zeros([avgfreq.shape[0] + 1, avgfreq.shape[1]]) 

avgfreqFT[1:] = avgfreq # cut same frequency components as HHT

ftdata = np.zeros([nIMFs, sd.shape[1],imf.shape[2]])
#FT
for i in range(nIMFs):
    for j in range(sd.shape[1]):
        
        fc = avgfreqFT[i,j] # 
        
        N = data.shape[0] # 
        f = data[:,j] # 

        meanData = np.mean(f)
        f = f - meanData

        F = fft(f)
        
        F_abs = np.abs(F)
       
        F_abs_amp = F_abs / N * 2 # 
        F_abs_amp[0] = F_abs_amp[0] / 2 #
        
        fq = np.linspace(0, 1.0/dt, N) 
        
        F_ifft = ifft(F) # IFFT
        F_ifft_real = F_ifft.real 
        
        F2 = np.copy(F) 

        if fc != 0: 
            F2[(fq >= fc)] = 0 
        

        F2_abs = np.abs(F2)

        F2_abs_amp = F2_abs / N * 2 
        F2_abs_amp[0] = F2_abs_amp[0] / 2 
        

        F2_ifft = np.fft.ifft(F2) # IFFT
        
        
        F2_ifft_real = F2_ifft.real * 2 
        
        tmpft = F2_ifft_real + meanData
    
        ftdata[i,j] = tmpft
            
sdFT = []
mseFT = []
maxspeedFT = []
maxaccelerateFT = []
avgcurvatureFT = []
for i in range(ftdata.shape[0]):
    tmpdata = ftdata[i]
    sdFT.append(np.std(ftdata[i], axis =1))
    mseFT.append(np.mean((data.T - ftdata[i]) ** 2,axis = 1))
    
    maxspeedFT.append(np.max(np.abs(np.diff(ftdata[i])), axis = 1))
    maxaccelerateFT.append(np.max(np.abs(np.diff(np.diff(ftdata[i]))), axis = 1))
    
    curvature_range_kutta = []
    for j in range(tmpdata.shape[0]):
        # curvature_range_kutta.append(calc_curvature_range_kutta(t, tmpdata[j]))  
        curvature_range_kutta.append(calc_curvature_2_derivative(t, tmpdata[j]))  
        

    avgcurvatureFT.append(np.mean(np.abs(curvature_range_kutta), axis = 1))

mseFT = np.array(mseFT)
sdFT = np.array(sdFT)
maxspeedFT = np.array(maxspeedFT)
maxaccelerateFT = np.array(maxaccelerateFT)

avgcurvatureFT = np.array(avgcurvatureFT)

# ########################################################################################################


linw = 3
msize = 15
#acceleration
plt.rcParams["font.size"] = 22
plt.rc('legend', fontsize=22)
plt.rcParams['axes.linewidth'] = 3
plt.figure(dpi=200, figsize=(25,10))
    

count = 1
for j in range(maxspeed.shape[1]):
    motor = j
    
    #plot

    if motors[j][1:] in immotors:
    
        plt.subplot(3,4,count)
        # plt.plot(maxaccelerate[1:-1,motor] / dt, marker = "o", markersize=msize, linestyle = "-", linewidth=linw, label='MEMD')
        # plt.plot(maxaccelerateFT[1:-1,motor] / dt, marker = "^", markersize=msize, linestyle = "-", linewidth=linw, label='FT')
        plt.plot(avgcurvature[1:-1,motor] , marker = "o", markersize=msize, linestyle = "-", linewidth=linw, label='MEMD')
        plt.plot(avgcurvatureFT[1:-1,motor] , marker = "^", markersize=msize, linestyle = "-", linewidth=linw, label='FT')
        plt.xticks(x,values)
        # plt.xticks(rotation=90)
        #plt.title("Average SD of reconstructed motion")
        plt.title(motors[j])
        plt.xlabel("Low-pass filter [Hz]")
        #plt.ylabel("Maximum motor acceleration [deg/sec^2]")
        # plt.ylabel("Acceleration [$deg/s^2$]")
        plt.ylabel("Curvature [$rad/m$]")
        plt.legend()
        #plt.show()
        count += 1

plt.subplot(3,4,count)
# plt.plot(maxaccelerate[1:-1,motor] / dt, marker = "o", markersize=msize, linestyle = "-", linewidth=linw, label='MEMD')
# plt.plot(maxaccelerateFT[1:-1,motor] / dt, marker = "^", markersize=msize, linestyle = "-", linewidth=linw, label='FT')
plt.plot(np.mean(avgcurvature[1:-1], axis=1), marker = "o", markersize=msize, linestyle = "-", linewidth=linw, label='MEMD')
plt.plot(np.mean(avgcurvatureFT[1:-1], axis=1), marker = "^", markersize=msize, linestyle = "-", linewidth=linw, label='FT')
plt.xticks(x,values)
# plt.xticks(rotation=90)
#plt.title("Average SD of reconstructed motion")
plt.title("Average all motors")
plt.xlabel("Low-pass filter [Hz]")
#plt.ylabel("Maximum motor acceleration [deg/sec^2]")
# plt.ylabel("Acceleration [$deg/s^2$]")
plt.ylabel("Curvature [$rad/m$]")
plt.legend()
        
plt.tight_layout()
plt.savefig("evaljointC.pdf", bbox_inches='tight')
plt.show()


linw = 3
msize = 15
#acceleration
plt.rcParams["font.size"] = 22
plt.rc('legend', fontsize=22)
plt.rcParams['axes.linewidth'] = 3
plt.figure(dpi=200, figsize=(25,10))
    

count = 1
#acceleration
for j in range(maxspeed.shape[1]):
    motor = j
    
    #plot
    if motors[j][1:] in immotors:
        
        
        plt.subplot(3,4,count)
        plt.plot(mse[1:-1,motor], marker = "o", markersize=msize,linestyle = "-", linewidth=linw, label='MEMD')
        plt.plot(mseFT[1:-1,motor], marker = "^", markersize=msize,linestyle = "-", linewidth=linw, label='FT')
        plt.xticks(x,values)
        # plt.xticks(rotation=90)
        plt.xlabel("Low-pass filter [Hz]")
        plt.ylabel("MSE [$deg$]")
        plt.title(motors[j])
        plt.legend()
        count += 1
        
 
                
plt.subplot(3,4,count)
plt.plot(np.mean(mse[1:-1], axis=1), marker = "o", markersize=msize,linestyle = "-", linewidth=linw, label='MEMD')
plt.plot(np.mean(mseFT[1:-1], axis=1), marker = "^", markersize=msize,linestyle = "-", linewidth=linw, label='FT')
plt.xticks(x,values)
# plt.xticks(rotation=90)
plt.xlabel("Low-pass filter [Hz]")
plt.ylabel("MSE [$deg$]")
plt.title("Average all motors")
plt.legend()
        
plt.tight_layout()
plt.savefig("evaljointM.pdf", bbox_inches='tight')
plt.show()

###################################
#robotic optimization



phi = maxspeedFT[:-1]
sigma = mse[:-1]


beta = 1 - alpha

imfmotorlist = []

for i in range(imf.shape[1]):

    testp = min_max(phi[:,i])
    tests = min_max(sigma[:,i])
    
    indca = alpha * testp + beta * tests
    
    imfmotorlist.append(np.argmin(indca) + 1)


########################################################################################################
#An examplem: implimitation




#Orignal


out = np.deg2rad(data)

listout = []
listout.append(l[0])

for i in range(out.shape[0]):
    listout.append([t[i] * slow + 3] + list(out[i]))

with open(output_opt + '/out_org.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(listout)

#HHT
out = np.zeros([imf.shape[2],imf.shape[1]])
for i, item in enumerate(imfmotorlist):
    out[:,i] = np.sum(imf[item:,i],axis = 0).T

out = np.deg2rad(out)


listout = []
listout.append(l[0])

for i in range(out.shape[0]):
    listout.append([t[i] * slow + 3] + list(out[i]))

with open(output_opt + '/out_hhtAgr.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(listout)
    


#FT

out = np.zeros([ftdata.shape[2],ftdata.shape[1]])
for i, item in enumerate(imfmotorlist):
    out[:,i] = ftdata[item - 1,i].T

out = np.deg2rad(out)

listout = []
listout.append(l[0])

for i in range(out.shape[0]):
    listout.append([t[i] * slow + 3] + list(out[i]))

with open(output_opt + '/out_hhtFT.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(listout)
    
