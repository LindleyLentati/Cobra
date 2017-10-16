import File
import libstempo as T
import scipy.interpolate as interp
import numpy as np
import os
import pickle 

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
import pycuda.driver as drv
import skcuda.fft as fft
import skcuda.linalg as cula

class DatFile(object):

	def __init__(self, root, subTime, bary=True, powerofTwo = True, FromPickle = False, doFFT = True):

		self.root = root
		self.subTime = subTime
		self.doBary = bary
		self.powerofTwo = powerofTwo
		self.FromPickle = FromPickle
		self.doFFT = doFFT

		self.RefMJD = None
		self.pepoch=None
		self.TSamp = None
		self.NSamps = None
		self.NChan = None
		self.NSubInts = 0
		self.ChanWidth = None
		self.LowChan = None
		self.DM = None
		self.TObs = None
		self.RA = None
		self.Dec = None

		self.Data = None
		self.Noise = None
		self.BaseTime = None
		self.SampleFreqs = None
		self.BCorrs = None

		self.gpu_fft_data = None
		self.Real = None
		self.Imag = None

		self.gpu_time = None
		self.plan = None
		self.gpu_pulsar_signal = None
		self.gpu_pulsar_fft = None
		self.Plan = None

		self.FSamps = None
		self.ZeroLike = None

		self.Tblocks = None
		self.Tblocks = None
		self.block_size = None	

		self.setupDat()

	def getpowerofTwo(self, length):
		minval = 10.0**100
		power = 0
		for i in range(30):
			diff=length - 2.0**i
			if(diff < minval and diff >= 0):
				minval = diff
				power = i
		return power


	def setupDat(self):

		if(self.FromPickle == False):
			self.Data=File.readDat(self.root+'.dat')
		else:
			print "Loading from Pickled Data"
			pick = open(self.root+'.pickle', 'rb')
			self.Data = pickle.load(pick)
			pick.close()

		if(self.powerofTwo == True):
			p2 = self.getpowerofTwo(len(self.Data))
			powerof2cut = (len(self.Data) - 2**p2)/2
			print "Using the first ", 100.0*(len(self.Data)-2*powerof2cut)/(len(self.Data)*1.0), "% of data (", len(self.Data)-2*powerof2cut," of ",len(self.Data), ") to get a power of 2"
			if(powerof2cut > 0):
				self.Data=self.Data[:-2*powerof2cut]
		self.Data=self.Data-np.mean(self.Data)
		self.parseInf()
		if(self.doBary == True):
			print "Barycentering Time Axis"
			self.getBaryCorrs()
			interpBCorrs=interp.griddata(24*60*60*(self.BCorrs[0]-np.min(self.BCorrs[0])), self.BCorrs[1], self.BaseTime)
			self.BaseTime += interpBCorrs

		if(self.subTime > 0):
			TDiff=self.RefMJD-self.subTime

			print "Different Times: ", self.RefMJD, self.subTime, TDiff
			self.BaseTime+=TDiff*24*60*60

		if(self.doFFT == True):
			gpu_Data = gpuarray.to_gpu(np.float64(self.Data))
			self.gpu_fft_data  = gpuarray.zeros(self.NSamps/2+1, np.complex128)

			self.Plan = fft.Plan(self.NSamps, np.float64, np.complex128)
			fft.fft(gpu_Data, self.gpu_fft_data, self.Plan) 
			self.gpu_fft_data = self.gpu_fft_data[1:-1]
			
			gpu_Data.gpudata.free()

#			self.Real = gpuarray.empty(self.NSamps/2-1, np.float64)
#			self.Imag = gpuarray.empty(self.NSamps/2-1, np.float64)
#			self.Real = gpuarray.to_gpu(np.float64(gpu_fftData.real[1:-1].get()))
#			self.Imag = gpuarray.to_gpu(np.float64(gpu_fftData.imag[1:-1].get()))
			self.FSamps=len(self.gpu_fft_data)

			self.CalcNoise(cut=False, mode=1)

			#self.SampleFreqs = gpuarray.empty(self.FSamps, np.float64)
			#self.SampleFreqs = gpuarray.to_gpu(2.0*np.pi*np.float64(np.arange(1,self.FSamps+1))/self.TObs)

			self.gpu_time = gpuarray.to_gpu(np.float64(self.BaseTime))
			self.gpu_pulsar_signal = gpuarray.empty(self.NSamps, np.float64)
			self.gpu_pulsar_fft = gpuarray.empty(self.NSamps/2+1, np.complex128)

			self.block_size = 128
			self.Tblocks = int(np.ceil(self.NSamps*1.0/self.block_size))
			self.Fblocks = int(np.ceil(self.FSamps*1.0/self.block_size))

		self.pepoch = self.BaseTime[len(self.BaseTime)/2]
			
	def RandomisePhase(self):
		ranPhases = np.random.uniform(0,1, self.FSamps)
		CompRan = np.cos(2*np.pi*ranPhases) + 1j*np.sin(2*np.pi*ranPhases)
		 
		
		OComp = self.gpu_fft_data.get()
		NComp = CompRan*OComp

		self.gpu_fft_data = gpuarray.to_gpu(np.complex128(NComp))
		self.Real = gpuarray.to_gpu(np.float64(NComp.real))
                self.Imag = gpuarray.to_gpu(np.float64(NComp.imag))
			

	def parseInf(self):
		inf=open(self.root+".inf").readlines()
		for i in range(len(inf)):
			if('Epoch of observation' in inf[i]):
				line=inf[i].strip('\n').split()
				self.RefMJD = np.float64(line[-1])
			if('Width of each time series bin' in inf[i]):
				line=inf[i].strip('\n').split()
				self.TSamp =  np.float64(line[-1])
			if('Channel bandwidth' in inf[i]):
				line=inf[i].strip('\n').split()
				self.ChanWidth =  np.float64(line[-1])
			if('Number of channels' in inf[i]):
				line=inf[i].strip('\n').split()
				self.NChan =  np.float64(line[-1])
			if('Central freq of low channel' in inf[i]):
				line=inf[i].strip('\n').split()
				self.LowChan =  np.float64(line[-1])
			if('Dispersion measure' in inf[i]):
				line=inf[i].strip('\n').split()
				self.DM =  np.float64(line[-1])
			if('J2000 Right Ascension' in inf[i]):
				line=inf[i].strip('\n').split()
				self.RA =  line[-1]
			if('J2000 Declination' in inf[i]):
				line=inf[i].strip('\n').split()
				self.Dec = line[-1]

		self.NSamps = len(self.Data)
		self.TObs = self.TSamp*self.NSamps
		self.BaseTime=np.linspace(0,(self.NSamps-1)*self.TSamp,self.NSamps)
		



	def getBaryCorrs(self):
		newpar=open('bary.par', 'w')



		newpar.write("PSRJ           J0000+0000\n")
		newpar.write("RAJ            "+str(self.RA )+"      0\n")
		newpar.write("DECJ           "+str(self.Dec)+"        0\n")
		newpar.write("F0             1.0 0 \n")
		newpar.write("PEPOCH         "+str(self.RefMJD)+"\n")
		newpar.write("POSEPOCH       "+str(self.RefMJD)+"\n")
		newpar.write("DMEPOCH        "+str(self.RefMJD)+"\n")
		newpar.write("DM		"+str(self.DM)+" 0\n")
		newpar.write("EPHVER         5\n")
		newpar.write("CLK            UNCORR\n")
		newpar.write("MODE 1\n")
		newpar.write("EPHEM          DE421\n")


		newpar.close()

		newtim=open('bary.tim', 'w')
		tstep = 10.0
		start=self.RefMJD
		stop=self.RefMJD+(self.NSamps*self.TSamp+tstep)/24/60/60
		N=np.floor((stop-start)*24*60*60/tstep).astype(int)
		x=np.linspace(start, stop,  N)

		newtim.write("FORMAT 1\n")
		for i in range(len(x)):
			newtim.write("barytim "+str(self.LowChan)+" "+str(x[i])+" 0.1 pks\n")

		newtim.close()

		psr=T.tempopulsar(parfile='bary.par', timfile='bary.tim')
		psr.fit()
		BCorrs = psr.batCorrs()
		self.BCorrs=np.zeros([2,len(BCorrs)])
		self.BCorrs[0] = (psr.stoas).copy()
		self.BCorrs[1] = BCorrs.copy()*24*60*60
		os.remove('bary.par')
		os.remove('bary.tim')

	def CalcNoise(self, cut=False, mode = 0):

		step = 100
		blocks=np.ceil(1.0*self.FSamps/step).astype(np.int)
		noisevec=np.zeros(self.FSamps)
		if(mode == 0):
			for i in range(blocks):
				start=i*step
				stop=(i+1)*step
				if(stop > self.FSamps):
					print "stop!", i, stop
					stop=self.FSamps

				r2 = cula.dot(self.Real[start:stop], self.Real[start:stop:])
				i2 = cula.dot(self.Imag[start:stop], self.Imag[start:stop])
				noisesamps=len(self.Imag[start:stop])*2
				noise=np.sqrt((r2+i2)/noisesamps)	
				noisevec[start:stop] = 1.0/noise
				self.Real[start:stop] /= noise
				self.Imag[start:stop] /= noise
				self.gpu_fft_data[start:stop] /= noise
				#print "noise", i, noise

			if(cut == True):
				fftdataR = (self.gpu_fft_data.get()).real
				fftdataI = (self.gpu_fft_data.get()).imag
				
				Rbad = np.where(np.abs(fftdataR) > 6.0)[0]
				Ibad = np.where(np.abs(fftdataI) > 6.0)[0]
				print fftdataR[Rbad], fftdataI[Ibad]
				NRbad=len(Rbad)
				NIbad=len(Ibad)
				fftdata = self.gpu_fft_data.get()
				fftdata.real[Rbad]=np.random.normal(0,1, NRbad)
				fftdata.imag[Ibad]=np.random.normal(0,1, NIbad)
				print "bad", NRbad, NIbad
				self.gpu_fft_data = gpuarray.to_gpu(np.complex128(fftdata))
		if(mode == 1):
			#r2 = cula.dot(self.Real[self.FSamps/2:], self.Real[self.FSamps/2:])
                        #i2   = cula.dot(self.Imag[self.FSamps/2:], self.Imag[self.FSamps/2:])
			fftD = self.gpu_fft_data.get()
		        r2 = np.dot(fftD.real[self.FSamps/2:], fftD.real[self.FSamps/2:])	
			i2 = np.dot(fftD.imag[self.FSamps/2:], fftD.imag[self.FSamps/2:])
			noisesamps=len(fftD.imag[self.FSamps/2:])*2


			del fftD

                        noise=np.sqrt((r2+i2)/noisesamps)
			print "Noise mode 1: ", noise
                        #self.Real = self.Real / noise
                        #self.Imag = self.Imag / noise
                        self.gpu_fft_data = self.gpu_fft_data / noise
			noisevec[:] = 1.0/noise
		
		#self.Noise = gpuarray.empty(self.FSamps, np.float64)
		self.Noise = gpuarray.to_gpu(np.float64(noisevec))


	def WriteSignalToDat(self, outfile, noise = 0):
		sig=self.gpu_pulsar_signal.get()
		if(noise > 0):
			noise=np.random.normal(0,1,len(sig))*noise
			sig += noise
		sig=np.float32(sig)
		sig.tofile(outfile)

		
