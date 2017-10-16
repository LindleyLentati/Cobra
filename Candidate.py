import numpy as np


class Candidate(object):

	def __init__(self, filename):

		self.filename = filename
		self.period = None
		self.FitBigP = False
		self.FitAcceleration = False

		self.FitCircBinary = False
		self.FitEccBinary = False
		self.FitGRBinary = False
		self.FitPKBinary = False

		self.FitDM = False
		self.FitScatter = False
		self.TestPoint = None


		self.pmin = [0,-3]
		self.pmax = [1, 0]
		self.wrapped = [1,0]
		self.params = []
		self.n_dims = None
		self.n_params = None

		self.setupCandidate()

	def setupCandidate(self):
	

		self.params.append('Phase')
		self.params.append('Log_10 Width')
		self.params.append('Period')

		cfile = open(self.filename).readlines()
		for i in range(len(cfile)):

			if('Phase' in cfile[i]):
				line=cfile[i].strip('\n').split()
				minph = np.float64(line[-2])
				maxph = np.float64(line[-1])
				self.pmin[0] = minph
				self.pmax[0] = maxph

			if('Width' in cfile[i]):
				line=cfile[i].strip('\n').split()
				minw = np.float64(line[-2])
				maxw = np.float64(line[-1])
				self.pmin[1] = minw
				self.pmax[1] = maxw

			if('Period' in cfile[i]):
				line=cfile[i].strip('\n').split()
				self.period = np.float64(line[-2])
				prange = np.float64(line[-1])
				self.pmin.append(self.period*(1.0 - prange))
				self.pmax.append(self.period*(1.0 + prange))
				self.wrapped.append(0)

			if('DM' in cfile[i]):
				self.FitDM = True				
				self.params.append('DM')

				line=cfile[i].strip('\n').split()
				minDM = np.float64(line[-2])
				maxDM = np.float64(line[-1])
				self.pmin.append(minDM)
				self.pmax.append(maxDM)
				self.wrapped.append(0)				

			if('Acceleration' in cfile[i]):
				self.FitAcceleration = True
				self.params.append('Acceleration')

				line=cfile[i].strip('\n').split()
				minacc = np.float64(line[-2])
				maxacc = np.float64(line[-1])
				self.pmin.append(minacc)
				self.pmax.append(maxacc)
				self.wrapped.append(0)

			if('CircBinary' in cfile[i]):

				self.FitCircBinary = True
				self.params.append('Log10 BinaryAmp')	
				self.params.append('BinaryPhase')
				self.params.append('Log10 BinaryPeriod')
				
	
				line=cfile[i].strip('\n').split()
				minp = np.float64(line[-4])
				maxp = np.float64(line[-3])

				minamp = np.float64(line[-2])
				maxamp = np.float64(line[-1])


				self.pmin.append(minamp)
				self.pmin.append(0)
				self.pmin.append(minp)
	
				self.pmax.append(maxamp)	
				self.pmax.append(2*np.pi)
				self.pmax.append(maxp)

				self.wrapped.append(0)
				self.wrapped.append(1)
				self.wrapped.append(0)
			

			if('EccBinary' in cfile[i]):

				self.FitEccBinary = True
				self.params.append('Log10 BinaryAmp')	
				self.params.append('BinaryPhase')
				self.params.append('Log10 BinaryPeriod')
				self.params.append('Omega')
				self.params.append('Log10 Ecc')
	
				line=cfile[i].strip('\n').split()
				minp = np.float64(line[-4])
				maxp = np.float64(line[-3])

				minamp = np.float64(line[-2])
				maxamp = np.float64(line[-1])


				self.pmin.append(minamp)
				self.pmin.append(0)
				self.pmin.append(minp)
				self.pmin.append(0)
				self.pmin.append(np.log10(0.0000001))
	
				self.pmax.append(maxamp)	
				self.pmax.append(2*np.pi)
				self.pmax.append(maxp)
				self.pmax.append(2*np.pi)
				self.pmax.append(np.log10(0.005))

				self.wrapped.append(0)
				self.wrapped.append(1)
				self.wrapped.append(0)
				self.wrapped.append(1)
                                self.wrapped.append(0)

			if('GRBinary' in cfile[i]):

				self.FitGRBinary = True
				self.params.append('Log10 BinaryAmp')	
				self.params.append('BinaryPhase')
				self.params.append('Log10 BinaryPeriod')
				self.params.append('Omega')
				self.params.append('Log10 Ecc')
				self.params.append('Log10 M1')
				self.params.append('Log10 M2')

	
				line=cfile[i].strip('\n').split()
				minp = np.float64(line[-4])
				maxp = np.float64(line[-3])

				minamp = np.float64(line[-2])
				maxamp = np.float64(line[-1])


				self.pmin.append(minamp)		#Binary Amp
				self.pmin.append(0)			#Binary Phase
				self.pmin.append(minp)			#Binary Period
				self.pmin.append(0)			#Omega
				self.pmin.append(np.log10(0.0000001))	#Ecc
				self.pmin.append(-1)			#M1
				self.pmin.append(-1)			#M2
	
				self.pmax.append(maxamp)		#Binary Amp
				self.pmax.append(2*np.pi)		#Binary Phase
				self.pmax.append(maxp)			#Binary Period
				self.pmax.append(2*np.pi)		#Omega
				self.pmax.append(np.log10(0.005))	#Ecc			
				self.pmax.append(2)			#M1
				self.pmax.append(2)			#M2


				self.wrapped.append(0)
				self.wrapped.append(1)
				self.wrapped.append(0)
				self.wrapped.append(1)
                                self.wrapped.append(0)
				self.wrapped.append(0)
				self.wrapped.append(0)

			if('PKBinary' in cfile[i]):

				self.FitPKBinary = True
				self.params.append('Log10 BinaryAmp')	
				self.params.append('BinaryPhase')
				self.params.append('Log10 BinaryPeriod')
				self.params.append('Omega')
				self.params.append('Log10 Ecc')
				self.params.append('OMDot')
				self.params.append('SINI')
				self.params.append('Gamma')
				self.params.append('PBDot')
				self.params.append('M2')

	
				line=cfile[i].strip('\n').split()
				minp = np.float64(line[-4])
				maxp = np.float64(line[-3])

				minamp = np.float64(line[-2])
				maxamp = np.float64(line[-1])


				self.pmin.append(minamp)		#Binary Amp
				self.pmin.append(0)			#Binary Phase
				self.pmin.append(minp)			#Binary Period
				self.pmin.append(0)			#Omega
				self.pmin.append(np.log10(0.0000001))	#Ecc
				self.pmin.append(2400)			#OMDot
				self.pmin.append(0.999)			#SINI
				self.pmin.append(0.002)                  #Gamma
				self.pmin.append(-5e-10)		#PBDot
				self.pmin.append(27)			#M2
	
				self.pmax.append(maxamp)		#Binary Amp
				self.pmax.append(2*np.pi)		#Binary Phase
				self.pmax.append(maxp)			#Binary Period
				self.pmax.append(2*np.pi)		#Omega
				self.pmax.append(np.log10(0.005))	#Ecc			
				self.pmax.append(2600)                  #OMDot
                                self.pmax.append(0.9999)                 #SINI
                                self.pmax.append(0.005)                  #Gamma
                                self.pmax.append(5e-10)                #PBDot
                                self.pmax.append(33)                    #M2	

				self.wrapped.append(0)
				self.wrapped.append(1)
				self.wrapped.append(0)
				self.wrapped.append(1)
                                self.wrapped.append(0)
				self.wrapped.append(0)
				self.wrapped.append(0)
				self.wrapped.append(0)
				self.wrapped.append(0)
				self.wrapped.append(0)

                        if('BPh' in cfile[i]):

                                line=cfile[i].strip('\n').split()
                                minamp = np.float64(line[-2])
                                maxamp = np.float64(line[-1])
                                print 'setting BP prior:', minamp, maxamp, line, cfile[i]
                                self.pmin[4] = minamp
                                self.pmax[4] = maxamp



                        if('OM' in cfile[i]):

                                line=cfile[i].strip('\n').split()
                                minamp = np.float64(line[-2])
                                maxamp = np.float64(line[-1])
                                print 'setting OM prior:', minamp, maxamp, line, cfile[i]
                                self.pmin[6] = minamp
                                self.pmax[6] = maxamp


                        if('ECC' in cfile[i]):

                                line=cfile[i].strip('\n').split()
                                minamp = np.float64(line[-2])
                                maxamp = np.float64(line[-1])
                                print 'setting Ecc prior:', minamp, maxamp, line, cfile[i]
                                self.pmin[7] = minamp
                                self.pmax[7] = maxamp

                        if('M1' in cfile[i]):

                                line=cfile[i].strip('\n').split()
                                minamp = np.float64(line[-2])
                                maxamp = np.float64(line[-1])
                                print 'setting M1 prior:', minamp, maxamp, line, cfile[i]
                                self.pmin[8] = minamp
                                self.pmax[8] = maxamp

                        if('M2' in cfile[i]):

                                line=cfile[i].strip('\n').split()
                                minamp = np.float64(line[-2])
                                maxamp = np.float64(line[-1])
                                print 'setting M2 prior:', minamp, maxamp, line, cfile[i]
                                self.pmin[9] = minamp
                                self.pmax[9] = maxamp

			if('Scattering' in cfile[i]):

				self.FitScatter = True
				self.params.append('Log10 Scatter')

				line=cfile[i].strip('\n').split()
				minsc = np.float64(line[-2])
				maxsc = np.float64(line[-1])
				self.pmin.append(minsc)
				self.pmax.append(maxsc)
				self.wrapped.append(0)

			if('BigP' in cfile[i]):
				self.FitBigP = True
				self.params.append('BigP')

                                line=cfile[i].strip('\n').split()
				minsc = np.float64(line[-2])
                                maxsc = np.float64(line[-1])
                                self.pmin.append(minsc)
                                self.pmax.append(maxsc)

                                self.wrapped.append(0)

		
		
		self.pmin=np.array(self.pmin)
		self.pmax=np.array(self.pmax)
		print "wrapped", self.wrapped
		self.wrapped=np.array(self.wrapped).astype(np.int)
		
		self.n_dims = len(self.params)
		self.n_params = 2*self.n_dims  - 1

		if(self.FitGRBinary == True):
			self.n_params = 2*self.n_dims  - 1 + 6 #includes PK parameters

		self.TestPoint = np.zeros(self.n_params)
		self.TestPoint[:self.n_dims] = 0.5*(self.pmin+self.pmax) 


