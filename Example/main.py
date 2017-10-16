import Cobra


MySearch = Cobra.Search()

MySearch.addDatFile('NoBinary')

MySearch.addCandidate('Cand1.dat')
MySearch.ChainRoot = './results/Cand1-'

MySearch.sample(doplot = True, resume=True, nlive = 200)



