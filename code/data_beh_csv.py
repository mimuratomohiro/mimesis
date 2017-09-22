# coding: UTF-8

import glob
import numpy as np
import pyhsmm
import pyhsmm.basic.distributions as distributions
from pyhsmm.util.text import progprint_xrange
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MAX_VALUE = 1000000000

def Bhattacharyya(mu1,mu2,sigma1,sigma2):
	sigma     = (sigma1 + sigma2)/2.
	mu        = (mu1 - mu2)
	distance  = mu.dot(np.linalg.inv(sigma)).dot(mu.T)/8. + np.log(np.linalg.det(sigma) / np.sqrt(np.linalg.det(sigma1) * np.linalg.det(sigma2)))/2.
	return distance

if __name__ == '__main__':
	names       = glob.glob('./../mdb/*')
	data_all    = []
	mu_datas    = []
	sigma_datas = []
	hist_datas  = []
	label_name  = []
	num_syurui  = 0

	for name in names:
		print(name)
		num_syurui    += 1
		failes         = glob.glob(name+"/*")
		data_motion    = []
		mu_motion      = []
		sigma_motion   = []
		obs_dim        = 20
		Nmax           = 20

		obs_hypparams  = {'mu_0':np.zeros(obs_dim),
			'sigma_0':np.eye(obs_dim),
			'kappa_0':0.3,
			'nu_0':obs_dim+5}

		dur_hypparams  = {'alpha_0':20*30,
			'beta_0':20}

		obs_distns     = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
		dur_distns     = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

		posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
				alpha=10.,gamma=10., # better to sample over these; see concentration-resampling.py
				init_state_concentration=10., # pretty inconsequential
				obs_distns=obs_distns,
				dur_distns=dur_distns)

		for faile in failes:
			f        = open(faile)
			line     = f.readlines()
			data_num = []
			f.close()
			for i,j in zip(line,range(len(line))):
				if j > 4:
					num  = i[:-1].split(",")
					data = np.array(list(map(float, num)))
					data_num.append(data)
			print(np.array(data_num).shape)
			
			posteriormodel.add_data(np.array(data_num)[:,1:],trunc=60)
			data_motion.append(np.array(data_num))

		for idx in progprint_xrange(150):
			posteriormodel.resample_model()

		hist = np.zeros(Nmax)
		for i in range(len(posteriormodel.states_list)):
			hist += np.histogram(posteriormodel.states_list[i].stateseq,range(Nmax+1))[0]
		for i in range(Nmax):
			mu_motion.append(posteriormodel.obs_distns[i].mu)
			sigma_motion.append(posteriormodel.obs_distns[i].sigma)

		hist = np.array(hist,bool)
		data_all.append(data_motion)
		mu_datas.append(np.array(mu_motion))
		sigma_datas.append(np.array(sigma_motion))
		hist_datas.append(hist)
		label_name.append(name[9:])

	data_all    = np.array(data_all)
	mu_datas    = np.array(mu_datas)
	sigma_datas = np.array(sigma_datas)
	hist_datas  = np.array(hist_datas)

	distance    = np.zeros((num_syurui,num_syurui))

	for i,hist_1 in zip(range(num_syurui),hist_datas):
		for j,hist_2 in zip(range(num_syurui),hist_datas):
			if i == j:
				distance[i][j] = 0.
			else:
				distance_bata  = 0.
				for class_1,ct1 in zip(hist_1,range(hist.shape[0])):
					if class_1 == 1:
						distance_bata_sub  = MAX_VALUE
						for class_2,ct2 in zip(hist_2,range(hist.shape[0])):
							if class_2 == 1:
								dis = Bhattacharyya(mu_datas[i,ct1],mu_datas[j,ct2],sigma_datas[i,ct1],sigma_datas[j,ct2])
								if distance_bata_sub  > dis:
									distance_bata_sub = dis
						distance_bata += distance_bata_sub
				distance[i][j] = distance_bata

	distance     = (distance.T + distance)/2.
	n_components = 3
	mds          = manifold.MDS(n_components, max_iter=100, n_init=1)
	Y            = mds.fit_transform(distance)

	colors       = ["red","blue","green","cyan","magenta","yellow","black"]
	fig          = plt.figure()
	ax           = Axes3D(fig)
	for i in range(num_syurui):
		ax.plot([Y[i,0]],[Y[i,1]],[Y[i,2]], "o",color=colors[i], ms=7, mew=0.5, label = label_name[i])
	plt.legend()
	plt.show()
'''
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot([Y[0,0]],[Y[0,1]],[Y[0,2]], "o",color="red"    , ms=7, mew=0.5, label = label_name[0])
	ax.plot([Y[1,0]],[Y[1,1]],[Y[1,2]], "o",color="blue"   , ms=7, mew=0.5, label = label_name[1])
	ax.plot([Y[2,0]],[Y[2,1]],[Y[2,2]], "o",color="green"  , ms=7, mew=0.5, label = label_name[2])
	ax.plot([Y[3,0]],[Y[3,1]],[Y[3,2]], "o",color="cyan"   , ms=7, mew=0.5, label = label_name[3])
	ax.plot([Y[4,0]],[Y[4,1]],[Y[4,2]], "o",color="magenta", ms=7, mew=0.5, label = label_name[4])
	ax.plot([Y[5,0]],[Y[5,1]],[Y[5,2]], "o",color="yellow" , ms=7, mew=0.5, label = label_name[5])
	ax.plot([Y[6,0]],[Y[6,1]],[Y[6,2]], "o",color="black"  , ms=7, mew=0.5, label = label_name[6])
	plt.legend()
	plt.show()
'''
