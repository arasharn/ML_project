import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import pandas as pd
from matplotlib.patches import Ellipse

# @title
def paceka(x, B=1.0,C=0.5,D=0.5,E=1.0):
  mu = D * np.sin( \
                  C * np.arctan( \
                                B *( \
                                 (1-E) * \
                                     x + \
                                      (E) * \
                                     np.arctan(B* x)\
                                     )\
                                 )
                  );
  slip = x
  return mu

def noise_maker(axis, sd):
  #np.random.seed(3)
  n = np.random.normal(0, sd, axis.shape)
  #n = np.random.uniform(low=-sd, high=sd, size=axis.shape)
  noisy_axis = axis + n
  return noisy_axis

def corr_calcs(mu, slip):

  coarsity = .01

  rng = np.max(mu)-np.min(mu)
  sd_mu_data = np.std(mu)

  sd_range = np.arange(0, 0.11, coarsity) #0.171 for uniform; for gaussian 0.091
  #sd_range_by_rng = np.arange(0, 1, coarsity)

  sd_len = len(sd_range)

  mu_normalized = mu/np.max(mu)

  epoch = 500000

  cost = np.zeros((len(sd_range)*epoch, 3))

  k = 0
  for i in np.arange(epoch):

    for j in np.arange(sd_len):

      sd = sd_range[j]

      #sd_rng = sd*rng
      #sd_sd = sd*sd_mu_data

      mu_noisy = noise_maker(mu, sd)
      #mu_noisy_normalized = noise_maker(mu_normalized, sd)
      #mu_noisy_by_rng = noise_maker(mu, sd_rng)
      #mu_noisy_by_sd = noise_maker(mu, sd*sd_mu_data)

      cost[k, 0] = i
      #cost[k, 1] = sd_rng
      #cost[k, 2] = sd_sd

      cost[k, 1] = sd

      cost[k, 2] = np.corrcoef(mu, mu_noisy)[0][1]
      #cost[k, 5] = np.corrcoef(mu_normalized, mu_noisy_normalized)[0][1]

      #cost[k, 3] = np.std(cost[0:k, 2])/np.mean(cost[0:k, 2])
      #cost[k, 6] = np.corrcoef(mu, mu_noisy_by_rng)[0][1]
      #cost[k, 7] = np.corrcoef(mu, mu_noisy_by_sd)[0][1]
      if cost[k,2]<0:
        print(i, sd)
        return cost

      if cost[k, 2] >1:
        print(i, sd, cost[k, 2])

      k = k+1


    if i%10000 == 0:
      print(i)
    #print(np.std(cost[0:k, :], 0)/np.mean(cost[0:k], 0))'''
      #print('')
    '''if (i!=0) and (i%1000 == 0):
      np.save('costs_100gaussian.npy', cost[0:k,:])
      files.download('costs_100gaussian.npy')'''

  return cost

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

Data2 = pd.DataFrame(np.load('costs_normal_trim.npy'))


plt.figure(figsize=(11,5))
plt.subplot(211)
plt.plot(Data2.iloc[:,1], Data2.iloc[:,2], 'o', c = 'gray')
plt.plot(Data2.groupby(1).mean().index, Data2.groupby(1).mean().iloc[:,1], c = 'orange')
plt.plot(Data2.groupby(1).min().index, Data2.groupby(1).min().iloc[:,1], c = 'k')
plt.plot(Data2.groupby(1).max().index, Data2.groupby(1).max().iloc[:,1], c = 'k')
plt.ylabel('$r^2$',fontweight="bold")
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
plt.autoscale(enable=True, axis = 'both', tight = True)
plt.grid()

plt.subplot(212)
plt.plot(Data2.groupby(1).mean().index, Data2.groupby(1).mean().iloc[:,1].diff(), 'maroon')
plt.autoscale(enable=True, axis = 'both', tight = True)
plt.xlabel('$\sigma$',fontweight="bold")
plt.ylabel('$d r^2/ d \sigma$',fontweight="bold")
plt.grid()

plt.subplot(211)
plt.plot(Data2.iloc[:,1], Data2.iloc[:,2], 'o')
plt.plot(Data2.groupby(1).mean().index, Data2.groupby(1).mean().iloc[:,1])
plt.subplot(212)
plt.plot(Data2.groupby(1).mean().index, Data2.groupby(1).mean().iloc[:,1].diff())

