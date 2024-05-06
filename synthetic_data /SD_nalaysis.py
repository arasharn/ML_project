import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

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
  #n = np.random.normal(0, sd, axis.shape)
  n = np.random.uniform(low=-sd, high=sd, size=axis.shape)
  noisy_axis = axis + n
  return noisy_axis

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
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


# Start of the analysis
epoch = 1000000
j = 0
coarsity = .01
sd_range = np.arange(0, 7, coarsity)
cost = np.empty((len(sd_range)*epoch, 2))
slip = np.arange(0,101)/100
mu = paceka(slip, B=1.0,C=0.5,D=0.5,E=1.0)
rng = np.max(mu)-np.min(mu)
#mu = (mu-np.min(mu))/np.max(mu)
for epo in np.arange(epoch):
  # for loop
  for i in np.arange(len(sd_range)):
    mu_noisy = noise_maker(mu, sd_range[i]*rng)
    cost[j, 0] = sd_range[i]*rng
    cost[j, 1] = np.corrcoef(mu, mu_noisy)[0][1]
    j = j+1
  # end of for loop
#end of for loop
out = pd.DataFrame()
out['std'] = cost[:, 0]
out['corr'] = cost[:, 1]
#cost = cost[cost[:,1]>=.5,:]
pareto_front = cost[is_pareto_efficient_simple(cost), :]
pareto_front = pareto_front[pareto_front[:,0].argsort()]


diff_pareto_front = pareto_front[1::, :].copy()
diff_pareto_front[:,1] = np.diff(pareto_front[:,1])
idx_min_slope = np.min(np.where(diff_pareto_front[:,1] == np.min(diff_pareto_front[:,1])))
print(pareto_front[idx_min_slope-1, 0], pareto_front[idx_min_slope-1, 1])



plt.figure()
plt.subplot(211)
plt.plot(pareto_front[0:-1, 0], np.diff(pareto_front[:, 1]), '-s')
plt.subplot(212)
plt.plot(out['std'], out['corr'], 'o')
plt.plot(pareto_front[:, 0], pareto_front[:, 1], '-x')

plt.figure()
plt.subplot(312)
plt.plot(out.groupby('std').mean()['corr'].diff())
plt.subplot(311)
plt.plot(out.groupby('std').mean()['corr'])
plt.subplot(313)
plt.plot(out.groupby('std').mean()['corr'].diff().diff())

plt.figure()
plt.subplot(311)
plt.plot(out.groupby('std').mean()['corr'].diff())
plt.subplot(312)
plt.plot(out.groupby('std').mean()['corr'])
plt.subplot(313)
plt.plot(out.groupby('std').mean()['corr'].diff().diff())


df = out.groupby('std').mean()
df['std'] = df.index
df = df[['std', 'corr']]
df.head()

pf = df.values[is_pareto_efficient_simple(df.values), :]
plt.subplot(311)
plt.plot(df['std'], df['corr'])
plt.plot(pf[:,0], pf[:,1], 'x')
plt.subplot(312)
plt.plot(pf[0:-1,0], np.diff(pf[:,1]))

out50 = out[out['corr']>=.5].copy()
out50.reset_index
plt.figure()
plt.subplot(221)
plt.plot(out50.groupby('std').std()['corr'].diff())
plt.subplot(223)
plt.plot(out50.groupby('std').std()['corr'])
plt.subplot(222)
plt.plot(out50.groupby('std').mean()['corr'].diff())
plt.subplot(224)
plt.plot(out50.groupby('std').mean()['corr'])
