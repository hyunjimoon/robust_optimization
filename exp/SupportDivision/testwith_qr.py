import itertools
from demand_generator import gen_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from profit import profit, calc_beta_dist_model_profit
from sklearn.preprocessing import MinMaxScaler

p = 10
# c = 20 determined by beta
s = 1
mu = 1000
sd = 1000 / 2
mu1s = np.arange(700, 900, 100).tolist()
n = 1000

nana = list()
dfdf = list()
idid = list()
qq = list()
nana_p = list()
dfdf_p = list()
idid_p = list()
qq_p = list()
############
betas = [0.2, 0.5, 0.8]  #
models = ['normal_ass', 'dist_free', 'interval_div' 'quantile_q']
model = 'interval_div'
############

print("###########")
demand_type = 'unif'
dist = gen_data(demand_type, mu, sd, n)
# dist = gen_data('mm', mu, sd, n, mu1=300, sd1=300 / 4, w1=.4)
# dist = gen_data('leather', 7920, 8405, 1)
pfs = [[m, calc_beta_dist_model_profit(betas, dist, m, p, s)] for m in models]

print(pfs)
NA = list()
DF = list()
ID = list()
Q = list()

MM = list()
NA_n = list()
DF_n = list()
ID_n = list()
Q_n = list()
for i in range(0, len(betas)):
    NA = (pfs[0][1][i])
    DF = (pfs[1][1][i])
    ID = (pfs[2][1][i])
    Q = (pfs[3][1][i])

    MM = (NA + DF + ID + Q) / 4
    if MM == 0:
        MM = 1
    nana.append(NA)
    dfdf.append(DF)
    idid.append(ID)
    qq.append(Q)

    NA_n.append(NA / MM)
    DF_n.append(DF / MM)
    ID_n.append(ID / MM)
    Q_n.append(Q /MM)

    nana_p.append(NA / MM)
    dfdf_p.append(DF / MM)
    idid_p.append(ID / MM)
    qq_p.append(Q / MM)
# print(np.mean(nana), np.mean(dfdf), np.mean(idid))
## dataframe 형태로 csv 파일 저장 분산도 구해야
pd_nana_p = pd.DataFrame(nana_p)
pd_dfdf_p = pd.DataFrame(dfdf_p)
pd_idid_p = pd.DataFrame(idid_p)
pd_qq_p = pd.DataFrame(qq_p)

pd_nana = pd.DataFrame(nana)
pd_dfdf = pd.DataFrame(dfdf)
pd_idid = pd.DataFrame(idid)
pd_qq = pd.DataFrame(qq)
print('NA', round(np.mean(nana_p), 3), '   DF', round(np.mean(dfdf_p), 3), '   ID', round((np.mean(idid_p)), 5), '   Q', round(np.mean(qq_p), 3))
print('NA', round(np.mean(nana), 5), round(np.std(nana), 5), '   DF', round(np.mean(dfdf), 5), round(np.std(dfdf), 5),
      '   ID', round((np.mean(idid)), 5), round((np.std(idid)), 5), '   Q', round((np.mean(idid)), 5), round((np.std(idid)), 5))

pd_nana_p.to_csv(f"res/{demand_type}_nana_p.csv", header=True, index=True)
pd_dfdf_p.to_csv(f"res/{demand_type}_dfdf_p.csv", header=True, index=True)
pd_idid_p.to_csv(f"res/{demand_type}_idid_p.csv", header=True, index=True)

pd_nana.to_csv(f"res/{demand_type}_nana.csv", header=True, index=True)
pd_dfdf.to_csv(f"res/{demand_type}_dfdf.csv", header=True, index=True)
pd_idid.to_csv(f"res/{demand_type}_idid.csv", header=True, index=True)

plt.plot(betas, NA_n, label="Normal assumption model")
plt.plot(betas, DF_n, label="Distribution free model")
plt.plot(betas, ID_n, label="Interval divide model")
plt.plot(betas, Q_n, label="Interval divide model")

plt.xlabel("Beta")
plt.ylabel("Profit")
plt.xlim(0, 1)

Filename = demand_type + 'mu' + str(mu) + 'sd' + str(int(sd))
plt.title(Filename)
plt.savefig(f'fig/{Filename}')
plt.show()