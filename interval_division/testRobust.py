import itertools
from demand_generator import get_demand
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from profit import profit, calc_beta_dist_model_profit
from sklearn.preprocessing import MinMaxScaler

p = 10
# c = 20 determined by beta
s = 1
mu = 1000
sd = 1000 / 3
mu1s = np.arange(700, 900, 100).tolist()
n = 1000

nana = list()
dfdf = list()
idid = list()
nana_p = list()
dfdf_p = list()
idid_p = list()

prof_na = list()
prof_df = list()
prof_id = list()

############
betas = [ 0.2, 0.5, 0.8]  #
models  = ['normal_ass', 'dist_free', 'interval_div']
model = 'interval_div'
############
for c in np.arange(2):
    demand_type = 'mm'
    #dist = get_demand(demand_type, mu, sd, n)
    dist = get_demand('mm', mu, sd, n, mu1=300, sd1=300 / 2, w1=.4)
    # dist = get_demand('leather', 7920, 8405, 1)
    pfs = [[m, calc_beta_dist_model_profit(betas, dist, m, p, s)] for m in models]
    NA = list()
    DF = list()
    ID = list()
    MM = list()
    NA_n = list()
    DF_n = list()
    ID_n = list()
    for i in range(0, len(betas)):
       NA = (pfs[0][1][i])
       DF = (pfs[1][1][i])
       ID = (pfs[2][1][i])

       MM = (NA + DF + ID) / 3
       if MM == 0:
           MM = 1
       nana.append(NA)
       dfdf.append(DF)
       idid.append(ID)

       NA_n.append(NA / MM)
       DF_n.append(DF / MM)
       ID_n.append(ID / MM)

       nana_p.append(NA/MM)
       dfdf_p.append(DF/MM)
       idid_p.append(ID/MM)

    # print(np.mean(nana), np.mean(dfdf), np.mean(idid))
    ## dataframe 형태로 csv 파일 저장 분산도 구해야
    pd_nana_p = pd.DataFrame(nana_p)
    pd_dfdf_p = pd.DataFrame(dfdf_p)
    pd_idid_p = pd.DataFrame(idid_p)

    pd_nana = pd.DataFrame(nana)
    pd_dfdf = pd.DataFrame(dfdf)
    pd_idid = pd.DataFrame(idid)

    print('NA', round(np.mean(nana_p), 5), round(np.std(nana_p), 5), '   DF', round(np.mean(dfdf_p), 5), round(np.std(dfdf_p), 5), '   ID',round((np.mean(idid_p)), 5), round((np.std(idid_p)), 5))
    print('NA', round(np.mean(nana), 5),round(np.std(nana), 5), '   DF', round(np.mean(dfdf), 5),round(np.std(dfdf), 5), '   ID',round((np.mean(idid)), 5), round((np.mean(idid)), 5))

    Filename = demand_type + '_mu_' + str(mu) + '_sd_' + str(int(sd))
    pd_nana_p.to_csv(f"res/{Filename}_nana_p.csv", header = True, index = True)
    pd_dfdf_p.to_csv(f"res/{Filename}_dfdf_p.csv", header = True, index = True)
    pd_idid_p.to_csv(f"res/{Filename}_idid_p.csv", header = True, index = True)

    pd_nana.to_csv(f"res/{Filename}_nana.csv", header = True, index = True)
    pd_dfdf.to_csv(f"res/{Filename}_dfdf.csv", header = True, index = True)
    pd_idid.to_csv(f"res/{Filename}_idid.csv", header = True, index = True)

    fig, axes = plt.subplots(1,2, figsize =(12,6))
    axes[0].hist(dist)

    axes[1].plot(betas, NA_n, label = "Normal assumption model")
    axes[1].plot(betas, DF_n, label = "Distribution free model")
    axes[1].plot(betas, ID_n, label = "Interval divide model")

    axes[1].set_xlabel("Beta")
    axes[1].set_ylabel("ProfitRatio")
    axes[1].set_ylim(1 - 5*np.std(ID_n), 1 + 5*np.std(ID_n)) #
    axes[1].set_title(Filename)
    #fig.title(Filename)
    fig.savefig(f'fig/{Filename}')
    fig.show()
    print("####", nana)
    prof_na.append(nana)
    prof_id.append(idid)
    prof_df.append(dfdf)


print('mean', np.mean(prof_na), np.mean(prof_df), np.mean(prof_id), 'sd', np.std(prof_na), np.std(prof_df), np.std(prof_id))
prof_na = pd.DataFrame(prof_na[1])
prof_df = pd.DataFrame(prof_df[1])
prof_id = pd.DataFrame(prof_id[1])

prof_na.to_csv(f"res/{Filename}_prof_nana.csv", header = True, index = True)
prof_df.to_csv(f"res/{Filename}_prof_idid.csv", header = True, index = True)
prof_id.to_csv(f"res/{Filename}_prof_dfdf.csv", header = True, index = True)

