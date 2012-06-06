# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
from pandas.io.data import DataReader as dr
import datetime
import hp_filter as hp
import Bandpass_Filter as bpf
from prettytable import PrettyTable as pt

# <codecell>

gdp = np.asarray(
    dr('GDPC1', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
cpi = np.asarray(
    dr('CPIAUCSL', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
cons = np.asarray(
    dr('PCECC96', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])
inv = np.asarray(
    dr('GCEC96', 'fred', start = datetime.datetime(1947,1,1))['VALUE'])

mask = np.arange(1,cpi.size, 3)
cpi = cpi[mask]

# <codecell>


# <codecell>

# Generate GDP filtered datas
first_diff_gdp = gdp[1:] - gdp[:-1]
hp_gdp = hp.hp_filter(gdp)[1]
bp_gdp = bpf.bandpass_filter(gdp, 16, 6, 32)[16:-16]
all_gdp = np.array([first_diff_gdp, hp_gdp, bp_gdp])

gdp_table = np.zeros((4,all_gdp.shape[0]))

for i in range(all_gdp.shape[0]):
    gdp_table[0, i] = np.mean(all_gdp[i])
    gdp_table[1, i] = np.std(all_gdp[i])
    gdp_table[2, i] = np.corrcoef(all_gdp[i][:-1], all_gdp[i][1:])[0,1]
    cpi_table[3, i] = np.corrcoef(all_gdp[i], all_gdp[i])[0,1]


# Generate cpi filtered datas
first_diff_cpi = cpi[1:] - cpi[:-1]
hp_cpi = hp.hp_filter(cpi)[1]
bp_cpi = bpf.bandpass_filter(cpi, 16, 6, 32)[16:-16]
all_cpi = np.array([first_diff_cpi, hp_cpi, bp_cpi])

cpi_table = np.zeros((4, all_gdp.shape[0]))

for i in range(all_cpi.shape[0]):
    cpi_table[0, i] = np.mean(all_cpi[i])
    cpi_table[1, i] = np.std(all_cpi[i])
    cpi_table[2, i] = np.corrcoef(all_cpi[i][:-1], all_cpi[i][1:])[0,1]
    cpi_table[3, i] = np.corrcoef(all_cpi[i], all_gdp[i])[0,1]


# Generate consumption filtered datas
first_diff_cons = cons[1:] - cons[:-1]
hp_cons = hp.hp_filter(cons)[1]
bp_cons = bpf.bandpass_filter(cons, 16, 6, 32)[16:-16]
all_cons = np.array([first_diff_cons, hp_cons, bp_cons])

cons_table = np.zeros((4,all_gdp.shape[0]))

for i in range(all_cons.shape[0]):
    cons_table[0, i] = np.mean(all_cons[i])
    cons_table[1, i] = np.std(all_cons[i])
    cons_table[2, i] = np.corrcoef(all_cons[i][:-1], all_cons[i][1:])[0,1]
    cons_table[3, i] = np.corrcoef(all_cons[i], all_gdp[i])[0,1]

# Generate investmentfiltered datas
first_diff_inv = inv[1:] - inv[:-1]
hp_inv = hp.hp_filter(inv)[1]
bp_inv = bpf.bandpass_filter(inv, 16, 6, 32)[16:-16]
all_inv = np.array([first_diff_inv, hp_inv, bp_inv])

inv_table = np.zeros((4,all_inv.shape[0]))

for i in range(all_inv.shape[0]):
    inv_table[0, i] = np.mean(all_inv[i])
    inv_table[1, i] = np.std(all_inv[i])
    inv_table[2, i] = np.corrcoef(all_inv[i][:-1], all_inv[i][1:])[0,1]
    inv_table[3, i] = np.corrcoef(all_inv[i], all_gdp[i])[0,1]

# <codecell>

gdp_table

# <codecell>

cpi_table

# <codecell>

cons_table

# <codecell>

inv_table

# <codecell>

gtab = pt()

# <codecell>

all_tables = np.array([gdp_table, cpi_table, cons_table, inv_table])

# <codecell>

blank_tables = np.zeros((4,4,3))

# <codecell>


# <codecell>


