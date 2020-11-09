#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool
import dynesty
import uvlfmc
import pickle

processes = 8
nlive = 2000

DM = 'WDM'
fs_model = 'PL'
f_z_dep = False
dataset_name = 'B'
z_arr = [6]

priors = {
    'log10_ks': (-2., 1.),
    'a_s1': (-1., 1.),
    'log10_M_t': (8., 10.),
    'mx': (0.5, 10.),
}

LF_data=uvlfmc.dataset(dataset_name,z_arr)

likecls = uvlfmc.Distrib(LF_data, DM, fs_model, f_z_dep, priors)
lnlike = likecls.lnlike
ptform = likecls.ptform

ndim = len(priors)

with Pool(processes=processes) as pool:
    dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim, pool=pool, queue_size=processes,nlive=nlive)
    dsampler.run_nested(print_progress=True)
dresults = dsampler.results

fout='dresults_{}_{}_z{}_{}.pkl'.format(dataset_name, DM.lower(), '-'.join(str(z) for z in z_arr), fs_model.lower())
with open(fout, 'wb') as fn:
    pickle.dump(dresults,fn)
