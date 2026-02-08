#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
import emcee
import uvlfmc

processes = 16
nwalkers, nsteps = 100, 55000

DM = 'WDM'
fs_model = 'DPL'
f_z_dep = False
dataset_name = 'A'
z_arr = [6]

priors = {
    'log10_ks': (-2., 1.),
    'a_s1': (-1., 1.),
    'a_s2': (-1., 1.),
    'log10_M_c': (8., 10.),
    'log10_M_t': (8., 10.),
    'mx': (0.5, 5.),
}

LF_data=uvlfmc.dataset(dataset_name,z_arr)

likecls = uvlfmc.Distrib(LF_data, DM, fs_model, f_z_dep, priors)
lnlike = likecls.lnlike
lnprob = likecls.lnprob

starting_point = {
    'log10_ks': -0.99,
    'a_s1': 0.25,
    'a_s2': 0.25,
    'log10_M_c': 8.5,
    'log10_M_t': 8.5,
    'mx': 3.5,
}

result = op.minimize(lambda *args: -lnlike(*args), 
                     list(starting_point.values()),
                     method='TNC',
                     bounds=list(priors.values()))

ndim = len(priors)

filename = "sample_{}_{}_z{}_nw{}_nst{}_{}.h5".format(DM.lower(), 
                                                        dataset_name, 
                                                        '-'.join(str(x) for x in z_arr), 
                                                        nwalkers,
                                                        nsteps,
                                                        fs_model.lower())

pos = [result['x'] + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
with Pool(processes=processes) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    sampler.run_mcmc(initial_state=pos, nsteps=nsteps, progress=False, store=True)

