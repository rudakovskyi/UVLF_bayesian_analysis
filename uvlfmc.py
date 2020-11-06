import numpy as np
import pandas as pd
import astropy
import hmf
import os
from hmf.alternatives.wdm import MassFunctionWDM
from hmf.density_field.filters import SharpK, TopHat
from hmf.density_field.transfer_models import EH, camb, BBKS, BondEfs

h = 0.673
Omegab = 0.049
Omegam0 = 0.315
ns = 0.965
Tcmb = 2.725

ro = 2.775*10**11 #(in h^2 Msun/Mpc^3)
rodm = Omegam0*ro
mpc = 3.0856*10**24

sigm8 = 0.83
OmegaM = Omegam0
OmegaLambda = 1.0 - Omegam0

datapath = 'data'

cosmology = astropy.cosmology.FlatLambdaCDM(H0=100*h, Om0=Omegam0, Ob0=Omegab, Tcmb0=Tcmb)

def dataset(name, z_arr, dust_corr=False):
    if dust_corr and z_arr!=[6]:
        raise Exception
    corr=''
    if dust_corr: corr='_corr'
    LFdata={}
    
    if name == 'A':
        z = 6
        fname = 'UVLF_Atek_z{}{}.csv'.format(z, corr)
        
        df = pd.read_csv(os.path.join(datapath, fname))
        x = df.mag.to_numpy()
        y = df.logphi.to_numpy()
        yerr = df.delta.to_numpy()
        LFdata[z] = np.array([x, y, yerr, yerr])
        
        
    elif name == 'B':
        for z in z_arr:
            fname = 'UVLF_Bouwens_z{}{}.csv'.format(z, corr)
            df = pd.read_csv(os.path.join(datapath, fname))
            x = df.mag.to_numpy()
            y = df.logphi.to_numpy()
            yerr_up = df.deltaup.to_numpy()
            yerr_down = df.deltado.to_numpy()
            LFdata[z] = np.array([x, y, yerr_up, yerr_down])
    
    for z,lf in LFdata.items():
        mask = lf[0] > -20
        LFdata[z] = lf[:, mask]  

    return LFdata


def halos_mass_function(z, function_name='ST', DM='WDM', mx=20):
    '''
    Compute the halos mass funtion
    inputs:
    - z : redshift (int)
    - function_name ='SMT' : the fitting function (string)
    
    choice possible :
    'Angulo', 'AnguloBound', 'Behroozi', 'Bhattacharya', 'Courtin', 'Crocce', 'Ishiyama', 'Jenkins', 
    'Manera', 'PS', 'Peacock', 'Pillepich', 'Reed03', 'Reed07', 'SMT', 'ST', 'Tinker08', 'Tinker10',
    'Warren', 'Watson', 'Watson_FoF',
    
    outputs:
    - M_h : halos mass [M_sun]
    - HMF : dn/dM [M_sun-1.Mpc-3]
    
    It used the module hmfCALC:
    http://hmf.icrar.org/
    https://github.com/steven-murray/hmf/blob/master/README.rst
    https://hmf.readthedocs.io/en/latest/index.html
    '''
    
    if DM=='CDM':
        mass_function = hmf.MassFunction(z=z, Mmin=7, Mmax=12, 
                                         sigma_8=sigm8, n=ns, 
                                         cosmo_model=cosmology, 
                                         transfer_model=EH,
                                         hmf_model=function_name)
        M_h = mass_function.m / h
        HMF = mass_function.dndm * h**4

        return M_h, HMF
    
    elif DM=='WDM':
        filter_pars={'c': 2.5} #parameter a in Lovell:15
        hmf_params={'a':1, 'p': 0.3, 'A': 0.322} 
        
        mass_function=MassFunctionWDM(z=z, Mmin=7, Mmax=12, 
                                      sigma_8=sigm8, n=ns, 
                                      cosmo_model=cosmology, 
                                      transfer_model=EH, 
                                      hmf_model=function_name, hmf_params=hmf_params, 
                                      filter_model=SharpK, filter_params=filter_pars,
                                      wdm_mass=mx)
        M_h = mass_function.m / h
        HMF = mass_function.dndm * h**4

        return M_h, HMF   
    
def compute_LF(z=6, 
               log10_ks=1, a_s1=0.5, a_s2=0.5, 
               log10_M_c=10, log10_M_t=6, mx=20, gamma=0,
               HMF_name='ST', DM='WDM',
               ind=1):
    M_t=10**log10_M_t
    M_c=10**log10_M_c
    ks=10**log10_ks
        
    K_UV = 1.15e-28 ### Mo.yr-1 / ( erg.s-1.Hz-1 )
    M_h, HMF = halos_mass_function(z=z, function_name=HMF_name, DM=DM, mx=mx)
    f_z = ((1 + z) / 7)**gamma
    fst = ks * ((((M_h / M_c)**a_s1 + (M_h / M_c)**a_s2) / 2)**ind) * f_z
    #ks -- f_s_th/t_s
    M_s = fst * Omegab / Omegam0 * M_h
    f_duty = np.exp(-M_t / M_h)
    M_s_dot = M_s * cosmology.H(z).to('1/yr').value
    L_UV = M_s_dot / K_UV
    M_UV = 51.63 - np.log10(L_UV) / 0.4
    phi = f_duty * HMF * np.abs(np.gradient(M_h, M_UV))
    
    return M_UV, phi


class Distrib:
    def __init__(self, dataset, DM, fs_model, f_z_dep, priors):
        self.dataset = dataset
        self.DM = DM
        self.fs_model = fs_model
        self.f_z_dep = f_z_dep
        self.priors = priors
        self.ptransformarr = np.array([x[1]-x[0] for x in priors.values()])
        self.priors_lo = np.array([x[0] for x in priors.values()])
        self.priors_up = np.array([x[1] for x in priors.values()])
        self.log_norm_vol = np.sum(np.log(self.ptransformarr))
        
        if fs_model=='DPLM':
            self.ind = -1
        else:
            self.ind = 1
            
        #compute data-dependent normalizing constant
        self.norm_const=0
        for z, lf in dataset.items():
                x = lf[0]
                y = lf[1]
                yerr_down = lf[3]
                yerr_up = lf[2]
                self.norm_const += -0.5 * len(y) * np.log((np.pi/2)) -\
                    sum(np.log(yerr_down+yerr_up))
    
    @staticmethod
    def _chi2(y_pred, y, yerr_up, yerr_down):
        return np.sum(
            np.heaviside(y_pred - y, 0) * ((y_pred - y) / yerr_up)**2 +\
            np.heaviside(y - y_pred, 0) * ((y_pred - y) / yerr_down)**2
            )

    def lnlike(self, theta):
        pars = dict(zip(self.priors, theta))
        pars.setdefault('a_s2', pars['a_s1'])
        
        chi2_sum=0
        for z, lf in self.dataset.items():
            x = lf[0]
            y = lf[1]
            yerr_down = lf[3]
            yerr_up = lf[2]
                                
            M_UV_Model, phi_Model = compute_LF(
                z=z, 
                **pars,
                DM=self.DM,
                ind = self.ind)

            y_pred = np.interp(x, M_UV_Model[::-1], np.log10(phi_Model)[::-1])
            chi2_sum+=self._chi2(y_pred,y,yerr_up,yerr_down)
            
        return -0.5*chi2_sum+self.norm_const    

    def ptform(self, u):
        return u * self.ptransformarr + self.priors_lo
    
    def lnprior(self, theta):
        if np.all(np.logical_and(theta > self.priors_lo, 
                                 theta < self.priors_up)):
            return -self.log_norm_vol
        else:
            return -np.inf
        
    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll=self.lnlike(theta)
        if np.isnan(ll):
            return -np.inf
        
        return ll+lp
    
    
        
        
        
        
