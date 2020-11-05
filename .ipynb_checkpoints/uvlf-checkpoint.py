
import numpy as np
import pandas as pd
import astropy
import hmf
from hmf.alternatives.wdm import MassFunctionWDM
from hmf.density_field.filters import SharpK, TopHat
from hmf.density_field.transfer_models import EH, camb, BBKS,BondEfs

h = 0.673
Omegab = 0.049
Omegam0 = 0.315
ns = 0.965
Tcmb = 2.725

ro = 2.775*10**11 #(in h^2 Msun/Mpc^3)
rodm = Omegam0*ro
aa = 2.7 #parameter a in Lovell:15
mpc = 3.0856*10**24

sigm8 = 0.83
OmegaM = Omegam0
OmegaLambda = 1.0 - Omegam0


def dataset(name,z_arr, dust_corr=False):
    if dust_corr and z_arr!=[6]:
        raise Exception
    corr=''
    if dust_corr: corr='_corr'
    LFdata={}
    
    if name=='A':
        z=6
        fname='UVLF_Atek_z{}{}.csv'.format(z, corr)
        
        df=pd.read_csv(fname)
        x=df.mag.to_numpy()
        y=df.logphi.to_numpy()
        yerr=df.delta.to_numpy()
        LFdata[z]=np.array([x,y,yerr,yerr])
        
        
    elif name=='B':
        for z in z_arr:
            fname='UVLF_Bouwens_z{}{}.csv'.format(z, corr)
            df=pd.read_csv(fname)
            x=df.mag.to_numpy()
            y=df.logphi.to_numpy()
            yerr_up=df.deltaup.to_numpy()
            yerr_down=df.deltado.to_numpy()
            LFdata[z]=np.array([x,y,yerr_up,yerr_down])
            
        
    
    for z,lf in LFdata.items():
        
        mask=lf[0]>-20
        LFdata[z]=lf[:,mask]  
    return LFdata


def halos_mass_function(z, function_name='SMT', DM='WDM',mx=20 ):
   

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
        cosmology=astropy.cosmology.FlatLambdaCDM(H0=100*h,Om0=Omegam0,Ob0=Omegab,Tcmb0=Tcmb)
        mass_function  = hmf.MassFunction( z=z, Mmin=7, Mmax=12, sigma_8=sigm8,n=ns,cosmo_model=cosmology, transfer_model=EH,hmf_model=function_name )
        M_h = mass_function.m / h
        HMF = mass_function.dndm * h**4

        return M_h, HMF
    elif DM=='WDM':
        cosmology=astropy.cosmology.FlatLambdaCDM(H0=100*h,Om0=Omegam0,Ob0=Omegab,Tcmb0=Tcmb)
        filter_pars={'c':aa}
        
        hmf_params={'a':1, 'p':0.3,'A':0.322}
        mass_function=MassFunctionWDM(Mmin=7, Mmax=12,z=z, sigma_8=sigm8, n=ns, cosmo_model=cosmology, transfer_model=EH, hmf_model=function_name, hmf_params=hmf_params, 
                                      filter_params=filter_pars, filter_model=SharpK, wdm_mass=mx)
        M_h = mass_function.m / h
        HMF = mass_function.dndm * h**4

        return M_h, HMF   
    
def compute_LF( z=6, ks=1, a_s1=0.215, a_s2=0.215, M_c=1e10, gamma=0, M_t=5.e5, HMF_name='ST',DM='WDM', mx=20, ind=-1 ):

    
    
    
    ##cosmology --Hubble parameter
    cosmo=astropy.cosmology.FlatLambdaCDM(H0=h*100,Om0=Omegam0)
    ### constants
    K_UV = 1.15e-28 ### Mo.yr-1 / ( erg.s-1.Hz-1 )
    M_h, HMF = halos_mass_function( z=z, function_name=HMF_name,DM=DM,mx=mx )
    f_z=((1+z)/7)**gamma    
    fst = ks * ((((M_h / M_c)**a_s1+(M_h / M_c)**a_s2)/2)**ind) * f_z
    #ks -- f_s_th/t_s
    M_s = fst * Omegab  / Omegam0 * M_h
    #f_esc = f_esc_10 * (M_h / 10**10)**a_esc
    f_duty = np.exp( -M_t / M_h )
    M_s_dot = M_s * cosmo.H(z).to('1/yr').value
    L_UV = M_s_dot / K_UV
    M_UV = 51.63 - np.log10( L_UV ) / 0.4
    phi = f_duty * HMF * np.abs( np.gradient( M_h, M_UV ) )
        

    
    return M_UV, phi