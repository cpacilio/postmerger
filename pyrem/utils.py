## THIS MODULE COLLECTS USEFUL FUNCTIONS ##
## FOR REMNANT AND RINGDOWN PROPERTIES ##
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np

## fundamental constants from astropy
G = 6.6743e-11 ## gravitational constant
msun = 1.988409870698051e+30 ## solar mass
cc = 299792458.0 ## speed of light
## solar mass in secs
tsun = msun*G/cc**3


def custom_factorial(k):
    """
    Custom factorial that sets to zero negative arguments.
    Useful in the calculation of the Wigner 3j-symbol.
    """
    if k<0:
        return 0.
    return np.math.factorial(k)


def Wigner3j(j1,j2,j3,m1,m2,m3):
    """
    Returns Wigner 3j-symbol.
    Ref: https://mathworld.wolfram.com/Wigner3j-Symbol.html .
    See also: https://en.wikipedia.org/wiki/3-j_symbol .
    """
    if m1+m2+m3:
        return 0.
    ## verify triangular condition
    if (j3>j1+j2) or (j3<abs(j1-j2)):
        return 0.
    if abs(m1)>j1:
        return 0.
    if not isinstance(j1-m1,int):
        return 0.
    if abs(m2)>j2:
        return 0.
    if not isinstance(j2-m2,int):
        return 0.
    if abs(m3)>j3:
        return 0.
    if not isinstance(j3-m3,int):
        return 0.
    if not isinstance(j1+j2+j3,int):
        return 0.
    ##
    kmin = max(0,int(j2-j3-m1),int(j1-j3+m2))
    kmax = min(int(j1+j2-j3),int(j1-m1),int(j2+m2))
    tmp = [(-1)**k/custom_factorial(int(k))\
            /custom_factorial(int(j1+j2-j3-k))\
            /custom_factorial(int(j1-m1-k))\
            /custom_factorial(int(j2+m2-k))\
            /custom_factorial(int(j3-j2+m1+k))\
            /custom_factorial(int(j3-j1-m2+k))\
            for k in range(kmin,kmax+1)]
    out = (-1)**int(j1-j2-m3)*np.sqrt(custom_factorial(int(j1+j2-j3))*\
            custom_factorial(int(j1-j2+j3))*\
            custom_factorial(int(-j1+j2+j3))\
            /custom_factorial(int(j1+j2+j3+1)))*\
            np.sqrt(custom_factorial(int(j1-m1))*\
            custom_factorial(int(j1+m1))*\
            custom_factorial(int(j2-m2))*\
            custom_factorial(int(j2+m2))*\
            custom_factorial(int(j3-m3))*\
            custom_factorial(int(j3+m3)))*\
            sum(tmp)
    return out


def clebsch_gordan(j1,m1,j2,m2,j3,m3):
    """
    Returns Clebsch Gordan coefficients as function of the Wigner 3j-symbol.
    Ref: https://mathworld.wolfram.com/Clebsch-GordanCoefficient.html .
    """
    out = (-1)**int(j1-j2+m3)*np.sqrt(2*j3+1)*\
        Wigner3j(j1,j2,j3,m1,m2,-m3)
    return out


def q_from_eta(eta):
    """
    Compute mass ratio from symmetric mass ratio.
    The convention used in this code is q=m1/m2>1

    Parameters
    ----------
    eta : float or array_like
        Symmetric mass ratio.

    Returns
    -------
        float or array_like
    """
    if not np.all(eta<=0.25):
        raise ValueError("eta must be no greater than 0.25.")
    if eta==0.:
        return np.inf
    out = (1-2*eta+np.sqrt(abs(1-4*eta)))*0.5/eta
    return out


def r_isco(spin):
    """
    Returns the dimensionless ISCO radius of a Kerr black hole.
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'.

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole.

    Returns
    -------
        float or array_like
    """
    if not np.all(np.abs(spin)<=1):
        raise ValueError("spin magnitude must be no greater than 1.")
    Z1 = 1+(1-spin**2)**(1/3)*((1+spin)**(1/3)+(1-spin)**(1/3))
    Z2 = (3*spin**2+Z1**2)**0.5
    out = 3+Z2-np.sign(spin)*((3-Z1)*(3+Z1+2*Z2))**0.5
    return out


def E_isco(spin):
    """
    Returns the dimensionless energy for a test particle at the ISCO of a Kerr black hole.
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'.

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole.

    Returns
    -------
        float or array_like
    """
    return (1-2/3/r_isco(spin))**0.5


def L_isco(spin):
    """
    Returns the dimensionless angular momentum for a test particle at the ISCO of a Kerr black hole.
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'.

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole.

    Returns
    -------
        float or array_like
    """
    return 2/3**1.5*(1+2*(3*r_isco(spin)-2)**0.5)


def final_mass(mass1,mass2,spin1,spin2,alpha=0.,beta=0.,gamma=0.,aligned_spins=False,method='B12'):
    """
    Returns the final mass (in solar masses) of the Kerr black hole remnant from a quasi-circular binary black hole merger.
    All available methods are calibrated on numerical simulations of binaries with aligned spins.

    Parameters
    ----------
    mass1 : float or array_like
        Mass of the primary component (in solar masses).
    
    mass2 : float or array_like
        Mass of the secondary component (in solar masses).
    
    spin1 : float or array_like 
        Magnitude of the dimensionless spin of the primary component.
    
    spin2 : float or array_like
        Magnitude of the dimensionless spin of the secondary component.
    
    alpha : float or array_like. Default=0.
        Angle between the progenitor spins.
        This parameter is never used and it is made available ony for consistency with the arguments of the final_spin function.
    
    beta : float or array_like. Default=0.
        Angle between spin1 and the z direction.
    
    gamma : float or array-like. Default=0.
        Angle between spin2 and the z direction.
    
    method : str. Default='B12'.
        Method to use to compute the final spin. Allowed methods: ['B12','phenom'].
        If 'B12', it uses the fit in https://arxiv.org/abs/1206.3803 .
        If 'phenom', it uses the fit in https://arxiv.org/abs/1508.07250 .
    
    aligned_spins : bool. Default=False.
        Whethter to assume aligned spins. If True, spin1 and spin2 can also be negative.
        Enabling this option overwrites the parameters alpha, beta ang gamma, setting them to zero.

    Returns
    -------
        float or array_like
    """
    allowed_methods = ['B12','phenom']
    if method not in allowed_methods:
        raise ValueError("method must be one of "+str(allowed_methods))

    if not np.all(mass1>0):
        raise ValueError("mass1 must be greater than zero.")
    if not np.all(mass2>0):
        raise ValueError("mass2 must be greater than zero.")
    if not np.all(mass1>=mass2):
        raise ValueError("mass1 must be greater than mass2.")
    if not np.all((beta>=0)*(beta<=np.pi)):
        raise ValueError("beta must be between 0 and pi.")
    if not np.all((gamma>=0)*(gamma<=np.pi)):
        raise ValueError("gamma must be between 0 and pi.")
    if not np.all(np.abs(spin1)<=1):
        raise ValueError("spin1 magnitude must be no greater than 1.")
    if not np.all(np.abs(spin2)<=1):
        raise ValueError("spin2 magnitude must be no greater than 1.")
    if not aligned_spins:
        if not np.all(spin1>=0):
            raise ValueError("spin1 must be non-negative. If you want spin1 to point in the negative direction, set beta!=0 or aligend_spins=True.")
        if not np.all(spin2>=0):
            raise ValueError("spin2 must be non-negatve. If you want spin2 to point in the negative direction, set gamma!=0 or aligned_spins=True.")
    elif aligned_spins:
        alpha, beta, gamma = 0., 0., 0.

    q = mass1/mass2
    eta = q/(1+q)**2
    m_tot = mass1+mass2

    if method=='B12':
        ## use https://arxiv.org/abs/1206.3803
        a_tot = (spin1*np.cos(beta)*mass1**2+spin2*np.cos(gamma)*mass2**2)/(mass1+mass2)**2
        p0, p1 = 0.04827, 0.01707
        E_rad = (1-E_isco(a_tot))*eta+4*eta**2*(4*p0+16*p1*a_tot*(a_tot+1)+E_isco(a_tot)-1)
        m_rad = E_rad*m_tot
        m_fin = m_tot-m_rad

    elif method=='H15':
        ## use https://arxiv.org/abs/1508.07250
        a_tot = (spin1*np.cos(beta)*mass1**2+spin2*np.cos(gamma)*mass2**2)/(mass1+mass2)**2/(1-2*eta)
        E_rad = 0.0559745*eta+0.580951*eta**2-0.960673*eta**3+3.35241*eta**4
        E_rad = E_rad*(1+a_tot*(-0.00303023-2.00661*eta+7.70506*eta**2))/(1+\
                a_tot*(-0.67144-1.47569*eta+7.30468*eta**2))
        m_rad = E_rad*m_tot
        m_fin = m_tot-m_rad

    return m_fin


def final_spin(mass1,mass2,spin1,spin2,alpha=0.,beta=0.,gamma=0.,method='H16',aligned_spins=False,return_angle=False):
    """
    Returns the magnitude of the dimensionless final spin of the Kerr black hole remnant from a quasi-circular binary black hole merger.
    Optionally, returns the angle between the final spin and the orbital plane.
    Note that avaiable methods are calibrated on numerical simulations of binaries with aligned spins.
    The effects of precession are treated effectively: they are included by summing in quadrature the trasverse components of the initial spins to the fitted expression for the parallel component of the final spin, and assuming that the evolution of the transverse components of the spins has a negligible effect on the final expression. See https://dcc.ligo.org/T1600168/public for further details.

    Parameters
    ----------
    mass1 : float or array_like
        Mass of the primary component (in solar masses).
    
    mass2 : float or array_like:
        Mass of the secondary component (in solar masses).
    
    spin1 : float or array_like
        Magnitude of the dimensionless spin of the primary component.
    
    spin2 : float or array_like
        Magnitude of the dimensionless spin of the secondary component.
    
    alpha : float or array_like. Default=0.
        Angle between the progenitor spins.
        This parameter is never used and it is made available only for consistency with the arguments of the final_spin function.
    
    beta : float or array_like. Default=0.
        Angle between spin1 and the z direction.
    
    gamma : float or array-like. Default=0.
        Angle between spin2 and the z direction.
    
    method : str. Default='H16'.
        Method to use to compute the final spin. Allowed options: ['H16','phenom'].
        If 'H16', it uses the fit in https://arxiv.org/abs/1605.01938 .
        If 'phenom', it uses the fit in https://arxiv.org/abs/1508.07250 .
    
    aligned_spins : bool. Default=False.
        Whethter to assume aligned spins. If True, spin1 and spin2 can also be negative.
        Enabling this option overwrites the parameters alpha, beta ang gamma, setting them to zero.
    
    return_angle : bool. Default=False.
        Whether to return the angle between the final spin and the orbital plane.
        
    Returns
    -------
        final spin: float or array_like
        angle : float or array_like (optional)
    """
    allowed_methods = ['B12','phenom']
    if method not in allowed_methods:
        raise ValueError("method must be one of "+str(allowed_methods))

    if not np.all(mass1>0):
        raise ValueError("mass1 must be greater than zero.")
    if not np.all(mass2>0):
        raise ValueError("mass2 must be greater than zero.")
    if not np.all(mass1>=mass2):
        raise ValueError("mass1 must be greater than mass2.")
    if not np.all((beta>=0)*(beta<=np.pi)):
        raise ValueError("beta must be between 0 and pi.")
    if not np.all((gamma>=0)*(gamma<=np.pi)):
        raise ValueError("gamma must be between 0 and pi.")
    if not np.all(np.abs(spin1)<=1):
        raise ValueError("spin1 magnitude must be no greater than 1.")
    if not np.all(np.abs(spin2)<=1):
        raise ValueError("spin2 magnitude must be no greater than 1.")
    if not aligned_spins:
        if not np.all(spin1>=0):
            raise ValueError("spin1 must be non-negative. If you want spin1 to point in the negative direction, set beta!=0 or aligend_spins=True.")
        if not np.all(spin2>=0):
            raise ValueError("spin2 must be non-negatve. If you want spin2 to point in the negative direction, set gamma!=0 or aligned_spins=True.")
    elif aligned_spins:
        alpha, beta, gamma = 0., 0., 0.


    q = mass2/mass1 ## phenom fits use a different def of mass ratio than this code
    eta = q/(1+q)**2

    if method=='H16':
        ## Eq.(7) in https://arxiv.org/abs/1605.01938
        a_tot = (spin1*np.cos(beta)*mass1**2+spin2*np.cos(gamma)*mass2**2)/(mass1+mass2)**2
        xi = 0.41616
        a_eff = a_tot+xi*eta*(spin1*np.cos(beta)+spin2*np.cos(gamma))
        kappas = [[-3.82,-1.2019,-1.20764],\
                  [3.79245,1.18385,4.90494]]
        a_fin_z = a_tot+eta*(L_isco(a_eff)-2*a_tot*(E_isco(a_eff)-1))
        for i in range(len(kappas)):
            for j in range(len(kappas[i])):
                a_fin_z += kappas[i][j]*eta**(2+i)*a_eff**j

    elif method=='phenom':
        ## https://arxiv.org/abs/1508.07250
        a_tot = (spin1*np.cos(beta)*mass1**2+spin2*np.cos(gamma)*mass2**2)/(mass1+mass2)**2
        kappas = [[2*3**0.5,-0.085,0.102,-1.355,-0.868],\
                  [-4.399,-5.837,-2.097,4.109,2.064],\
                  [9.397],[-13.181]]
        a_fin_z = deepcopy(a_tot)
        for i in range(len(kappas)):
            for j in range(len(kappas[i])):
                a_fin_z += kappas[i][j]*eta**(1+i)*a_tot**j

    if aligned_spins and return_angle:
        return a_fin_z, 0.
    elif aligned_spins:
        return a_fin_z
    ## add xy components according to
    ## Eq.(23) in https://arxiv.org/abs/2301.06558
    ## Eq.(16) in https://arxiv.org/abs/1605.01938
    ## see also https://dcc.ligo.org/T1600168/public
    a_xy_2 = (spin1**2+q**4*spin2**2+2*q**2*spin1*spin2*np.cos(alpha)\
            -(spin1*np.cos(beta)+q**2*spin2*np.cos(gamma))**2)/(1+q)**4
    a_fin = (a_fin_z**2+a_xy_2)**0.5

    if return_angle:
        theta_fin = np.arccos(np.clip(a_fin_z/a_fin,-1.,1.))
        return np.clip(a_fin,-1.,1.), theta_fin
    else:
        return np.clip(a_fin,-1.,1.)

def qnm_Kerr(mass,spin,mode,prograde=1,qnm_method='L18',SI_units=False):
    """
    Returns the frequency and the damping time of a Kerr black hole.
    Conventions follow XXX.

    Parameters
    ----------
    mass : float or array-like
        Mass of the Kerr black hole.
    
    spin : float or array-like
        Dimensionless spin of the kerr black hole.
    
    mode : tuple
        A tuple (l,m,n) with the indices labeling the mode.
    
    prograde : int. Default=1.
        Allowed options: [-1,1]. If 1, return prograde modes. If -1, return retrograde modes.
    
    qnm_method : str. Default='interp'.
        The method used to approximate the Kerr spectrum. Allowed options: ['interp','L18'].
        If 'interp', it interpolates linearly from the numerical tables provided at https://pages.jh.edu/eberti2/ringdown/ .
            They are only defined for spin in [-0.998,0.998] and any use outside this range is not guaranteed to produce sensible results.
            Note that we only support 2<=l<=5, but original tables are also available for l=6 and 7.
        If 'L18', it uses the fits in https://arxiv.org/abs/1810.03550 . They are defined for spin in the whole physical range [-1,1].
    
    SI_units : bool. Default=False.
        If True, returns frenquency in units of Hz and damping time in units of s.
        If False, returns frequency and damping time in dimensionless units, rescaling them by tsun=G*M_SUM/c**3.

    Returns
    -------
        frequency : float or array-like
        damping time : float or array_like
    """
    mode_tmp = tuple(mode)

    allowed_methods = ['L18','interp']
    if qnm_method not in allowed_methods:
        raise ValueError("qnm_method must be one of "+str(allowed_methods))

    if hasattr(mass,'__len__') and (hasattr(spin,'__len__')) and len(mass)!=len(spin):
        raise TypeError("mass ans spin must have the same length.")
    if prograde not in [-1,1]:
        raise ValueError("prograde must be one of [-1,1].")
    if not np.all(mass>0):
        raise ValueError("mass must be greater than zero.")
    if not np.all(np.abs(spin)<=1):
        raise ValueError("spin magnitude must be no greater than 1.")

    if qnm_method=='L18':
        ## use https://arxiv.org/abs/1810.03550
        mode_tmp = (mode_tmp[0],prograde*mode_tmp[1],mode_tmp[2])
        spin_tmp = prograde*spin
        sign_m = np.sign(mode_tmp[1]+0.5)
        mode_tmp = (mode_tmp[0],abs(mode_tmp[1]),mode_tmp[2])
        l,m,n = mode_tmp
        ##
        kappa = (np.log(2-spin_tmp)/np.log(3))**(1/(2+l-abs(m)))
        ##
        A = {(2,2,0):np.array([1.,1.5578,1.9510,2.0997,1.4109,0.4106]),\
            (2,2,1):np.array([1.,1.8709,2.7192,3.0565,2.0531,0.5955]),\
            (3,3,0):np.array([1.5,2.0957,2.4696,2.6655,1.7584,0.4991]),\
            (3,3,1):np.array([1.5,2.3391,3.1399,3.5916,2.4490,0.7004]),\
            (4,4,0):np.array([2.,2.6589,2.9783,3.2184,2.1276,0.6034]),\
            (4,3,0):np.array([1.5,0.2050,3.1033,4.2361,3.0289,0.9084]),\
            (5,5,0):np.array([2.5,3.2405,3.4906,3.7470,2.4725,0.6994]),\
            (3,2,0):np.array([1.0225,0.2473,1.7047,0.9460,1.5319,2.2805,0.9215]),\
            (2,1,0):np.array([0.5891,0.1890,1.1501,6.0459,11.1263,9.3471,3.0384])}
        ##
        B = {(2,2,0):np.array([0.,2.9031,5.9210,2.7606,5.9143,2.7952]),\
            (2,2,1):np.array([0.,2.5112,5.4250,2.2857,5.4862,2.4225]),\
            (3,3,0):np.array([0.,2.9650,5.9967,2.8176,5.9327,2.7817]),\
            (3,3,1):np.array([0.,2.6497,5.5525,2.3472,5.4435,2.2830]),\
            (4,4,0):np.array([0.,3.0028,6.0510,2.8775,5.9897,2.8300]),\
            (4,3,0):np.array([0.,0.5953,3.0162,6.0388,2.8262,5.9152]),\
            (5,5,0):np.array([0.,3.0279,6.0888,2.9212,6.0365,2.8766]),\
            (3,2,0):np.array([0.0049,0.6653,3.1383,0.1632,5.7036,2.6852,5.8417]),\
            (2,1,0):np.array([0.0435,2.2899,5.8101,2.7420,5.8441,2.6694,5.7915])}
        ##
        A_mode, B_mode = A[mode_tmp], B[mode_tmp]
        omega_tilde = 0.
        for i in range(len(A_mode)):
            omega_tilde += A_mode[i]*np.exp(1j*B_mode[i])*kappa**i
        omega_re = np.real(omega_tilde)/mass
        omega_im = np.imag(omega_tilde)
        f = sign_m*omega_re/2/np.pi
        tau = mass/omega_im

    elif qnm_method=='interp':
        sign_p = prograde
        ## data available at https://pages.jh.edu/eberti2/ringdown/
        sign_m = int(np.sign(mode_tmp[1]+0.5))
        mode_tmp = (mode_tmp[0],abs(mode_tmp[1]),mode_tmp[2])
        l,m,n = mode_tmp
        ## if all spins are non-negative
        if np.all(spin>=0):
            if sign_p>=0:
                filename = dir_path+'/../data/bcw_qnm_tables/l%s/n%sl%sm%s.dat'%(l,n+1,l,m)
            elif sign_p<0:
                filename = dir_path+'/../data/bcw_qnm_tables/l%s/n%sl%smm%s.dat'%(l,n+1,l,m)
            try:
                chi, omega_re, omega_im, _, _ = np.loadtxt(filename).T
            except:
                raise ValueError('Tabulated values for the (%s,%s,%s) mode are not stored and cannot be interpolated'%mode)
            f = sign_p*sign_m*np.interp(np.abs(spin),chi,omega_re)/mass/2/np.pi
            tau = mass/np.interp(np.abs(spin),chi,-omega_im)
        ## elif all spins are negative
        elif np.all(spin<0):
            f, tau = qnm_Kerr(mass,-spin,mode=(l,-sign_m*m,n),prograde=-sign_p,qnm_method='interp')
        ## elif spins have mixed signature
        else:
            mask = spin>=0
            f = np.zeros_like(spin)
            tau = np.zeros_like(spin)
            if not hasattr(mass,'__len__'):
                mass_tmp = np.ones_like(spin)*mass
            else:
                mass_tmp = mass
            f[mask], tau[mask] = qnm_Kerr(mass_tmp[mask],spin[mask],mode=(l,sign_m*m,n),prograde=sign_p,qnm_method='interp')
            f[~mask], tau[~mask] = qnm_Kerr(mass_tmp[~mask],-spin[~mask],mode=(l,-sign_m*m,n),prograde=-sign_p,qnm_method='interp')
    
    if SI_units:
        f = f/tsun
        tau = tau*tsun
    
    return f, tau


def spherical_spheroidal_mixing(lm,mode,spin,method='BK14',prograde=1,qnm_method='interp',\
                               enforce_sxs_convention=True,s=-2,fitting_coeffs={}):
    """
    Returns spherical-spheroidal mixing coefficients mu_{m,l,lt,n} as defined by Eq. (5) in https://arxiv.org/abs/1408.1860 .
    Note that they are the complex conjugate of the mixing coefficients C_{lt,l,m,n} defined in https://arxiv.org/abs/1908.10377 .

    Parameters
    ----------
    lm : tuple
        Indices (lt,m) of the spherical harmonic.
    
    mode : tuple
        Indices (l,m,n) of the spheroidal harmonic.
    
    spin : float or array_like
        Dimensionless spin of the Kerr black hole.
    
    method : str. Default='BK14'.
        The method used to compute mixing coefficients. Allowed options: ['BK14','PT73'].
        If 'BK14', it uses the fitting coefficients presented in https://arxiv.org/abs/1408.1860 and provided at https://pages.jh.edu/eberti2/ringdown/ .
        If 'PT73', it uses the leading order expressions in perturbation theory, see Press & Teukolsky 1973, ` Perturbations of a rotating black hole. II. Dynamical stability of the Kerr metric`.
    
    prograde : int. Default=1.
        Allowed options: [-1,1]. If 1, return mixing coefficients for modes. If -1, return mixing coefficients for retrograde modes.
    
    qnm_method : str. Default='interp'.
        The method used to approximate the Kerr spectrum. Allowed options: ['interp','L18'].
        If 'interp', it interpolates linearly from the numerical tables provided at https://pages.jh.edu/eberti2/ringdown/ .
            They are only defined for spin in [-0.998,0.998] and any use outside this range is not guaranteed to produce sensible results.
            Note that we only support 2<=l<=5, but original tables are also available for l=6 and 7.
        If 'L18', it uses the fits in https://arxiv.org/abs/1810.03550 . They are defined for spin in the whole physical range [-1,1].
    
    enforce_sxs_convention : bool. Default=True.
        If True, applies a sign correction to match the conventions of SXS. See footnote 4 of https://arxiv.org/abs/1902.02731 .

    Returns
    -------
        mu_re : float or array_like
            Real part of the mixing coefficients.
        mu_im : float or array_like
            Imaginary part of the mixing coefficients.
    """
    allowed_methods = ['BK14','PT74']
    if method not in allowed_methods:
        raise ValueError("method must be one of "+str(allowed_methods))
    
    if lm[1]!=mode[1]:
        raise ValueError("lm and mode must have the same azimuthal number m.")
    
    lt,m = lm
    l,_,n = mode

    if method=='PT74':
        ## use clebsch gordan coeff
        if lt==l:
            return 1, 0
        f, tau = qnm_Kerr(1.,spin,mode=mode,prograde=prograde,\
                      qnm_method=qnm_method)
        omega = 2*np.pi*f - 1j/tau
        ##
        out = -2*(spin*omega)*s*np.sqrt((2*l+1)/(2*lt+1))*\
            clebsch_gordan(l,m,1,0,lt,m)*\
            clebsch_gordan(l,-s,1,0,lt,-s)
        out += (spin*omega)**2*\
            2/3*np.sqrt((2*l+1)/(2*lt+1))*\
            clebsch_gordan(l,m,2,0,lt,0)*\
            clebsch_gordan(l,-s,2,0,lt,-s)
        out = out/(l*(l+1)-lt*(lt+1))
        mu_re = np.real(out)
        mu_im = -np.imag(out)
    #####
    elif method=='BK14':
        print(spin)
        ## use Berti & Klein fits
        ## use the fits at https://arxiv.org/abs/1408.1860
        ## available at https://pages.jh.edu/eberti2/ringdown/
        if fitting_coeffs=={}:
            filename = dir_path+'/../data/berti_klein_mixing/swsh_fits.dat'
            x = np.loadtxt(filename,dtype='object')
            for xi in x:
                mlln = tuple(xi[:4].astype(int))
                fitting_coeffs[mlln] = [float(c) for c in xi[4:12]]
        allowed_keys = list(fitting_coeffs.keys())
        mlln = (m,lt,l,n)
        fre, fim = 1., 1.
        sign_p = prograde
        ## if spins have mixed signature
        if np.mean(spin<0) and np.mean(spin<0)!=1:
            mask = spin>=0
            mu_re = np.zeros_like(spin)
            mu_im = np.zeros_like(spin)
            mu_re[mask], mu_im[mask] = spherical_spheroidal_mixing(lm,mode,spin[mask],s=s,prograde=prograde,\
                            method=method,qnm_method=qnm_method,enforce_sxs_convention=enforce_sxs_convention)
            mu_re[~mask], mu_im[~mask] = spherical_spheroidal_mixing(lm,mode,spin[~mask],s=s,prograde=prograde,\
                              method=method,qnm_method=qnm_method,enforce_sxs_convention=enforce_sxs_convention)
            return mu_re, mu_im
        ## elif spins are all negative
        elif np.all(spin<0):
            sign_p = -sign_p
            fim *= -1
        if np.sign(mode[0]+0.5)*sign_p < 0:
            mlln = (-m,lt,l,n)
            fre *= (-1)**(l+lt)
            fim *= (-1)**(l+lt+1)
        if mlln not in allowed_keys:
            raise KeyError('(%s,%s,%s,%s) mode is not allowed by this method. Allowed modes are: '%mlln+str(allowed_keys))
        p1_re, p2_re, p3_re, p4_re, p1_im, p2_im, p3_im, p4_im = fitting_coeffs[mlln]
        mu_re = p1_re*abs(spin)**p2_re + p3_re*abs(spin)**p4_re + 1 - np.sign(abs(lt-l))
        mu_im = p1_im*abs(spin)**p2_im + p3_im*abs(spin)**p4_im
        mu_re = mu_re*fre
        mu_im = mu_im*fim
    #####
    if enforce_sxs_convention:
        mu_re *= (-1)**(l+lt)
        mu_im *= (-1)**(l+lt)
    ####
    return mu_re, mu_im
