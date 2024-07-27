## THIS MODULE COLLECTS USEFUL FUNCTIONS ##
## FOR REMNANT AND RINGDOWN PROPERTIES ##

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
    alpha : float or array_like, optional
        Angle between the progenitor spins. Default: 0
        This parameter is never used and it is made available ony for consistency with the arguments of the final_spin function.
    beta : float or array_like, optional
        Angle between spin1 and the z direction. Default: 0
    gamma : float or array-like,optional
        Angle between spin2 and the z direction. Default: 0
    method : str, optional
        Method to use to compute the final spin. Allowed methods: ['B12','phenom'].
        If 'B12', it uses the fit in https://arxiv.org/abs/1206.3803 .
        If 'phenom', it uses the fit in https://arxiv.org/abs/1508.07250 .
        Default: B12
    aligned_spins : bool, optional
        Whethter to assume aligned spins. If True, spin1 and spin2 can also be negative.
        Enabling this option overwrites the parameters alpha, beta ang gamma, setting them to zero.
        Default: False

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
    alpha : float or array_like, optional
        Angle between the progenitor spins. Default: 0
        This parameter is never used and it is made available only for consistency with the arguments of the final_spin function.
    beta : float or array_like, optional
        Angle between spin1 and the z direction. Default: 0
    gamma : float or array-like,optional
        Angle between spin2 and the z direction. Default: 0
    method : str, optional
        Method to use to compute the final spin. Allowed methods: ['H16','phenom']
        If 'H16', it uses the fit in https://arxiv.org/abs/1605.01938 .
        If 'phenom', it uses the fit in https://arxiv.org/abs/1508.07250 .
        Default: H16
    aligned_spins : bool, optional
        Whethter to assume aligned spins. If True, spin1 and spin2 can also be negative.
        Enabling this option overwrites the parameters alpha, beta ang gamma, setting them to zero.
        Default: False
    return_angle : bool, optional
        Whether to return the angle between the final spin and the orbital plane.
        Default: False
        

    Returns
    -------
        float or array_like, float or array_like (optional)
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
