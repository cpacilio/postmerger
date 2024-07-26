## THIS MODULE COLLECTS USEFUL FUNCTIONS ##
## FOR REMNANT AND RINGDOWN PROPERTIES ##

def q_from_eta(eta):
    """
    Compute mass ratio from symmetric mass ratio.
    The convention used in this code is q=m1/m2>1

    Parameters
    ----------
    eta : float or array_like
        Symmetric mass ratio

    Returns
    -------
        float or array_like
    """
    out = (1-2*eta+np.sqrt(abs(1-4*eta)))*0.5/eta
    return out


def r_isco(spin):
    """
    Returns the dimensionless ISCO radius of a Kerr black hole.
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole

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
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole

    Returns
    -------
        float or array_like
    """
    return (1-2/3/r_isco(spin))**0.5


def L_isco(spin):
    """
    Returns the dimensionless angular momentum for a test particle at the ISCO of a Kerr black hole.
    Ref : Bardeen et al. 1972, 'Rotating black holes: locally nonrotating frames, energy extraction, and scalar synchrotron radiation'

    Parameters
    ----------
    spin : float or array_like
        Dimensionless spin of the black hole

    Returns
    -------
        float or array_like
    """
    return 2/3**1.5*(1+2*(3*r_isco(spin)-2)**0.5)


def final_mass(mass1,mass2,spin1,spin2,alpha=0.,beta=0.,gamma=0.,aligned_spins=False,method='B12'):
    """
    Returns the final mass (in solar masses) of the Kerr black hole remnant from a quasi-circular binary black hole merger.

    Parameters
    ----------
    mass1 : float or array_like
        Mass of the primary component (in solar masses).
    mass2 : float or array_like with shape: 
        Mass of the secondary component (in solar masses).
    spin1 : float or array_like 
        Magnitude of the dimensionless spin of the primary component.
    spin2 : float or array_like
        Magnitude of the dimensionless spin of the secondary component.
    alpha : float or array_like, optional
        Angle between the progenitor spins. Default: 0
    beta : float or array_like, otpional
        Angle between spin1 and the z direction. Default: 0
    gamma : float or array-like,optional
        Angle between spin2 and the z direction. Default: 0
    method : str, optional
        Method to use to compute the final spin. Allowed methods: ['B12','H15']
        If 'B12', it uses the fit in https://arxiv.org/abs/1206.3803
        If 'H15', it uses the fit in https://arxiv.org/abs/1508.07250
        Default: B12
    aligned_spins : bool, optional
        Whethter to assume aligned spins. If True, spin1 and spin2 can also be negative.
        Default: False

    Returns
    -------
        float or array-like
    """
    allowed_methods = ['B12','H15']
    if method not in allowed_methods:
        raise ValueError("Method must be one of "+str(allowed_methods))

    if not np.all(mass1>0):
        raise ValueError("mass1 must be greater than zero.")
    if not np.all(mass2>0):
        raise ValueError("mass2 must be greater than zero.")
    if not np.all(mass1>=mass2):
        raise ValueError("mass1 must be greater than mass2.")
    if not aligned_spins:
        if not np.all(spin1>=0):
            raise ValueError("spin1 must be non-negative. If you want spin1 to point in the negative direction, set beta!=0 or aligend_spins=True.")
        if not np.all(spin2>=0):
            raise ValueError("spin2 must be non-negatve. If you want spin2 to point in the negative direction, set gamma!=0 or aligned_spins=True.")
    if not np.all(np.abs(spin1)<=1):
        raise ValueError("spin1 magnitude must be no greater than 1.")
    if not np.all(np.abs(spin2)<=1):
        raise ValueError("spin2 magnitude must be no greater than 1.")
    if not np.all((beta>=0)*(beta<=np.pi)):
        raise ValueError("beta must be between 0 and pi.")
    if not np.all((gamma>=0)*(gamma<=np.pi)):
        raise ValueError("gamma must be between 0 and pi.")

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
