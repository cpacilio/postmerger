'swsh_fits.dat' presents fitting parameters for the overlap
between spherical harmonics and spheroidal harmonics, as a function of the spin
parameter j.


The functions we fit are
mu_{m,l,l',n'}(j) = \int {}_{-2}Y_{l,m} {}_{-2}S_{l',m,n'}(j)^* d\Omega


We present fitting functions as

Re(mu_{m,l,l',n'}(j)) ~ delta_{l,l'} + p1r j^{p2r} + p3r j^{p4r}

and

Im(mu_{m,l,l',n'}(j)) ~ p1i j^{p2i} + p3i j^{p4i}

with p2r and p2i in [0,20], p4r and p4i in [0, 100], p2r < p4r, and p2i < p4i


The columns listed are

m  l  l'  n'  p1r  p2r  p3r  p4r  p1i  p2i  p3i  p4i  sr  si  mr  mi

sr, si, mr, mi represent additional information on the fits:

- sr is the standard deviation between the real part of the fit and that of the
numerical function, defined as sr = sqrt((1/N) sum |Re(mu_{fit} - mu_{num})|)

- si is similar for the imaginary part,
si = sqrt((1/N) sum |Im(mu_{fit} - mu_{num})|)

- mr is the maximum absolute deviation between the real part of the fit and that
of the numerical function, mr = max_{N} |Re(mu_{fit} - mu_{num})|

- mi is similar for the imaginary part, mi = max_{N} |Im(mu_{fit} - mu_{num})|


The fits are least squares fits, i.e. p1r, ..., p4r minimize sr, and
p1i, ..., p4i minimize si.


