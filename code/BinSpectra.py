import numpy as np
def binspec(wvl, flux, uncer, binfactor):
    wvllog = np.log(wvl)
    numberofnewbins = float(len(wvllog))/binfactor
    if float(numberofnewbins).is_integer():
        N = binfactor - 0.000000000000001
    else:
        N = binfactor
    integerbinfactor = int(N)    
    integerbinfactor2 = int(N)
    remainingbinfraction = N - integerbinfactor
    remainingbinfraction2 = N - integerbinfactor
    i = 0
    k = 0
    oldbinlength = wvllog[i+1] - wvllog[i]
    oldbinradius = float(oldbinlength)/2
    newbinlength = oldbinlength*N
    newbinradius = float(newbinlength)/2
    binradiusdiff = newbinradius - oldbinradius
    newbins = []
    newflux = []
    newuncer = []
    newbincenter = wvllog[i] + binradiusdiff
    newbins.append(np.exp(newbincenter))
    newflux.append(flux[i:i + integerbinfactor].sum() + remainingbinfraction * flux[i + integerbinfactor])
    newuncer.append(np.sqrt(np.sum(np.square(uncer[i:i + integerbinfactor])) + np.square(remainingbinfraction * uncer[i + integerbinfactor])))
    while i + integerbinfactor + (int(N - 1 + remainingbinfraction) + 1) <= len(wvllog) - 1:
        i = i + integerbinfactor
        k = k + integerbinfactor2
        integerbinfactor = int(N - (1 - remainingbinfraction)) + 1
        integerbinfactor2 = int(N - (1 - remainingbinfraction2)) + 1
        remainingbinfraction2 = N - (integerbinfactor - remainingbinfraction2)
        newbincenter +=  (1 - remainingbinfraction) * (wvllog[i+1] - wvllog[i])  + wvllog[i + integerbinfactor] - wvllog[i + 1] + remainingbinfraction2 * (wvllog[i + integerbinfactor] - wvllog[i + integerbinfactor - 1])
        newbins.append(np.exp(newbincenter))
        newflux.append((1 - remainingbinfraction) * flux[i] + flux[i + 1:i + integerbinfactor].sum() + remainingbinfraction2 * flux[i + integerbinfactor])
        newuncer.append(np.sqrt(np.square((1 - remainingbinfraction) * uncer[i]) + np.sum(np.square(uncer[i + 1:i + integerbinfactor])) + np.square(remainingbinfraction2 * uncer[i + integerbinfactor])))
        remainingbinfraction = N - (integerbinfactor - remainingbinfraction)
    return np.array(newbins), np.array(newflux), np.array(newuncer)