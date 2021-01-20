from glob import glob
import xarray as xr
import parmap
from multiprocessing import Pool
import numpy as np


def ComputeTropo(t,T):
    # use tropopause algorithm from http://www.inscc.utah.edu/~reichler/research/projects/TROPO/code.txt
    #  cite Reichler, T., M. Dameris, R. Sausen (2003): Determining the tropopause height from gridded data, Geophys. Res. L., 30, No. 20, 2042
    from GeneralPython import tropopause as tpp
    ttemp = T.isel(time=t)
    if 'lon' in ttemp.coords:
        nlon = len(ttemp.lon)
    else:
        nlon = 1
        ttemp = ttemp.expand_dims('lon',axis=-1)
    ttemp = ttemp.transpose('lon','lat','level')
    tp,tperr = tpp.tropo(ttemp.values,ttemp.level.values,45000,7500,7500,True,[nlon,len(ttemp.lat),len(ttemp.level)])
    if nlon == 1:
        return xr.DataArray(tp.squeeze()*0.01,coords=[ttemp.lat],name='tpp')
    else:
        return xr.DataArray(tp*0.01,coords=[ttemp.lon,ttemp.lat],name='tpp')

files = glob('/Volumes/cyclone/CM2.1/atmos*.temp.nc')
files.sort()

pool = Pool()
for file in files:
    temp = xr.open_dataarray(file)
    temp.load()
    times = np.arange(len(temp.time))
    # compute tropopause for each time step
    tps = parmap.map(ComputeTropo,times,temp,pm_pbar=True,pm_pool=pool)
    tp = xr.concat(tps,dim='time')
    tp = tp.assign_coords({'time':temp.time})
    outFile = file.replace('temp','tpp')
    tp.to_netcdf(outFile)
    print('written file '+outFile)
pool.close()

