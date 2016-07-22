# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:26:24 2016

@author: dongyu
"""

import numpy as np
from netCDF4 import Dataset, num2date
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
import utm
import string
import pdb


class tide_average(object):
    """
    class that averages SUNTANS velocity over a tidal cycle
    add the residual to ROMS velocity
    """
    period = 24.
    
    def __init__(self, starttime, endtime, **kwargs):
        """
        Initiate a few variables
        """
        self.__dict__.update(kwargs)
        self.start = starttime
        self.end = endtime
        self.starttime = datetime.strptime(starttime, '%Y-%m-%d')
        self.endtime = datetime.strptime(endtime, '%Y-%m-%d')
        self.wdr = os.getcwd()
        
        mypath = os.getcwd()
        mypath = os.path.abspath(os.path.join(mypath, os.pardir))
        mypath = os.path.abspath(os.path.join(mypath, os.pardir))
        self.sunfile = mypath + '/Wind_driven_flow/SUNTANS_file/GalvCoarse.nc'
        #self.sunfile = self.wdr+'/SUNTANS_file/GalvCoarse_0000.nc'
        self.romsfile = self.wdr+'/download_ROMS/txla_subset_HIS.nc'
        
        ##
        self.observation()
        self.read_suntans(self.sunfile)
        self.read_roms(self.romsfile)
        ##
        self.tide_uv()
        #self.extract_tide()
        
    def observation(self):
        """
        source the observational data
        """
        filename = self.wdr+'/DATA/g06010.txt'

        data = []        
        w = file(filename, 'r').readlines()
        for s in w:
            if '#' not in s:
                line = s.split()
                if line != []:
                    data.append(line)
        #pdb.set_trace()
        Spd = []
        Dir = []
        Time = []
        for i in range(len(data)):            
            Spd.append(string.atof(data[i][2])/100.)
            Dir.append(string.atof(data[i][3]))
            Time.append(data[i][0].replace("-", "")+data[i][1].replace(":", ""))

        time_obs = []
        for i in range(len(Time)):
            time_obs.append(datetime.strptime(Time[i],'%Y%m%d%H%M%S'))
            
        ind0 = self.findNearest(self.starttime,time_obs)
        ind1 = self.findNearest(self.endtime,time_obs)
        self.time_obs = time_obs[ind0:ind1+1]
        self.spd_obs = Spd[ind0:ind1+1]
        self.dir_obs = Dir[ind0:ind1+1]
        
        
        
    def read_suntans(self, sunfile):
        """
        funtion that reads SUNTANS output file
        """
        nc = Dataset(sunfile, 'r')
        timei = nc.variables['time']
        time = num2date(timei[:], timei.units)
        ind0 = self.findNearest(self.starttime,time)
        ind1 = self.findNearest(self.endtime,time)
        self.time_sun = time[ind0:ind1+1]
        self.xv = nc.variables['xv'][:]
        self.yv = nc.variables['yv'][:]
        self.Nc = len(self.xv)
        
        self.uc = nc.variables['uc'][:][ind0:ind1+1,0,:]
        self.vc = nc.variables['vc'][:][ind0:ind1+1,0,:]
        
        self.lon_sun = np.zeros_like(self.xv)
        self.lat_sun = np.zeros_like(self.yv)
        for i in range(len(self.xv)):
            self.lat_sun[i], self.lon_sun[i] = utm.to_latlon(self.xv[i],self.yv[i], 15, 'U')[0:2]
        
        
    def read_roms(self, romsfile):
        """
        funtion that reads ROMS velocity
        """
        
        nc = Dataset(romsfile, 'r')
        
        timei = nc.variables['ocean_time']
        time = num2date(timei[:], timei.units)
        ind0 = self.findNearest(self.starttime,time)
        ind1 = self.findNearest(self.endtime,time)
        self.time_roms = time[ind0:ind1+1]
        self.lon = nc.variables['lon_rho'][:][0:-1,0:-1]
        self.lat = nc.variables['lat_rho'][:][0:-1,0:-1]   
        mask = nc.variables['mask_rho'][:][0:-1,0:-1]
        uu = nc.variables['u'][ind0:ind1+1,0,:,:]
        vv = nc.variables['v'][ind0:ind1+1,0,:,:]
        ang = nc.variables['angle'][:][0:-1,0:-1]
        
        # average u,v to central rho points           
        uroms = np.zeros((len(self.time_roms),mask.shape[0], mask.shape[1]))
        vroms = np.zeros((len(self.time_roms),mask.shape[0], mask.shape[1]))
        for t in range(len(self.time_roms)):
            uroms[t,:,:] = self.shrink(uu[t,:,:], mask.shape)
            vroms[t,:,:] = self.shrink(vv[t,:,:], mask.shape)
    
        uroms[np.abs(uroms)>3] = 0
        #uroms[:,mask==0] = 0
        vroms[np.abs(vroms)>3] = 0
        #vroms[:,mask==0] = 0
        
        def rot2d(x, y, ang):
            """
            rotate vectors by geometric angle
            This routine is part of Rob Hetland's OCTANT package:
            https://github.com/hetland/octant
            """
            xr = x*np.cos(ang) - y*np.sin(ang)
            yr = x*np.sin(ang) + y*np.cos(ang)
            return xr, yr
        
        self.uroms, self.vroms =  rot2d(uroms, vroms, ang)

    def tide_uv(self):
        """
        obtain the tidal induced velocity
        """
        u_avg = self.time_average(self.uc, self.period)
        v_avg = self.time_average(self.vc, self.period)
        time = self.align(self.time_sun, self.period)
        uc = self.align(self.uc, self.period)
        vc = self.align(self.vc, self.period)
        spd_sun = np.sqrt(uc*uc+vc*vc)
        dir_sun = self.flow_direction(uc, vc)        
        
        uroms = self.align(self.uroms, self.period)
        vroms = self.align(self.vroms, self.period)
        spd_roms = np.sqrt(uroms*uroms+vroms*vroms)
        dir_roms = self.flow_direction(uroms, vroms)
        uroms_avg = self.time_average(uroms, self.period)        
        vroms_avg = self.time_average(vroms, self.period)
        
        loc = [29.344896, -94.746949]  #observation site       
        #loc = [29.309838, -94.539067] #good position
        #loc = [29.303663, -94.575288] #best postion
        #loc = [29.266714, -94.489439] #mid ranged
        #loc = [29.058223, -94.890604] #not suitable for tidally average
        #loc = [29.228974, -94.447277] #far area, tide is not dominant, or SUNTANS is incorrect        
        ## SUNTANS
        ind = self.findNearset(loc[1], loc[0], self.lon_sun, self.lat_sun)[0][0] 
        ## ROMS
        ind2 = self.findNearset(loc[1], loc[0], self.lon, self.lat)
        J0=ind2[0][0] 
        I0=ind2[0][1]
        
        unew = uc[:,ind]-u_avg[:,ind]+uroms_avg[:,J0,I0]
        vnew = vc[:,ind]-v_avg[:,ind]+vroms_avg[:,J0,I0]
        spd_new = np.sqrt(unew*unew+vnew*vnew)
        dir_new = self.flow_direction(unew, vnew)

#        l1 = plt.plot(time, uc[:,ind]-u_avg[:,ind]+uroms[:,J0,I0], label='new')
#        l2 = plt.plot(time, uc[:,ind], label='SUNTANS')
#        l3 = plt.plot(time, u_avg[:,ind], label='averaged')
#        l4 = plt.plot(time, uroms[:,J0,I0], label='ROMS')
#        l5 = plt.plot(time, uc[:,ind]-u_avg[:,ind]+uroms_avg[:,J0,I0], label='+ROMS avg')
#        l6 = plt.plot(time, uroms_avg[:,J0,I0], label='ROMS avg')
#        plt.legend()
#        plt.show()         
        fig, (ax1, ax2) = plt.subplots(2, figsize=(14,12), sharex=True)
        #l1 = ax1.plot(time, spd_new[:], '-r', label='new')
        l2 = ax1.plot(time, spd_sun[:,ind], 'ob', label='SUNTANS')
        l3 = ax1.plot(time, spd_roms[:,J0,I0], 'oy', label='ROMS')
        l4 = ax1.plot(self.time_obs, self.spd_obs[:], '-k', label='obs')
        ax1.legend()
        ax1.set_ylabel('velocity (m/s)', fontsize=18)
        ax1.set_title('flow velocity at (%s, %s)'%(str(loc[0]), str(loc[1])), fontsize=20)        
        
        
        #l5 = ax2.plot(time, dir_new[:], 'or', label='new')
        l6 = ax2.plot(time, dir_sun[:, ind], 'ob', label='SUNTANS')
        l7 = ax2.plot(time, dir_roms[:,J0,I0], 'oy', label='ROMS')
        l8 = ax2.plot(self.time_obs, self.dir_obs[:], '-k', label='obs')
        ax2.legend()
        ax2.set_ylabel('flow angle', fontsize=18)
        ax2.set_title('flow direction at (%s, %s)'%(str(loc[0]), str(loc[1])), fontsize=20)    
        
        plt.show() 
        pdb.set_trace()
        
        
    def extract_tide(self):
        """
        decompose tidal current
        """
        from ttide.t_tide import t_tide
        uc = self.align(self.uc, self.period)
        uroms = self.align(self.uroms, self.period)

        loc = [29.309838, -94.539067]
        ## SUNTANS
        ind = self.findNearset(loc[1], loc[0], self.lon_sun, self.lat_sun)[0][0] 
        ## ROMS
        ind2 = self.findNearset(loc[1], loc[0], self.lon, self.lat)
        J0=ind2[0][0] 
        I0=ind2[0][1]
        
        u = uc[:,ind]
        u2 = uroms[:,J0,I0]      
        
        [name, freq, tidecon, xout] = t_tide(u)
        [name2, freq2, tidecon2, xout2] = t_tide(u2)
        pdb.set_trace()
        
        amp = tidecon[:,0]
        pha = tidecon[:,2]
        amp2 = tidecon[:,0]
        pha2 = tidecon[:,2]
        time = range(uc.shape[0]) 
        
        uu = np.zeros_like(u)
        uu2 = np.zeros_like(u2)
        for i in range(len(time)):
            uu[i] = sum(amp[:]*np.cos(2*np.pi*freq[:]*time[i]-np.radians(pha[:])))
            uu2[i] = sum(amp2[:]*np.cos(2*np.pi*freq2[:]*time[i]-np.radians(pha2[:])))
        
#        l1 = plt.plot(xout, label='xout')
#        l2 = plt.plot(u2, label='u')
#        l3 = plt.plot(uu, label='sum')
#        plt.legend()
#        plt.show()
        
        #l1 = plt.plot(u-uu, label='SUNTANS mean')
        #l2 = plt.plot(u2-uu2, label='ROMS mean')
        l3 = plt.plot(u, label='SUNTANS')
        l4 = plt.plot(u2+uu, label='ROMS')
        plt.legend()
        plt.show()


        pdb.set_trace()        
        
        
    def time_average(self, uu, period):
        """
        average the velocity over a time period (tidal cycle)
        Input: 2D velocity and time period length (hours)
        Outputed: averaged velocity
        """        
        uu = self.align(uu, period)
        Nt_new = int(uu.shape[0]/period)
        
        if len(uu.shape) == 2:
            ## SUNTANS velocity ##
            uout = np.zeros((Nt_new, uu.shape[1]))
                        
            for i in range(self.Nc):
                utem = uu[:,i]  ## velocity time series at every grid point
                for t in range(Nt_new):
                    uout[t, i] = sum(utem[t*period:(t+1)*period])/period
        elif len(uu.shape) ==3:
            ## ROMS velocity ##
            uout = np.zeros((Nt_new, uu.shape[1], uu.shape[2]))
            for i in range(uu.shape[1]):
                for j in range(uu.shape[2]):
                    utem = uu[:,i,j]
                    for t in range(Nt_new):
                        uout[t,i,j] = sum(utem[t*period:(t+1)*period])/period
        else:
            raise IOError('Unknown velocity variable format!!!!')
        
        x1 = np.linspace(0, uu.shape[0], Nt_new)
        Ft = interp1d(x1, uout, axis=0)
        x2 = range(uu.shape[0])
        uout2 = Ft(x2)
        
        return uout2    
                    
            
    def align(self, var, period):
        """
        
        """
        
        Nt = var.shape[0]
        Nt_new = int(np.floor(Nt/period) * period) #new length
        if len(var.shape) == 2:
            ## SUNTANS velocity ##
            var_out = np.zeros((Nt_new, var.shape[1]))
            for i in range(Nt_new):
                var_out[i,:] = var[i,:]
        elif len(var.shape) == 3:
            ## ROMS velocity ##
            var_out = np.zeros((Nt_new, var.shape[1], var.shape[2]))
            for i in range(Nt_new):
                var_out[i,:,:] = var[i,:,:]
        else:
            ## other variables, like time, 1D ##
            #var_out = np.zeros((Nt_new))
            var_out = []
            for i in range(Nt_new):
                #var_out[i] = var[i]
                var_out.append(var[i])
        
        return var_out
        
    def findNearest(self,t,timevec):
        """
        Return the index from timevec the nearest time point to time, t. 
        
        """
        tnow = self.SecondsSince(t)
        tvec = self.SecondsSince(timevec)
        
        return np.searchsorted(tvec,tnow)[0]
        
    def SecondsSince(self,timein,basetime = datetime(1990,1,1)):
        """
        Converts a list or array of datetime object into an array of seconds since "basetime"
        
        Useful for interpolation and storing in netcdf format
        """
        timeout=[]
        try:
            timein = timein.tolist()
        except:
            timein = timein
    
        try:
            for t in timein:
                dt = t - basetime
                timeout.append(dt.total_seconds())
        except:
            dt = timein - basetime
            timeout.append(dt.total_seconds()) 
            
        return np.asarray(timeout)
        
    def findNearset(self,x,y,lon,lat):
        """
        Return the J,I indices of the nearst grid cell to x,y
        """
        
        dist = np.sqrt( (lon - x)**2 + (lat - y)**2)
        
        return np.argwhere(dist==dist.min())
        
        
    def shrink(self,a,b):
        """Return array shrunk to fit a specified shape by triming or averaging.
        
        a = shrink(array, shape)
        
        array is an numpy ndarray, and shape is a tuple (e.g., from
        array.shape). a is the input array shrunk such that its maximum
        dimensions are given by shape. If shape has more dimensions than
        array, the last dimensions of shape are fit.
        
        as, bs = shrink(a, b)
        
        If the second argument is also an array, both a and b are shrunk to
        the dimensions of each other. The input arrays must have the same
        number of dimensions, and the resulting arrays will have the same
        shape.
        
        This routine is part of Rob Hetland's OCTANT package:
            https://github.com/hetland/octant
            
        Example
        -------
        
        >>> shrink(rand(10, 10), (5, 9, 18)).shape
        (9, 10)
        >>> map(shape, shrink(rand(10, 10, 10), rand(5, 9, 18)))        
        [(5, 9, 10), (5, 9, 10)]   
        
        """

        if isinstance(b, np.ndarray):
            if not len(a.shape) == len(b.shape):
                raise Exception, \
                      'input arrays must have the same number of dimensions'
            a = self.shrink(a,b.shape)
            b = self.shrink(b,a.shape)
            return (a, b)

        if isinstance(b, int):
            b = (b,)

        if len(a.shape) == 1:                # 1D array is a special case
            dim = b[-1]
            while a.shape[0] > dim:          # only shrink a
                if (dim - a.shape[0]) >= 2:  # trim off edges evenly
                    a = a[1:-1]
                else:                        # or average adjacent cells
                    a = 0.5*(a[1:] + a[:-1])
        else:
            for dim_idx in range(-(len(a.shape)),0):
                dim = b[dim_idx]
                a = a.swapaxes(0,dim_idx)        # put working dim first
                while a.shape[0] > dim:          # only shrink a
                    if (a.shape[0] - dim) >= 2:  # trim off edges evenly
                        a = a[1:-1,:]
                    if (a.shape[0] - dim) == 1:  # or average adjacent cells
                        a = 0.5*(a[1:,:] + a[:-1,:])
                a = a.swapaxes(0,dim_idx)        # swap working dim back

        return a
        
    def flow_direction(self, u, v):
        """
        Input: u, v velocity //
        Output: flow direction
        """
        
        return np.arctan2(u, v) * 180. / np.pi % 360

#### For testing ####        
if __name__ == "__main__":
    starttime = '2009-06-01'
    endtime = '2009-06-28'    
    tide_average(starttime, endtime)

    
        