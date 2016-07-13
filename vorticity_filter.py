# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:07:21 2016

@author: dongyu
"""

import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import os
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import pdb
from vorticity_SUNTANS import vorticity as vor_suntans
from vorticity_ROMS import vorticity as vor_roms
from velocity import velocity as vel


class vorticity(object):
    """
    class for calculating, averaging vorticity from two models
    """
    
    def __init__(self, starttime, endtime, **kwargs):
        self.__dict__.update(kwargs)
        #### Specify the necessary ####
        self.start = starttime
        self.end = endtime
        
        self.starttime = datetime.strptime(starttime, '%Y-%m-%d')
        self.endtime = datetime.strptime(endtime, '%Y-%m-%d')
        self.wdr = os.getcwd()
        self.sunfile = self.wdr+'/SUNTANS_file/GalvCoarse_0000.nc'
        self.romsfile = self.wdr+'/download_ROMS/txla_subset_HIS.nc'
        
        nc = Dataset(self.sunfile, 'r')
        ftime = nc.variables['time']
        time = num2date(ftime[:],ftime.units)
        self.ind0 = self.findNearest(self.starttime,time)
        self.ind1 = self.findNearest(self.endtime,time)
        self.time = time[self.ind0:self.ind1+1]
        
        #### call the classes to calculated the vorticity
        ## Step 1) SUNTANS vorticity
        #self.vorticity_sun(starttime, endtime)
        ## Step 2) ROMS vorticity
        #self.vorticity_roms(starttime, endtime)
        
        #self.plot_filtered_diff()
        #self.plot_diff()
        

    def vorticity_sun(self):
        """
        calling SUNTANS vorticity class to calculate SUNTANS vorticity
        returned is the interpolated vorticity of SUNTANS
        """        
        vor = vor_suntans(self.start, self.end)
        
        vor.readFile(self.sunfile)
        
        w, lonss, latss, maskss = vor.interp(self.romsfile)
        
        
        #pdb.set_trace()
        return w, lonss, latss, maskss
        
    def vorticity_roms(self):
        """
        calling ROMS vorticity class to calculate ROMS vorticity
        returned is the subsetted ROMS vorticity
        """
        
        vor = vor_roms(self.start, self.end)
        
        vor.readFile(self.romsfile)
        
        w = vor.get_vorticity()
        lon = vor.data['lon_psi'][0:-1,0:-1]
        lat = vor.data['lat_psi'][0:-1,0:-1]
        mask = vor.data['mask'][0:-1,0:-1]
        w, lon, lat, mask = vor.subset(w, lon, lat, mask)
        
        #pdb.set_trace()        
        return w, lon, lat, mask
        
    def vorticity_diff(self):
        """
        calculate the normalized difference between SUNTANS and ROMS vorticity
        ROMS is transporting large scale vorticity that is missing in SUNTANS?
        SUNTANS is creating vorticity that ROMS doesn't see?
        """
                    
        w_sun, lonss, latss, maskss =  self.vorticity_sun()
        w_roms, lon, lat, mask = self.vorticity_roms()
        
        #w_diff = w_sun -  w_roms
        
        #pdb.set_trace()
        #w_nom = np.zeros_like(w_diff)
        #for t in range(w_nom.shape[0]):
        #    w_nom[t,maskss==1] = (w_sun[t,maskss==1] - w_roms[t,maskss==1]) / w_roms[t,maskss==1]
        
        #w_nom[np.abs(w_nom)>10] = 10
        #w_diff[:,maskss==0] = 0
        
        return w_sun-w_roms, lonss, latss, maskss
                
        
    def vorticity_filter(self):
        """
        do the 2D spatial average of the vorticity
        return the difference of vorticity
        """
        w_sun, lonss, latss, maskss =  self.vorticity_sun()
        w_roms, lon, lat, mask = self.vorticity_roms()
        
        
        #### do the average ####
        vor = vor_suntans(self.start, self.end)
        w_sun, lonss, latss = vor.average(w_sun, lonss, latss, maskss)
        w_roms, lon, lat = vor.average(w_roms, lon, lat, mask)
        
        #pdb.set_trace()
        
        return w_sun-w_roms, lonss, latss, maskss


    def plot_diff(self):
        """
        function for plotting the difference of unfiltered vorticity
        """        
        w_diff, lon, lat, mask = self.vorticity_diff()

        vor = vor_suntans(self.start, self.end) 
        vor.readFile(self.sunfile)
        mask = vor.obc_mask(lon, lat, mask)
        w_diff[:,mask==0] = 0
                       
        south = lat.min(); north =lat.max()
        west = lon.min(); east = lon.max()
        
        timeformat = '%Y%m%d-%H%M'
        for i in range(len(self.time)):
            fig = plt.figure(figsize=(10,8))
            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
                      llcrnrlon=west,urcrnrlon=east, resolution='h')
            
            basemap.drawcoastlines()
            basemap.fillcontinents(color='coral',lake_color='aqua')
            basemap.drawcountries()
            basemap.drawstates()  
            
            llons, llats=basemap(lon,lat)   
            con = basemap.pcolormesh(llons,llats,w_diff[i,:,:])
            con.set_clim(vmin=-0.0004, vmax=0.0004)
            cbar = plt.colorbar(con, orientation='vertical')
            cbar.set_label("vorticity")
            #plt.show()
            timestr = datetime.strftime(self.time[i], timeformat)
            plt.title('vorticity at %s'%timestr)
            plt.savefig(self.wdr+'/vorticity_figure/vorticity_diff_unfiltered/'+str(i)+'.png')
            print "Saving figure %s to vorticity figure directory"%str(i)

        
    def plot_filtered_diff(self):
        """
        function for plotting the difference of filtered vorticity
        """
        
        w_diff, lon, lat, mask = self.vorticity_filter()
        
        south = lat.min(); north =lat.max()
        west = lon.min(); east = lon.max()
        
        timeformat = '%Y%m%d-%H%M'
        for i in range(len(self.time)):
            fig = plt.figure(figsize=(10,8))
            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
                      llcrnrlon=west,urcrnrlon=east, resolution='h')
            
            basemap.drawcoastlines()
            basemap.fillcontinents(color='coral',lake_color='aqua')
            basemap.drawcountries()
            basemap.drawstates()  
            
            llons, llats=basemap(lon,lat)   
            con = basemap.pcolormesh(llons,llats,w_diff[i,:,:])
            #con.set_clim(vmin=-0.0003, vmax=0.0003)
            cbar = plt.colorbar(con, orientation='vertical')
            cbar.set_label("vorticity")
            #plt.show()
            timestr = datetime.strftime(self.time[i], timeformat)
            plt.title('vorticity at %s'%timestr)
            plt.savefig(self.wdr+'/vorticity_figure/vorticity_diff/'+str(i)+'.png')
            print "Saving figure %s to ROMS figure directory"%str(i)
        
        
    def findNearest(self,t,timevec):
        """
        Return the index from timevec the nearest time point to time, t. 
        
        """
        tnow = self.SecondsSince(t)
        tvec = self.SecondsSince(timevec)
        
        #tdist = np.abs(tnow - tvec)
        
        #idx = np.argwhere(tdist == tdist.min())
        
        #return int(idx[0])
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
        



        
#### For testing ####        
if __name__ == "__main__":
    starttime = '2014-03-22'
    endtime = '2014-03-27'
    vorticity(starttime, endtime)