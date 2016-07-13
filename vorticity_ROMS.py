# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:17:31 2016

@author: dongyu
"""

import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import os
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import utm
import math
import pdb

class vorticity(object):
    """
    general class for calculating vorticity from surface velcoity field
    """
    def __init__(self, starttime, endtime, **kwargs):
        self.__dict__.update(kwargs)

        #### Specify the starttime and endtime of the vorticity ####        
        self.starttime = datetime.strptime(starttime, '%Y-%m-%d')
        self.endtime = datetime.strptime(endtime, '%Y-%m-%d')
        
        #### Read data from model output
        wdr = os.getcwd()
        romsfile = wdr+'/download_ROMS/txla_subset_HIS.nc'
        #self.readFile(romsfile)
        #self.plot()
        ##self.get_vorticity()
        
        
        
    def readFile(self, filename):
        """
        function that reads from netcdf file
        """
        nc = Dataset(filename, 'r')
        print "#### Reading ROMS output file !!!! ####\n"
        #print nc
        self.data = dict()
        x1 = 0; x2 = nc.variables['lon_psi'].shape[1]        
        y1 = 0; y2 = nc.variables['lon_psi'].shape[0]
        lon = nc.variables['lon_psi'][:]
        lat = nc.variables['lat_psi'][:]
        x = np.zeros_like(lon)
        y = np.zeros_like(lat)
        nx, ny = lon.shape    
        for i in range(nx):
            for j in range(ny):
                (y[i,j], x[i,j]) = utm.from_latlon(lat[i,j], lon[i,j])[0:2]
        
        self.data['lon'] = x
        self.data['lat'] = y
        self.data['lon_psi'] = lon
        self.data['lat_psi'] = lat
        u = nc.variables['u'][:,0,:,:]
        v = nc.variables['v'][:,0,:,:]
        self.data['mask'] = nc.variables['mask_psi'][:]
        
        ftime = nc.variables['ocean_time']
        time = num2date(ftime[:],ftime.units)
        self.ind0 = self.findNearest(self.starttime,time)
        self.ind1 = self.findNearest(self.endtime,time)
        self.data['time'] = time[self.ind0:self.ind1+1]
        
        self.data['u'] = u[self.ind0:self.ind1+1,y1:y2+1,x1:x2]  
        self.data['v'] = v[self.ind0:self.ind1+1,y1:y2,x1:x2+1] 
        
        #ftime = nc.variables['ocean_time']
        #self.data['time'] = num2date(ftime[:],ftime.units)
        #pdb.set_trace()
        
        
    def get_vorticity(self):
        """
        This function calculates the vorticity
        """
        T,y,x = self.data['u'].shape
        
        dx = np.diff(self.data['lon'])
        dV = np.diff(self.data['v'], axis=2)
        dV_dx = np.zeros_like(dV)
        for i in range(dV.shape[0]):
            dV_dx[i,:,:] = dV[i,:,:]/dx
        
        dy = np.diff(self.data['lat'], axis=0)
        dU = np.diff(self.data['u'], axis=1)      
        dU_dy = np.zeros_like(dU)
        for i in range(dU.shape[0]):
            dU_dy[i,:,:] = dU[i,:,:]/dy
            
        #### average the matrix ####
        dV_dx2 = np.zeros((T,y-1,x-1))
        for tt in range(T):
            for j in range(x-1):
                for i in range(y-1):
                    dV_dx2[tt,i,j] = (dV_dx[tt,i,j]+dV_dx[tt,i+1,j])
                    
        dU_dy2 = np.zeros((T,y-1,x-1))
        for tt in range(T):
            for i in range(y-1):
                for j in range(x-1):
                    dU_dy2[tt,i,j] = (dU_dy[tt,i,j]+dU_dy[tt,i,j+1])
        
        mask = self.data['mask'][0:-1,0:-1]
        #### vorticity w = dV/dx - dU/dy ####
        w =  dV_dx2 - dU_dy2
        
        w[np.abs(w)>5] = 0
        w[:,mask==0] = np.nan
        #### Note this has to be 0, not np.nan
        
#        plt.figure(1)
#        plt.pcolor(w[12,:,:])
#        plt.xlim((0,w.shape[2]))
#        plt.ylim((0,w.shape[1]))
#        plt.show()
#        pdb.set_trace()
        return w
        
    def plot(self):
        """
        this funtion plot the contour of vorticity on the basemap
        """
        T,y,x = self.data['u'].shape
        time = self.data['time']
        timeformat = '%Y%m%d-%H%M'

        wdr = os.getcwd()        
        
        w = self.get_vorticity()
        lon = self.data['lon_psi'][0:-1,0:-1]
        lat = self.data['lat_psi'][0:-1,0:-1]
        
        mask = self.data['mask'][0:-1,0:-1]
        
        
        #### do the 2D spatial average ####        
        ## Step 1) Subset SUNTANS area (This step is unnecessary)
        w, lon, lat, mask = self.subset(w, lon, lat, mask)
        south = lat.min(); north =lat.max()
        west = lon.min(); east = lon.max()
        
        ## Step 2) do the average                
        w, lon, lat = self.average(w, lon, lat, mask)
        pdb.set_trace()
        

        #### Full ROMS area ####
        #south = lat.min(); north =lat.max()
        #west = lon.min(); east = lon.max()
        
        #### Zoomed area ####
        #south = 28.7820; north = 29.7889
        #west = -95.2831; east = -94.3699
        
        
        for i in range(len(time)):
            fig = plt.figure(figsize=(10,8))
            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
                      llcrnrlon=west,urcrnrlon=east, resolution='h')

            basemap.drawcoastlines()
            basemap.fillcontinents(color='coral',lake_color='aqua')
            basemap.drawcountries()
            basemap.drawstates()  
            
            llons, llats=basemap(lon,lat)   
            con = basemap.contourf(llons,llats,w[i,:,:])
            con.set_clim(vmin=-0.0003, vmax=0.0003)
            cbar = plt.colorbar(con, orientation='vertical')
            cbar.set_label("vorticity")
            #plt.show()
            timestr = datetime.strftime(time[i], timeformat)
            plt.title('vorticity at %s'%timestr)
            plt.savefig(wdr+'/vorticity_figure/ROMS_figure_average2/'+str(i)+'.png')
            print "Saving figure %s to ROMS figure directory"%str(i)
            
    def average(self, vor, lon, lat, mask):
        """
        do the spatial average for vorticity
        """
        from scipy import interpolate        
        
        vor[:,mask==0] = 0
        T, y, x = vor.shape
        #### determine the spatial interval length ####
        yy = 5; xx = 4
        #### Number of points in an interval ####
        yp = yy+1; xp = xx+1
        print "The spatial interval for latitude is %s \n"%yy
        print "The spatial interval for longitude is %s \n"%xx
        
        (xd, yd) = self.distance(lon,lat)
        xd = xd * xx
        yd = yd * yy
        #### Calculate the distance of the new scale ####
        print "The distance in x- direction is %s km and in y- direction is %s km\n"%(xd, yd)        
        print "The 2D averaged area is %s km^2"%(xd*yd)

        #### Specify the new matrix for storing the averaged vorticity ####
        y1, x1 = (int(np.ceil((y-1)/float(yy))), int(np.ceil((x-1)/float(xx))))
        y2, x2 = (int(np.floor((y-1)/float(yy))), int(np.floor((x-1)/float(xx))))
        new_w1 = np.zeros((T, y, x1))
        new_w2 = np.zeros((T, y1, x1))
        #### do the 2D spatial average ####
        ## x- direction     
        if x2 != x1:
            for tt in range(T):
                for i in range(y):
                    new_w1[tt,i,0] = sum(vor[tt,i,0:0+xp])
                    for j in range(1,x2):
                        new_w1[tt,i,j] = sum(vor[tt,i,j*xp-j:j*xp+xp-j])
                    new_w1[tt,i,-1] = sum(vor[tt, i, j*xp+xp-j-1:-1])
                    #0:4 3:7 6:10 9:13 
                    #pdb.set_trace()
        else:
            for tt in range(T):
                for i in range(y):
                    new_w1[tt,i,0] = sum(vor[tt,i,0:0+xp])
                    for j in range(1,x2):
                        new_w1[tt,i,j] = sum(vor[tt,i,j*xp-j:j*xp+xp-j])
                        #pdb.set_trace()
        ## y- direction
        if y2 != y1:
            for tt in range(T):
                for j in range(x1):
                    new_w2[tt,0,j] = sum(new_w1[tt,0:0+yp,j])
                    for i in range(1,y2):
                        new_w2[tt,i,j] = sum(new_w1[tt,i*yp-i:i*yp+yp-i,j])
                        #pdb.set_trace()
                    new_w2[tt,-1,j] = sum(new_w1[tt,i*yp+yp-i-1:-1,j])
        else:
            for tt in range(T):
                for j in range(x1):
                    new_w2[tt,0,j] = sum(new_w1[tt,0:0+yp,j])
                    for i in range(1,y2):
                        new_w2[tt,i,j] = sum(new_w1[tt,i*yp-i:i*yp+yp-i,j])
                        
        new_w2 = new_w2/(float(yp)*float(xp))
        #### Specify the new longitude and latitude ####
        #lon = self.data['lon_psi'][0:-1,0:-1]
        #lat = self.data['lat_psi'][0:-1,0:-1]  
        new_lon1 = np.zeros((y,x1))
        new_lat1 = np.zeros((y,x1))
        new_lon2 = np.zeros((y1,x1))
        new_lat2 = np.zeros((y1,x1))
        xold = np.arange(0,x)
        xnew = np.linspace(0,x-1,num=x1)
        yold = np.arange(0,y)
        ynew = np.linspace(0,y-1,num=y1)
        for i in range(y):
            f1 = interpolate.interp1d(xold, lon[i,:])
            new_lon1[i,:] = f1(xnew)
            f2 = interpolate.interp1d(xold, lat[i,:])
            new_lat1[i,:] = f2(xnew)
        for j in range(x1):
            f1 = interpolate.interp1d(yold, new_lon1[:,j])
            new_lon2[:,j] = f1(ynew)
            f2 = interpolate.interp1d(yold, new_lat1[:,j])
            new_lat2[:,j] = f2(ynew)
	
        #pdb.set_trace()
        
        return new_w2, new_lon2, new_lat2
                    
        
    def distance(self, lon, lat):
        """
        calculate the distance (The scale) of the grid
        """
        #### In the x- direction ####
        xlon1 = lon[0,0]; xlon2 = lon[0,1]
        xlat1 = lat[0,0]; xlat2 = lat[0,1]
        
        #### In the y- direction ####
        ylon1 = lon[0,0]; ylon2 = lon[1,0]
        ylat1 = lat[0,0]; ylat2 = lat[1,0]        
        
        def distance_on_unit_sphere(lat1, long1, lat2, long2):
            degrees_to_radians = math.pi/180.0
            phi1 = (90.0 - lat1)*degrees_to_radians
            phi2 = (90.0 - lat2)*degrees_to_radians
 
            theta1 = long1*degrees_to_radians
            theta2 = long2*degrees_to_radians
 
            cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
            math.cos(phi1)*math.cos(phi2))
            arc = math.acos( cos )
 
            return arc*6373  ##The unit is kilometer
        
        return (distance_on_unit_sphere(xlat1, xlon1, xlat2, xlon2), distance_on_unit_sphere(ylat1, ylon1, ylat2, ylon2) )
        
    def subset(self, w_in, lon_in, lat_in, mask_in):
        """
        This function is used to subset the ROMS output 
        Input: the index is hard coded from SUNTANS file
        """
        J0 = 58; I0 = 16
        J1 = 122; I1 = 57
        
        w_out = w_in[:,J0:J1,I0:I1]
        lon_out = lon_in[J0:J1,I0:I1]
        lat_out = lat_in[J0:J1,I0:I1]
        mask_out = mask_in[J0:J1,I0:I1]
        
        return w_out, lon_out, lat_out, mask_out
        
        

    
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
    vor = vorticity(starttime, endtime)
    wdr = os.getcwd()
    romsfile = wdr+'/download_ROMS/txla_subset_HIS.nc'
    vor.readFile(romsfile)
    vor.plot()
        
        
