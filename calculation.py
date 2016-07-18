# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:06:26 2016

@author: dongyu
"""

import numpy as np
from netCDF4 import Dataset, num2date
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import utm
from interpXYZ import interpXYZ
from vorticity_filter import vorticity as vor_filter
from vorticity_SUNTANS import vorticity as vor_suntans
import pdb


class cal_vorticity(object):
    """
    class for calculating sub-filtered vorticity from two models
    """
    nx = 90
    ny = 30    
    
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
        self.sunfile = self.wdr+'/SUNTANS_file/GalvCoarse_0000.nc'
        self.romsfile = self.wdr+'/download_ROMS/txla_subset_HIS.nc'
        
        ## test ##
        #self.grid_plot()
        self.readFile(self.romsfile)
        #self.stress()
        #self.calc()
        self.calc_uv()
        
    def readFile(self, roms_filename):
        """
        function that reads from ROMS netcdf file
        """
        #### Read ROMS file ####
        nc = Dataset(roms_filename, 'r')
        print "#### Reading ROMS output file !!!! ####\n"
        #print nc     
        
        #lon = nc.variables['lon_rho'][:]
        #lat = nc.variables['lat_rho'][:]        
        u = nc.variables['u'][:,:,:,:]   # velocity at all levels
        v = nc.variables['v'][:,:,:,:]
        ftime = nc.variables['ocean_time']
        time = num2date(ftime[:],ftime.units)
        self.ind0 = self.findNearest(self.starttime,time)
        self.ind1 = self.findNearest(self.endtime,time)
                        
        self.data_roms = dict()
        self.data_roms['time'] = time[self.ind0:self.ind1+1]
        self.data_roms['lon_rho'] = nc.variables['lon_rho'][:]
        self.data_roms['lat_rho'] = nc.variables['lat_rho'][:]      
        self.data_roms['u'] = u[self.ind0:self.ind1+1,:,:,0:-2] # surface velocity
        self.data_roms['v'] = v[self.ind0:self.ind1+1,:,0:-2,:]
        self.data_roms['mask'] = nc.variables['mask_rho'][:]
        self.data_roms['angle'] = nc.variables['angle'][:]
        self.data_roms['h'] = nc.variables['h'][:]

        self.data_roms['lon_psi'] = nc.variables['lon_psi'][:][0:-1,0:-1]
        self.data_roms['lat_psi'] = nc.variables['lat_psi'][:][0:-1,0:-1]
        self.data_roms['mask_psi'] = nc.variables['mask_psi'][:][0:-1,0:-1]
        
        
    def rectilinear(self, nx, ny):
        """
        function that generates a rectilinear grid
        """
        #### Step 1) define a regular grid ####
        #NW = (29.017842, -95.174746)
        NW = (29.036487, -95.131658)
        #NE = (29.459124, -94.413252)
        NE = (29.439981, -94.465266)
        #SW = (28.777523, -94.979306)
        SW = (28.807300, -95.005893)
        
        #### Calculate the length in x- and y- direction ####
        Lx = self.distance_on_unit_sphere(NW[0], NW[1], NE[0], NE[1])
        Ly = self.distance_on_unit_sphere(NW[0], NW[1], SW[0], SW[1])
        
        new_NW = utm.from_latlon(NW[0], NW[1])[0:2]
        #new_SW = (new_NW[0]-Ly, new_NW[1])
        #new_NE = (new_NW[0], new_NW[1]+Lx)
        
        y = np.linspace(new_NW[1]-Ly, new_NW[1], ny)
        x = np.linspace(new_NW[0], new_NW[0]+Lx, nx)
        
        xv, yv = np.meshgrid(x, y)        
        #origin = (xv[0,-1], yv[0,-1])
        origin = new_NW
        
        tem_xv = xv - origin[0]
        tem_yv = yv - origin[1]
        
        #### Step 2) rotate the grid from an angle ####                    
        def rotate(yv, xv, theta):
            """Rotates the given polygon which consists of corners represented as (x,y),
            around the ORIGIN, clock-wise, theta degrees"""
            theta = math.radians(theta)
            out_yv = np.zeros_like(yv)
            out_xv = np.zeros_like(xv)
            (nx, ny) = xv.shape
            for i in range(nx):
                for j in range(ny):
                    out_yv[i,j] = yv[i,j]*math.cos(theta)-xv[i,j]*math.sin(theta)
                    out_xv[i,j] = yv[i,j]*math.sin(theta)+xv[i,j]*math.cos(theta)
            
            return out_yv, out_xv
            
        tem_yv, tem_xv = rotate(tem_yv, tem_xv, -35)
        
        new_xv = tem_xv + origin[0]   #lon 
        new_yv = tem_yv + origin[1]   #lat
        
        lon = np.zeros_like(new_xv)
        lat = np.zeros_like(new_yv)
        #pdb.set_trace()
        for i in range(ny):
            for j in range(nx):
                (lat[i,j], lon[i,j]) = utm.to_latlon(new_xv[i,j], new_yv[i,j],15,'U')[0:2]
        
        dx = Lx / nx
        dy = Ly / ny                  
        return lon, lat, dx, dy   
        
    def interp_roms_uv(self, new_lon, new_lat):
        """
        function that interpolate the ROMS velocity
        Input: new grid
        Output: interpolated velocity field
        """        
        
        #### subset ROMS grid for interpolation ####
                 
        #### Step 1) Prepare x, y coordinate and velocity to do the interpolation ####
        time = self.data_roms['time']
        lon = self.data_roms['lon_rho'][1:-2,1:-2]
        lat = self.data_roms['lat_rho'][1:-2,1:-2]
        mask = self.data_roms['mask'][1:-2,1:-2]
        u = self.data_roms['u']
        v = self.data_roms['v']
        ang = self.data_roms['angle'][1:-2,1:-2]
        Nk = u.shape[1]

        # average u,v to central rho points           
        uroms = np.zeros((len(time), Nk, mask.shape[0], mask.shape[1]))
        vroms = np.zeros((len(time), Nk, mask.shape[0], mask.shape[1]))
        for t in range(len(time)):
            for k in range(Nk):
                uroms[t,k,:,:] = self.shrink(u[t,k,:,:], mask.shape)
                vroms[t,k,:,:] = self.shrink(v[t,k,:,:], mask.shape)
        
        #### adjust velocity direction ####        
        uroms, vroms = self.rot2d(uroms, vroms, ang)
        
        #### Step 2) subset ROMS grid for interpolation ####               
        SW=(new_lat.min(), new_lon.min())  ###(lat, lon)
        NE=(new_lat.max(), new_lon.max())
        
        ind = self.findNearset(SW[1], SW[0], lon, lat)
        J0=ind[0][0] 
        I0=ind[0][1] 
        
        ind = self.findNearset(NE[1], NE[0], lon, lat)
        J1=ind[0][0] 
        I1=ind[0][1] 
        
        xroms, yroms = self.convert_utm(lon, lat)  #### convert to utm for interpolation
        yss = yroms[J0:J1,I0:I1]  ##subset x,y
        xss = xroms[J0:J1,I0:I1]
        maskss = mask[J0:J1,I0:I1]
        u = uroms[:,:,J0:J1,I0:I1]
        v = vroms[:,:,J0:J1,I0:I1]
        
        #### Step 3) Prepare the grid variables for the interpolation class ####
        xy_roms = np.vstack((xss[maskss==1],yss[maskss==1])).T        

        xnew, ynew = self.convert_utm(new_lon, new_lat)
        xy_new = np.vstack((xnew.ravel(),ynew.ravel())).T
        Fuv = interpXYZ(xy_roms, xy_new)
        
        X, Y = xnew.shape
        uout = np.zeros((len(time),Nk, X, Y))
        vout = np.zeros((len(time),Nk, X, Y))
        
        print "interpolating ROMS U, V velocity onto the new rectilinear grid!!! \n"
        #### Loop through time to do the interpolation ####
        for tstep in range(len(time)):
            for k in range(Nk):
                utem = Fuv(u[tstep,k,:,:][maskss==1].flatten()) 
                vtem = Fuv(v[tstep,k,:,:][maskss==1].flatten())
                uout[tstep,k,:,:] = utem.reshape(X,Y)
                vout[tstep,k,:,:] = vtem.reshape(X,Y)
            
        #pdb.set_trace()
        
        return uout[:,:,:,:], vout[:,:,:,:]


    def vorticity_filtered(self, new_lon, new_lat):
        """
        This function calls the vorticity class that calculates the filtered vorticity
        interpolate the filtered vorticity into the original ROMS grid
        interpolate the filtered vorticity into the new rectilinear grid
        Input: new grid
        Output: interpolated filtered vorticity
        """
        
        #### Step 1) call vorticity class ####        
        vor = vor_filter(self.start, self.end)
        w_filter, lon_avg, lat_avg, mask_avg = vor.vorticity_filter()  ##Note: the mask_avg is not the same shape as other variables

        #### Step 2) interpolate the filtered vorticity into the original ROMS grid ####
        time = self.data_roms['time']
        
        ## Note the lon, lat and mask should be the subsetted variables
        vor_sun = vor_suntans(self.start, self.end) 
        vor_sun.readFile(self.sunfile)
        w_sun, lonss, latss, maskss =  vor.vorticity_sun()
        #lon = self.data_roms['lon_rho'][1:-2,1:-2]
        #lat = self.data_roms['lat_rho'][1:-2,1:-2]
        #mask = self.data_roms['mask'][1:-2,1:-2]     
        
        xroms, yroms = self.convert_utm(lonss, latss)
        xy_roms = np.vstack((xroms[maskss==1], yroms[maskss==1])).T
        
        
        x_avg, y_avg = self.convert_utm(lon_avg, lat_avg)
        xy_avg = np.vstack((x_avg.flatten(), y_avg.flatten())).T
        Favg = interpXYZ(xy_avg, xy_roms,  method='idw')
        w_out = np.zeros((len(time), lonss.shape[0], lonss.shape[1]))        
        
        for tstep in range(len(time)):
            w_tem = Favg(w_filter[tstep,:,:].flatten())
            w_out[tstep,:,:][maskss==1] = w_tem  
        
        w_out[np.isnan(w_out)] = 0.   ## some nan value comes from interpolation
        #### mask the unuseful part of vorticity ####            
        maskss = vor_sun.obc_mask(lonss, latss, maskss)
        w_out[:,maskss==0] = 0.
        
        
        #### Step 3) interpolate the filtered vorticity into the new rectilinear grid ####
        SW=(new_lat.min(), new_lon.min())  ###(lat, lon)
        NE=(new_lat.max(), new_lon.max())
        
        ind = self.findNearset(SW[1], SW[0], lonss, latss)
        J0=ind[0][0] 
        I0=ind[0][1] 
        
        ind = self.findNearset(NE[1], NE[0], lonss, latss)
        J1=ind[0][0] 
        I1=ind[0][1] 
        
        yss = yroms[J0:J1,I0:I1]  ##subset x,y
        xss = xroms[J0:J1,I0:I1]
        maskss = maskss[J0:J1,I0:I1]
        wss = w_out[:,J0:J1,I0:I1]
        
        xy_roms = np.vstack((xss[maskss==1],yss[maskss==1])).T        
        
        xnew, ynew = self.convert_utm(new_lon, new_lat)
        xy_new = np.vstack((xnew.ravel(),ynew.ravel())).T
        Fw = interpXYZ(xy_roms, xy_new)
        
        X, Y = xnew.shape
        wnew = np.zeros((len(time), X, Y))
        
        print "interpolating filtered vorticity onto the new rectilinear grid!!! \n"
        #### Loop through time to do the interpolation ####
        for tstep in range(len(time)):
            wtem = Fw(wss[tstep,:,:][maskss==1].flatten()) 
            wnew[tstep,:,:] = wtem.reshape(X,Y)

        return wnew

        
    def stress(self):
        """
        function that calculates the stress input for the filtered vorticity
        tau_x_avg = avg(tau_sx - tau_bx) 
        tau_y_avg = avg(tau_sy - tau_by)
        tau_x_avg, tau_y_avg: filtered stress input
        tau_sx, tau_sy: surface stress
        tau_bx, tau_by: bottom stress
        The calculation performs through the steps below:
            Step 1) bottom stress calculated from the bottom friction step
            Step 2) surface stress read from SUNTANS output
            Step 3) tau_s - tau_b, then do the average, interpolate the averaged value to the new grid
            Step 4) Calculate the differentiation of the resulting term
            
        Output: stress on the new grid, water depth on the new grid
        """        
        #### Step 1) bottom stress ####
        vor_sun = vor_suntans(self.start, self.end) 
        vor_sun.readFile(self.sunfile)
        uc = vor_sun.u  ## Note: these velocities are timely subsetted already
        vc = vor_sun.v
        Nt = vor_sun.Nt
        Nc = vor_sun.Nc
        
        H = vor_sun.dv  ## water depth
        ## Some coefficients
        R = 10. #m: hydraulic radius
        n = 0.05 # Manning coefficient
        Cc = pow(R, 1/6.)/n # Chezy coefficient
        g = 9.81 # gravitational acceleration
                
        tau_bx = np.zeros((Nt, Nc)) ## bottom stress
        tau_by = np.zeros((Nt, Nc))
        for tstep in range(Nt):
            tau_bx[tstep,:] = g/pow(Cc, 2.)/pow(H[:], 2.) *  \
                            np.sqrt(pow(uc[tstep,:], 2.)+ pow(vc[tstep,:], 2.))*uc[tstep,:]
            tau_by[tstep,:] = g/pow(Cc, 2.)/pow(H[:], 2.) *  \
                            np.sqrt(pow(uc[tstep,:], 2.)+ pow(vc[tstep,:], 2.))*vc[tstep,:]
        
        #### Step 2) surface stress ####
        tau_sx = vor_sun.tau_x
        tau_sy = vor_sun.tau_y
        
        
        #### Step 3) tau_s - tau_b and average ####
        tau_x = tau_sx - tau_bx
        tau_y = tau_sy - tau_by
        ## interpolate to ROMS grid and average ##
        tau_avg_x, tau_avg_y, h = self.avg_stress(tau_x, tau_y)
        
        tau_h_x = np.zeros_like(tau_avg_x) ## tau/h
        tau_h_y = np.zeros_like(tau_avg_y)
        for tstep in range(Nt):
            tau_h_x[tstep,:,:] = tau_avg_x[tstep,:,:]/h[:,:]
            tau_h_y[tstep,:,:] = tau_avg_y[tstep,:,:]/h[:,:]
        #### Step 4) Calculate the differentiation of the resulting term ####
        new_lon, new_lat, dx, dy = self.rectilinear(self.nx,self.ny)
        xnew, ynew = self.convert_utm(new_lon, new_lat)
        
        dx = np.diff(xnew)
        dtau_y = np.diff(tau_h_y, axis=2)
        dtau_y_x = np.zeros_like(dtau_y)  ## d tau_y / dx      
        for i in range(dtau_y.shape[0]):
            dtau_y_x[i,:,:] = dtau_y[i,:,:]/dx
        
        dy = np.diff(ynew, axis=0)
        dtau_x = np.diff(tau_h_x, axis=1)
        dtau_x_y = np.zeros_like(dtau_x)  ## d tau_x / dy
        for i in range(dtau_x.shape[0]):
            dtau_x_y[i,:,:] = dtau_x[i,:,:]/dy
        
        #### average the matrix ####
        dtau_y_x2 = np.zeros((Nt, dtau_y_x.shape[1]-1, dtau_y_x.shape[2]))
        dtau_x_y2 = np.zeros((Nt, dtau_x_y.shape[1], dtau_x_y.shape[2]-1))
          
        for tstep in range(Nt):
            dtau_y_x2[tstep,:,:] = self.shrink(dtau_y_x[tstep,:,:], (dtau_y_x.shape[1]-1, dtau_y_x.shape[2]))          
            dtau_x_y2[tstep,:,:] = self.shrink(dtau_x_y[tstep,:,:], (dtau_y_x.shape[1]-1, dtau_y_x.shape[2]))             
    
        stress = np.zeros((Nt, xnew.shape[0], xnew.shape[1]))
        for tstep in range(Nt):
            stress[tstep,1:,1:] = dtau_x_y2[tstep,:,:] - dtau_y_x2[tstep,:,:]        

        return stress
        
        
    
    def avg_stress(self, tau_x, tau_y):
        """
        funtion that does the average and interpolation of the stress
        """
        vor_sun = vor_suntans(self.start, self.end) 
        vor_sun.readFile(self.sunfile)
        xv = vor_sun.xv
        yv = vor_sun.yv  
        Nt = vor_sun.Nt
        
        lon = self.data_roms['lon_psi'] 
        lat = self.data_roms['lat_psi'] 
        mask = self.data_roms['mask_psi']
        
        #### Step 1) Prepare x, y coordinate to do the interpolation ####
        xroms = np.zeros_like(lon)
        yroms = np.zeros_like(lat)
        (y,x) = lon.shape
        for i in range(y):
            for j in range(x):
                (yroms[i,j],xroms[i,j])=utm.from_latlon(lat[i,j],lon[i,j])[0:2]
        #### Step 2) subset ROMS grid for interpolation ####
        SW=utm.to_latlon(xv.min(),yv.min(),15,'R')  ###(lat, lon)
        NE=utm.to_latlon(xv.max(),yv.max(),15,'R')
        ind = self.findNearset(SW[1], SW[0], lon, lat)
        J0=ind[0][0] - 15
        I0=ind[0][1] + 5
        
        ind = self.findNearset(NE[1], NE[0], lon, lat)
        J1=ind[0][0] + 5
        I1=ind[0][1] - 6
        
        yss = yroms[J0:J1,I0:I1]  ##subset x,y
        xss = xroms[J0:J1,I0:I1]
        maskss = mask[J0:J1,I0:I1]
        
        #### Step 3) Prepare the grid variables for the interpolation class ####
        xy_sun = np.vstack((yv.ravel(),xv.ravel())).T   ## SUNTANS grid, xi: latitude, yi: longitude
        xy_new = np.vstack((xss[maskss==1],yss[maskss==1])).T   ## subset ROMS grid
        Fs = interpXYZ(xy_sun,xy_new, method='idw')        

        (Y, X) = xss.shape
        #### Initialize the output array @ subset roms grid ####
        tau_out_x = np.zeros((Nt, Y, X))
        tau_out_y = np.zeros((Nt, Y, X))
        #### Step 4) Loop througth time to do the interpolation ####            
        for tstep in range(Nt):
            tau_tem_x = Fs(tau_x[tstep,:])
            tau_tem_y = Fs(tau_y[tstep,:])
            tau_out_x[tstep, maskss==1] = tau_tem_x
            tau_out_y[tstep, maskss==1] = tau_tem_y
            
        #### Step 5) do the 2D spatial average for stress ####
        lonss = lon[J0:J1,I0:I1] ## subset lon, lat
        latss = lat[J0:J1,I0:I1]
        tau_avg_x, lon_avg, lat_avg = vor_sun.average(tau_out_x, lonss, latss, maskss)
        tau_avg_y, lon_avg, lat_avg = vor_sun.average(tau_out_y, lonss, latss, maskss)
        
        #### Step 6) interpolate the averaged value to the original roms grid ####
        x_avg, y_avg = self.convert_utm(lon_avg, lat_avg)
        xy_avg = np.vstack((x_avg.flatten(), y_avg.flatten())).T
        Favg = interpXYZ(xy_avg, xy_new,  method='idw')
            
        tau_avg_x2 = np.zeros((Nt, Y, X))
        tau_avg_y2 = np.zeros((Nt, Y, X))
        
        for tstep in range(Nt):
            tau_tem_x2 = Favg(tau_avg_x[tstep,:,:].flatten())
            tau_tem_y2 = Favg(tau_avg_y[tstep,:,:].flatten())
            tau_avg_x2[tstep,:,:][maskss==1] = tau_tem_x2
            tau_avg_y2[tstep,:,:][maskss==1] = tau_tem_y2
        
        maskss = vor_sun.obc_mask(lonss, latss, maskss)
        tau_avg_x2[:,maskss==0] = 0.
        tau_avg_y2[:,maskss==0] = 0.
        
        #### Step 7) interpolate the averaged value from ROMS grid to the new rectilinear grid ####
        new_lon, new_lat, dx, dy = self.rectilinear(self.nx,self.ny)
        new_SW=(new_lat.min(), new_lon.min())  ###(lat, lon)
        new_NE=(new_lat.max(), new_lon.max())        

        ind = self.findNearset(new_SW[1], new_SW[0], lonss, latss)
        J0=ind[0][0] 
        I0=ind[0][1] 
        
        ind = self.findNearset(new_NE[1], new_NE[0], lonss, latss)
        J1=ind[0][0] 
        I1=ind[0][1]  

        new_yss = yss[J0:J1,I0:I1]  ##subset x,y from the subsetted ROMS grid to new grid
        new_xss = xss[J0:J1,I0:I1]          
        new_maskss = maskss[J0:J1,I0:I1]
        
        tau_x_ss = tau_avg_x2[:,J0:J1,I0:I1]
        tau_y_ss = tau_avg_y2[:,J0:J1,I0:I1]
        
        xy_roms_ss = np.vstack((new_xss[new_maskss==1],new_yss[new_maskss==1])).T
        
        xnew, ynew = self.convert_utm(new_lon, new_lat)
        xy_new = np.vstack((xnew.ravel(),ynew.ravel())).T
        Fnew = interpXYZ(xy_roms_ss, xy_new)
        
        tau_x_new = np.zeros((Nt, xnew.shape[0], xnew.shape[1]))
        tau_y_new = np.zeros((Nt, xnew.shape[0], xnew.shape[1]))
        
        
        print "interpolating averaged stress onto the new rectilinear grid!!! \n"
        #### Loop through time to do the interpolation ####
        for tstep in range(Nt):
            tau_tem_x3 = Fnew(tau_x_ss[tstep,:,:][new_maskss==1].flatten()) 
            tau_tem_y3 = Fnew(tau_y_ss[tstep,:,:][new_maskss==1].flatten())
            tau_x_new[tstep,:,:] = tau_tem_x3.reshape(xnew.shape[0],xnew.shape[1])
            tau_y_new[tstep,:,:] = tau_tem_y3.reshape(xnew.shape[0],xnew.shape[1])
                
#        ######################################################################
#        #### These are just for testing if the interpolation is correct   ####
#        #### commented if not using                                       #### 
#        west=-95.42; east=-93.94
#        south=28.39;  north=29.90               
#        fig = plt.figure(figsize=(10,8))
#        basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
#                    llcrnrlon=west,urcrnrlon=east, resolution='h')
#            
#        basemap.drawcoastlines()
#        basemap.fillcontinents(color='coral',lake_color='aqua')
#        basemap.drawcountries()
#        basemap.drawstates()  
#        
#        llons, llats=basemap(lonss,latss)   
#        con = basemap.pcolormesh(llons,llats,tau_avg_x2[-1,:,:])
#        #con.set_clim(vmin=-0.08, vmax=0.03)
#        cbar = plt.colorbar(con, orientation='vertical')
#        cbar.set_label("stress")
#        plt.show() 
#        ######################################################################

        hout = self.interp_h(new_lon, new_lat)

        return tau_x_new, tau_y_new, hout


    def interp_h(self, new_lon, new_lat):
        """
        funtion that interpolates the water depth onto the new rectilinear grid
        Input: lon, lat of the new grid
        Output: interpolated h
        """
        new_SW=(new_lat.min(), new_lon.min())  ###(lat, lon)
        new_NE=(new_lat.max(), new_lon.max()) 
        
        xnew, ynew = self.convert_utm(new_lon, new_lat)
        xy_new = np.vstack((xnew.ravel(),ynew.ravel())).T  ## new grid

        lon_rho = self.data_roms['lon_rho'][1:-2,1:-2]
        lat_rho = self.data_roms['lat_rho'][1:-2,1:-2]
        mask_rho = self.data_roms['mask'][1:-2,1:-2]
        h = self.data_roms['h'][1:-2,1:-2]
        
        ind = self.findNearset(new_SW[1], new_SW[0], lon_rho, lat_rho)
        J0=ind[0][0] 
        I0=ind[0][1] 
        
        ind = self.findNearset(new_NE[1], new_NE[0], lon_rho, lat_rho)
        J1=ind[0][0] 
        I1=ind[0][1] 
        
        x_rho, y_rho = self.convert_utm(lon_rho, lat_rho)  #### convert to utm for interpolation
        y_rho_ss = y_rho[J0:J1,I0:I1]  ##subset x,y
        x_rho_ss = x_rho[J0:J1,I0:I1]
        mask_rho_ss = mask_rho[J0:J1,I0:I1]
        h_ss = h[J0:J1,I0:I1]
        
        xy_rho = np.vstack((x_rho_ss[mask_rho_ss==1],y_rho_ss[mask_rho_ss==1])).T        
        Fh = interpXYZ(xy_rho, xy_new)
        hout = np.zeros((xnew.shape[0], xnew.shape[1]))       
        print "interpolating ROMS depth h onto the new rectilinear grid!!! \n"
        #### Loop through time to do the interpolation ####
        htem = Fh(h_ss[:,:][mask_rho_ss==1].flatten()) 
        hout[:,:] = htem.reshape(xnew.shape[0], xnew.shape[1])
                
        return hout
        

        
    def calc(self):
        """
        Funtion that performs the calculation
        returns the calculated vorticity from the 2D advection diffusion equation
        """
        
        lon, lat, dx, dy = self.rectilinear(self.nx,self.ny)
        #### The interpolated ROMS u, v velocity
        u_all, v_all = self.interp_roms_uv(lon, lat)
        hout = self.interp_h(lon, lat)
        uout = np.sum(u_all, axis=1)
        vout = np.sum(v_all, axis=1)
        u = np.zeros_like(uout)
        v = np.zeros_like(vout)
        for tstep in range(len(self.data_roms['time'])):
            u[tstep,:,:] = uout[tstep,:,:]/hout[:,:]
            v[tstep,:,:] = vout[tstep,:,:]/hout[:,:]
            
        u, v = self.rot2d(u, v, math.radians(-35))
        
        #### The interpolated filtered SUNTANS vorticity
        w_filter = self.vorticity_filtered(lon, lat)
        
        stress = self.stress()
        
        X, Y = lon.shape
        ## The time interval for ROMS and SUNTANS is one hour 
        dt = 3600. 
        AH = 10.    ## The horizontal eddy viscosity
        
        time = self.data_roms['time']
        nt = len(time) 
        #### Initilize the matrix ####
        wout1 = np.zeros((nt, X, Y))   ##X - i, Y - j
        wout2 = np.zeros((nt, X, Y))
        A = np.zeros((X, Y))
        B = np.zeros((X, Y))
        C = np.zeros((X, Y))
        D = np.zeros((X, Y))
        #### Define the boundary: only on SUNTANS side, there is filtered vorticity ####
        wout1[:,-1,:] = w_filter[:,-1,:]
        wout2[:,-1,:] = w_filter[:,-1,:]
        ## Initial Condition ##
        wout2[0,:,:] = w_filter[0,:,:]
        #### main loop for calculation ####
        for n in range(nt-1):
            ## Sweep in Y- direction ##            
            for j in range(1,Y-1):
                A[1,j] = u[n,1,j]*dt/4./dx - AH*dt/2./dx/dx
                B[1,j] = 1 + AH*dt/dx/dx
                C[1,j] = 0.
                D[1,j] = (-v[n,1,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,1,j+1] \
                        + (1-AH*dt/dy/dy)*wout2[n,1,j] + (v[n,1,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,1,j-1] \
                        + stress[n,1,j]*dt/2./3600. - (-u[n,1,j]*dt/4./dx - AH*dt/2./dx/dx)*wout1[n+1,0,j]
                A[-2,j] = 0.
                B[-2,j] = 1 + AH*dt/dx/dx
                C[-2,j] = -u[n,-2,j]*dt/4./dx - AH*dt/2./dx/dx
                D[-2,j] = (-v[n,-2,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,-2,j+1] \
                        + (1-AH*dt/dy/dy)*wout2[n,-2,j] + (v[n,-2,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,-2,j-1] \
                        + stress[n,-2,j]*dt/2./3600. - (u[n,-2,j]*dt/4./dx - AH*dt/2./dx/dx)*wout1[n+1,-1,j]
                for i in range(2,X-2):
                    A[i,j] = u[n,i,j]*dt/4./dx - AH*dt/2./dx/dx
                    B[i,j] = 1 + AH*dt/dx/dx
                    C[i,j] = -u[n,i,j]*dt/4./dx - AH*dt/2./dx/dx
                    D[i,j] = (-v[n,i,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,i,j+1] \
                        + (1-AH*dt/dy/dy)*wout2[n,i,j] + (v[n,i,j+1]*dt/4./dy+AH*dt/2/dy/dy)*wout2[n,i,j-1] \
                        + stress[n,1,j]*dt/2./3600.
                ## initilize the matrix ##
                M1 = np.zeros((X-2,X-2))
                for k in range(M1.shape[0]):
                    if k==0:
                        M1[k,0]=B[k+1,j]; M1[k,1]=A[k+1,j]
                    elif k==M1.shape[0]-1:
                        M1[k,-2]=C[k+1,j]; M1[k,-1]=B[k+1,j]
                    else:
                        M1[k,k-1]=C[k+1,j]; M1[k,k]=B[k+1,j]; M1[k,k+1]=A[k+1,j]
                ## Solve the diagonal matrix ##
                wout1[n,1:-1,j] = np.linalg.solve(M1[:,:],D[1:-1,j])  
            
            A = np.zeros((X, Y))
            B = np.zeros((X, Y))
            C = np.zeros((X, Y))
            D = np.zeros((X, Y))
            ## Sweep in X- direction ##
            for i in range(1,X-1):
                A[i,1] = v[n,i,1]*dt/4./dy - AH*dt/2./dy/dy
                B[i,1] = 1 + AH*dt/dy/dy
                C[i,1] = 0.
                D[i,1] = (-u[n,i+1,1]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i+1,1] \
                        + (1-AH*dt/dx/dx)*wout1[n,i,1] + (u[n,i+1,1]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i-1,1] \
                        + stress[n,i,1]*dt/2./3600. - (-v[n,i,1]*dt/4./dy - AH*dt/2./dy/dy)*wout2[n+1,i,0]
                A[i,-2] = 0.
                B[i,-2] = 1 + AH*dt/dy/dy
                C[i,-2] = -v[n,i,-2]*dt/4./dy - AH*dt/2./dy/dy
                D[i,-2] = (-u[n,i+1,-2]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i+1,-2] \
                        + (1-AH*dt/dx/dx)*wout1[n,i,-2] + (u[n,i+1,-2]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i-1,-2] \
                        + stress[n,i,-2]*dt/2./3600. - (v[n,i,-2]*dt/4./dy - AH*dt/2./dy/dy)*wout2[n+1,i,-1]
                for j in range(2,Y-2):
                    A[i,j] = v[n,i,j]*dt/4./dy - AH*dt/2./dy/dy
                    B[i,j] = 1 + AH*dt/dy/dy
                    C[i,j] = -v[n,i,j]*dt/4./dy - AH*dt/2./dy/dy
                    D[i,j] = (-u[n,i+1,j]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i+1,j] \
                        + (1-AH*dt/dx/dx)*wout1[n,i,j] + (u[n,i+1,j]*dt/4./dx+AH*dt/2/dx/dx)*wout1[n,i-1,j] \
                        + stress[n,i,1]*dt/2./3600.
                        
                ## initilize the matrix ##
                M2 = np.zeros((Y-2,Y-2))
                for k in range(M2.shape[0]):
                    if k==0:
                        M2[k,0]=B[i,k+1]; M2[k,1]=A[i,k+1]
                    elif k==M2.shape[0]-1:
                        M2[k,-2]=C[i,k+1]; M2[k,-1]=B[i,k+1]
                    else:
                        M2[k,k-1]=C[i,k+1]; M2[k,k]=B[i,k+1]; M2[k,k+1]=A[i,k+1]
                ## Solve the diagonal matrix ##
                wout2[n+1,i,1:-1] = np.linalg.solve(M2[:,:],D[i,1:-1]) 
            
            
#        ######################################################################
#        #### These are just for testing if the interpolation is correct   ####
#        #### commented if not using                                       ####
#        timeformat = '%Y%m%d-%H%M'
#        for i in range(len(time)):
#            west=-95.42; east=-93.94
#            south=28.39;  north=29.90               
#            fig = plt.figure(figsize=(10,8))
#            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
#                        llcrnrlon=west,urcrnrlon=east, resolution='h')
#            
#            basemap.drawcoastlines()
#            basemap.fillcontinents(color='coral',lake_color='aqua')
#            basemap.drawcountries()
#            basemap.drawstates()  
#            
#            llons, llats=basemap(lon,lat)   
#            con = basemap.pcolormesh(llons,llats,wout2[i,:,:])
#            #con.set_clim(vmin=-0.08, vmax=0.03)
#            cbar = plt.colorbar(con, orientation='vertical')
#            cbar.set_label("vorticity")
#            timestr = datetime.strftime(time[i], timeformat)
#            plt.title('vorticity at %s'%timestr)
#            plt.savefig(self.wdr+'/vorticity_figure/calculated_vorticity/'+str(i)+'.png')
#            print "Saving figure %s to calculated vorticity figure directory"%str(i)
#            #plt.show() 
            ######################################################################

        return wout2
        
    def calc_uv(self):
        """
        calculate the U, V velocity from the resulting vorticity
        """ 
        
        def flow_direction(u, v):
            """
            Input: u, v velocity //
            Output: flow direction
            """
            
            return np.arctan2(u, v) * 180. / np.pi % 360
        
        
        lon, lat, dx, dy = self.rectilinear(self.nx,self.ny)
        w = self.calc()
        
        #### The interpolated ROMS u, v velocity
        ur, vr = self.interp_roms_uv(lon, lat)
        ur = ur[:,0,:,:]
        vr = vr[:,0,:,:]
        ur, vr = self.rot2d(ur, vr, math.radians(35))
        spd_roms = np.sqrt(ur*ur+vr*vr)
        dir_roms = flow_direction(ur, vr)        
        
        uu = -w/2.* dy
        vv = w/2.* dx
        spd_filter = np.sqrt(uu*uu+vv*vv)        
        
        unew = ur + uu
        vnew = vr + vv
        #spd_new = spd_roms + spd_filter
        spd_new = np.sqrt(unew*unew+vnew*vnew)
        dir_new = flow_direction(unew, vnew)
                    
        pdb.set_trace()
        
        #### Compare ####
        #row = np.floor(self.ny/2.);
        row = self.ny-2
        col = np.floor(self.nx/2.)     
        
        Point1 = (lat[row,col], lon[row,col])        
        nc = Dataset(self.sunfile, 'r')
        ftime = nc.variables['time']
        time = num2date(ftime[:],ftime.units)
        xv = nc.variables['xv'][:]
        yv = nc.variables['yv'][:]
        lon_sun = np.zeros_like(xv)
        lat_sun = np.zeros_like(yv)
        for i in range(len(xv)):
            lat_sun[i], lon_sun[i] = utm.to_latlon(xv[i],yv[i], 15, 'U')[0:2]
        
        ind0 = self.findNearest(self.starttime,time)
        ind1 = self.findNearest(self.endtime,time)
        ind_sun = self.findNearset(Point1[1], Point1[0], lon_sun, lat_sun)[0][0]        
        usun = nc.variables['uc'][ind0:ind1+1,0,ind_sun]
        vsun = nc.variables['vc'][ind0:ind1+1,0,ind_sun]
        spd_sun = np.sqrt(usun*usun + vsun*vsun)
        dir_sun = flow_direction(usun, vsun)
        
        ind = self.findNearset(Point1[1], Point1[0], lon, lat)
        J0=ind[0][0] 
        I0=ind[0][1]
        #[J0,I0]
        
#        ######################################################################
#        #### commented if not using                                       ####
#        timeformat = '%Y%m%d-%H%M'
#        for i in range(len(time)):
#            west=-95.42; east=-93.94
#            south=28.39;  north=29.90
#            fig = plt.figure(figsize=(10,8))
#            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
#                        llcrnrlon=west,urcrnrlon=east, resolution='h')
#            
#            basemap.drawcoastlines()
#            basemap.fillcontinents(color='coral',lake_color='aqua')
#            basemap.drawcountries()
#            basemap.drawstates()  
#            
#            llons, llats=basemap(lon,lat)   
#            con = basemap.contourf(llons,llats,unew[i,:,:])
#            #con.set_clim(vmin=-0.08, vmax=0.03)
#            cbar = plt.colorbar(con, orientation='vertical')
#            cbar.set_label("velocity")
#            timestr = datetime.strftime(time[i], timeformat)
#            plt.title('new u velocity at %s'%timestr)
#            plt.savefig(self.wdr+'/vorticity_figure/new_velocity/'+str(i)+'.png')
#            print "Saving figure %s to calculated new velocity figure directory"%str(i)
#            #plt.show()
        
             
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        l1 = ax1.plot(self.data_roms['time'], spd_new[:,row,col], '-r', label='new')
        l2 = ax1.plot(self.data_roms['time'], spd_roms[:,J0,I0], '-b', label='ROMS')
        l3 = ax1.plot(self.data_roms['time'], spd_sun[:], '-k', label='SUNTANS')
        ax1.legend()
        ax1.set_ylabel('velocity (m/s)')
        ax1.set_title('flow velocity at (%s, %s)'%(str(lat[row,col]), str(lon[row,col])))        
        
        l4 = ax2.plot(self.data_roms['time'], dir_new[:, row, col], 'or', label='new')
        l5 = ax2.plot(self.data_roms['time'], dir_roms[:, J0, I0], 'ob', label='ROMS')
        l6 = ax2.plot(self.data_roms['time'], dir_sun[:], 'ok', label='SUNTANS')
        ax2.legend()
        ax2.set_ylabel('flow angle')
        ax2.set_title('flow direction at (%s, %s)'%(str(lat[row,col]), str(lon[row,col])))  
        
        plt.show()        
         
#        l4 = ax2.plot(self.data_roms['time'], unew[:, row, col], label='new')
#        l5 = ax2.plot(self.data_roms['time'], ur[:, J0, I0], label='ROMS')
#        l6 = ax2.plot(self.data_roms['time'], usun[:], label='SUNTANS')
#        ax2.legend()
#        ax2.set_ylabel('velocity (m/s)')
#        ax2.set_title('U velocity at (%s, %s)'%(str(lat[row,col]), str(lon[row,col])))  
#        
#        l7 = ax3.plot(self.data_roms['time'], vnew[:, row, col], label='new')
#        l8 = ax3.plot(self.data_roms['time'], vr[:, J0, I0], label='ROMS')
#        l9 = ax3.plot(self.data_roms['time'], vsun[:], label='SUNTANS')
#        ax3.legend()
#        ax3.set_ylabel('velocity (m/s)')
#        ax3.set_title('V velocity at (%s, %s)'%(str(lat[row,col]), str(lon[row,col])))  
#        
#        plt.show()
                              
        pdb.set_trace()         
         
          
    def grid_plot(self):
        """
        function for visualizing the generated grid
        """
        
        lon, lat, dx, dy  = self.rectilinear(self.nx,self.ny)
        pdb.set_trace()
        west=-95.42; east=-93.94
        south=28.39;  north=29.90
        fig = plt.figure(figsize=(10,10))
        basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
                        llcrnrlon=west,urcrnrlon=east, resolution='h')                        
        basemap.drawcoastlines()
        basemap.fillcontinents(color='coral',lake_color='aqua')
        basemap.drawcountries()
        basemap.drawcounties()
        basemap.drawstates()  
        basemap.drawrivers(color='b')

        llons, llats=basemap(lon,lat)
        basemap.plot(llons, llats, color='k', ls='-', markersize=.5)
        basemap.plot(llons.T, llats.T, color='k', ls='-', markersize=.5)
        plt.show()    
            
    def rot2d(self, x, y, ang):
        """
        rotate vectors by geometric angle
        This routine is part of Rob Hetland's OCTANT package:
            https://github.com/hetland/octant
        """
        xr = x*np.cos(ang) - y*np.sin(ang)
        yr = x*np.sin(ang) + y*np.cos(ang)
        return xr, yr            
     
     
    def distance_on_unit_sphere(self,lat1, long1, lat2, long2):
        """
        function that calcuates the distance between two coordinates
        """            
        degrees_to_radians = math.pi/180.0
        phi1 = (90.0 - lat1)*degrees_to_radians
        phi2 = (90.0 - lat2)*degrees_to_radians
        
        theta1 = long1*degrees_to_radians
        theta2 = long2*degrees_to_radians
        
        cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
        math.cos(phi1)*math.cos(phi2))
        arc = math.acos( cos )
        
        return arc*6373000  ##The unit is meter
        
    def convert_utm(self, lon, lat):
        """
        Input: lon, lat
        Output: x,   y (utm) 
        """
        x = np.zeros_like(lon)
        y = np.zeros_like(lat)
        (ly, lx) = lon.shape
        for i in range(ly):
            for j in range(lx):
                (y[i,j], x[i,j])=utm.from_latlon(lat[i,j],lon[i,j])[0:2]
                
        return x, y
        
        
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
        
    
    def findNearset(self,x,y,lon,lat):
        """
        Return the J,I indices of the nearst grid cell to x,y
        """
        
        dist = np.sqrt( (lon - x)**2 + (lat - y)**2)
        
        return np.argwhere(dist==dist.min())
    
    
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

        
#### For testing ####        
if __name__ == "__main__":
    #starttime = '2014-03-22'
    #endtime = '2014-03-27'
    starttime = '2009-06-01'
    endtime = '2009-06-08'
    cal_vorticity(starttime, endtime)       
        
        
    