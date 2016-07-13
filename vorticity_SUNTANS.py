# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:17:31 2016

@author: dongyu
"""

import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import os
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import utm
import operator
import pdb

class vorticity(object):
    """
    general class for calculating vorticity from SUNTANS unstructured surface velcoity field
    """
    _FillValue=999999    
    
    def __init__(self, starttime, endtime, **kwargs):
        self.__dict__.update(kwargs)
        
        #### Specify the starttime and endtime ####
        self.starttime = datetime.strptime(starttime, '%Y-%m-%d')
        self.endtime = datetime.strptime(endtime, '%Y-%m-%d')
        #### Read data from model output
        wdr = os.getcwd()
        sunfile = wdr+'/SUNTANS_file/GalvCoarse_0000.nc'
        #self.readFile(sunfile)
        
        #### Option 1) plot function is used to plot SUNTANS vorticity ####
        #self.plot()
        #### Option 2) interp function is used to interpolate first ####
        #self.interp_plot()
        
        
        
    def readFile(self, filename):
        """
        function that reads from netcdf file
        """
        nc = Dataset(filename, 'r')
        print "#### Reading ROMS output file !!!! ####\n"
        #print nc
        #self.data = dict()
        #self.grid = dict()
        timei = nc.variables['time']
        time = num2date(timei[:],timei.units)       
        self.ind0 = self.findNearest(self.starttime,time)
        self.ind1 = self.findNearest(self.endtime,time)
        self.time = time[self.ind0:self.ind1+1]
        
        self.Nt = len(self.time)
        self.u = nc.variables['uc'][self.ind0:self.ind1+1,0,:]
        self.v = nc.variables['vc'][self.ind0:self.ind1+1,0,:]
        
        self.Ac = nc.variables['Ac'][:]
        self.Nk = nc.variables['Nk'][:]
        self.dz = nc.variables['dz'][:]
        self.Nz = len(self.dz)
        self.klayer = np.arange(0,self.Nz)
        #xv = nc.variables['xv'][:]    #xi: latitude
        #yv = nc.variables['yv'][:]    #yi: longitude
        
        #### converison from utm ####
        #lat = np.zeros_like(xv)
        #lon = np.zeros_like(yv)
        #for i in range(len(xv)):
        #    (lat[i],lon[i]) = utm.to_latlon(xv[i],yv[i],15,'U')
        #self.data['xv'] = xv
        #self.data['yv'] = yv
        
        basedir = os.getcwd()
        edgedata = self.readTXT(basedir+'/SUNTANS_file/edges.dat')
        self.grad = np.asarray(edgedata[:,3:5],int)
        
        celldata = self.readTXT(basedir+'/SUNTANS_file/cells.dat')
        self.cells = np.asarray(celldata[:,3:6],int)
        self.Nc = celldata.shape[0]
        self.nfaces = 3*np.ones((self.Nc,),np.int)
        self.xv = np.asarray(celldata[:,1], float)
        self.yv = np.asarray(celldata[:,2], float)
        
        pointdata = self.readTXT(basedir+'/SUNTANS_file/points.dat')
        self.xp = pointdata[:,0]
        self.yp = pointdata[:,1]
        
        self.cellmask = self.cells==int(self._FillValue)
        
        self.face = nc.variables['face'][:]
        self.face[self.cellmask]=0
        self.face =\
                np.ma.masked_array(self.face,mask=self.cellmask,fill_value=0)

	self.mark = nc.variables['mark'][:]
        self.xe = nc.variables['xe'][:]
        self.ye = nc.variables['ye'][:] 

	self.dv = nc.variables['dv'][:]
	self.tau_x = nc.variables['tau_x'][self.ind0:self.ind1+1,:]
	self.tau_y = nc.variables['tau_y'][self.ind0:self.ind1+1,:]	       

        #pdb.set_trace()

    def vorticity_circ(self, uu, vv, k=0):
        """
        Calculate vertical vorticity component using the 
        circulation method
        """
        # Load the velocity
        #u,v,w = self.getVector()
        u = uu
        v = vv
        
                             
        def _AverageAtFace(phi,jj,k):
            
            grad1 = self.grad[:,0]
            grad2 = self.grad[:,1]
            #Apply mask to jj
            jj[jj.mask]=0
            nc1 = grad1[jj]
            nc2 = grad2[jj]
                    
            # check for edges (use logical indexing)
            ind1 = nc1==-1
            nc1[ind1]=nc2[ind1]
            ind2 = nc2==-1
            nc2[ind2]=nc1[ind2]
            
            # check depths (walls)
            indk = operator.or_(k>=self.Nk[nc1], k>=self.Nk[nc2])
            ind3 = operator.and_(indk, self.Nk[nc2]>self.Nk[nc1])
            nc1[ind3]=nc2[ind3]
            ind4 = operator.and_(indk, self.Nk[nc1]>self.Nk[nc2])
            nc2[ind4]=nc1[ind4]
            
            # Average the values at the face          
            return 0.5*(phi[nc1]+phi[nc2]) 
            
        # Calculate the edge u and v
        ne = self.face #edge-indices
        
        ue = _AverageAtFace(u,ne,k)
        ve = _AverageAtFace(v,ne,k)
        ue[self.cellmask]=0
        ve[self.cellmask]=0
        
        tx,ty,mag = self.calc_tangent()
        
        tx[self.cellmask]=0
        ty[self.cellmask]=0
        
        # Now calculate the vorticity
        return np.sum( (ue*tx + ve*ty )*mag,axis=-1)/self.Ac
        
    def vorticity(self):
        """
        Calculate the vertical vorticity component
        
        Uses gradient method
        """
        
        u,v,w = self.getVector()
            
        sz = u.shape
         
        if len(sz)==1: # One layer
            du_dx,du_dy = self.gradH(u,k=self.klayer[0])
            dv_dx,dv_dy = self.gradH(v,k=self.klayer[0])
            
            data = dv_dx - du_dy
            
        else: # 3D
            data = np.zeros(sz)
            
            for k in self.klayer:
                du_dx,du_dy = self.gradH(u[:,k],k=k)
                dv_dx,dv_dy = self.gradH(v[:,k],k=k)
            
                data[:,k] = dv_dx - du_dy
                
        return data
        
    def gradH(self,cell_scalar,k=0,cellind=None):
        """
        Compute the horizontal gradient of a cell-centred quantity

        """
        if self.maxfaces==3:
            dX,dY=self.gradHplane(cell_scalar,k=k,cellind=cellind)
        else:
            dX,dY=self.gradHdiv(cell_scalar,k=k)

        return dX,dY
        
#    def getVector(self):
#        """
#        Retrieve U and V vector components
#        """
#        tmpvar = self.variable
#        
#        u=self.loadDataRaw(variable='uc',setunits=False)
#        
#        v=self.loadDataRaw(variable='vc',setunits=False)
#        
#        try:
#            w=self.loadDataRaw(variable='w',setunits=False)
#        except:
#            w=u*0.0
#                               
#        self.variable=tmpvar
#        # Reload the original variable data
#        #self.loadData()
#        
#        return u,v,w        

    def calc_tangent(self):
        """
        Calculate the tangential vector for the edges of each cell
        """
        if not self.__dict__.has_key('_tx'):
            dx = np.zeros(self.cells.shape)    
            dy = np.zeros(self.cells.shape)  
    
            dx[:,0:-1] = self.xp[self.cells[:,1::]] - self.xp[self.cells[:,0:-1]]               
            dy[:,0:-1] = self.yp[self.cells[:,1::]] - self.yp[self.cells[:,0:-1]]               
    
            for ii in range(self.Nc):
                dx[ii,self.nfaces[ii]-1] = self.xp[self.cells[ii,0]] - self.xp[self.cells[ii,self.nfaces[ii]-1]]  
                dy[ii,self.nfaces[ii]-1] = self.yp[self.cells[ii,0]] - self.yp[self.cells[ii,self.nfaces[ii]-1]]  
   
            
            mag = np.sqrt(dx*dx + dy*dy)
            
            self._tx = dx/mag
            self._ty = dy/mag
            self._mag = mag
                     
        return self._tx, self._ty, self._mag        
        
        
    def plot(self):
        """
        this funtion plot the contour of vorticity on the basemap
        """     

        timeformat = '%Y%m%d-%H%M'        
        
        w = np.zeros((self.Nt, self.Nc))
        for tstep in range(self.Nt):
            w[tstep,:] = self.vorticity_circ(self.u[tstep,:], self.v[tstep,:], k=self.klayer[0])      
        
        #pdb.set_trace()
        maxfaces = self.cells.shape[1]
        
        #########################################################
        basedir = os.getcwd()
        pointdata = self.readTXT(basedir+'/SUNTANS_file/points.dat')
        xp1 = pointdata[:,0]
        yp1 = pointdata[:,1]
        #########################################################
        xp = np.zeros((self.Nc,maxfaces+1))
        yp = np.zeros((self.Nc,maxfaces+1))
            
        cells=self.cells.copy()
        #cells[cells.mask]=0
        xp[:,:maxfaces]=xp1[cells]
        xp[range(self.Nc),self.nfaces]=xp1[cells[:,0]]
        yp[:,:maxfaces]=yp1[cells]
        yp[range(self.Nc),self.nfaces]=yp1[cells[:,0]]
        ##########################################################
        xy = np.zeros((maxfaces+1,2))
        def _closepoly(ii):
            nf=self.nfaces[ii]+1
            xy[:nf,0]=xp[ii,:nf]
            xy[:nf,1]=yp[ii,:nf]
            return xy[:nf,:].copy()

        cellxy= [_closepoly(ii) for ii in range(self.Nc)]

        #clim=[w.min(),w.max()]
        clim=[-0.0010,0.0010]
        xlims=(self.xv.min(),self.xv.max())
        ylims=(self.yv.min(),self.yv.max())        
        
        
        for i in range(len(self.time)):
            fig = plt.figure(figsize=(10,8))
            axes = fig.add_subplot(111)
            collection = PolyCollection(cellxy,cmap='jet')
            collection.set_array(np.array(w[i,:]))
            collection.set_edgecolors('k')
            collection.set_linewidths(0.2)
            #collection.set_clim(vmin=clim[0],vmax=clim[1])
            collection.set_edgecolors(collection.to_rgba(np.array((w[i,:])))) 
            cbar = fig.colorbar(collection,orientation='vertical')
            axes.add_collection(collection)
            axes.set_xlim(xlims)
            axes.set_ylim(ylims)
            axes.set_aspect('equal')
            axes.set_xlabel('Easting [m]')
            axes.set_ylabel('Northing [m]')
            timestr = datetime.strftime(self.time[i], timeformat)
            plt.title('vorticity at %s'%timestr)
            plt.savefig(basedir+'/vorticity_figure/SUNTANS_figure/'+str(i)+'.png')

        #plt.show()
        
    def interp(self, ROMS_file):
        """
        This function interpolates the resulting vorticity to ROMS curvilinear grid
        for further operation
        """
        from interpXYZ import interpXYZ    
        #from vorticity_ROMS import vorticity
        
        wdr = os.getcwd()
	nc = Dataset(ROMS_file,'r')
        #nc = Dataset(wdr+'/download_ROMS/'+ROMS_file,'r')
        print "#### Reading ROMS output file !!!! ####\n"
        lon = nc.variables['lon_psi'][:][0:-1,0:-1]
        lat = nc.variables['lat_psi'][:][0:-1,0:-1]
        mask = nc.variables['mask_psi'][:][0:-1,0:-1]
        
        #### Step 1) Prepare x, y coordinate to do the interpolation ####
        xroms = np.zeros_like(lon)
        yroms = np.zeros_like(lat)
        (y,x) = lon.shape
        for i in range(y):
            for j in range(x):
                (yroms[i,j],xroms[i,j])=utm.from_latlon(lat[i,j],lon[i,j])[0:2]
        
        #### Step 2) subset ROMS grid for interpolation ####
        def findNearset(x,y,lon,lat):
            """
            Return the J,I indices of the nearst grid cell to x,y
            """
                        
            dist = np.sqrt( (lon - x)**2 + (lat - y)**2)
            
            return np.argwhere(dist==dist.min())
        
        SW=utm.to_latlon(self.xv.min(),self.yv.min(),15,'R')  ###(lat, lon)
        NE=utm.to_latlon(self.xv.max(),self.yv.max(),15,'R')
        
        ind = findNearset(SW[1], SW[0], lon, lat)
        J0=ind[0][0] - 15
        I0=ind[0][1] + 5
        
        ind = findNearset(NE[1], NE[0], lon, lat)
        J1=ind[0][0] + 5
        I1=ind[0][1] - 6
        
        #pdb.set_trace()
        yss = yroms[J0:J1,I0:I1]  ##subset x,y
        xss = xroms[J0:J1,I0:I1]
        maskss = mask[J0:J1,I0:I1]
        
        #### Step 3) Prepare the grid variables for the interpolation class ####
        xy_sun = np.vstack((self.yv.ravel(),self.xv.ravel())).T   ## SUNTANS grid, xi: latitude, yi: longitude
        xy_new = np.vstack((xss[maskss==1],yss[maskss==1])).T   ## blended grid
        Fw = interpXYZ(xy_sun,xy_new, method='idw')
        
        #### define the spatial scales ####
        (Y, X) = xss.shape
        
        #### Initialize the output array @ subset roms grid ####
        wout = np.zeros((self.Nt, Y, X))
        
        #### Calculate the SUNTANS vorticity ####
        w = np.zeros((self.Nt, self.Nc))
        for tstep in range(self.Nt):
            w[tstep,:] = self.vorticity_circ(self.u[tstep,:], self.v[tstep,:], k=self.klayer[0])
        
        #### Loop througth time to do the interpolation ####            
        for tstep in range(self.Nt):
            wtem = Fw(w[tstep,:])
            wout[tstep, maskss==1] = wtem
            
	
        #### Plotting !!! ####
        lonss = lon[J0:J1,I0:I1] ## subset lon, lat
        latss = lat[J0:J1,I0:I1]     
        
        ################################################
        #### Spatial average ####
        #wout, lonss, latss = self.average(wout, lonss, latss, maskss)
        ################################################
        return wout, lonss, latss, maskss        
        
        
    def interp_plot(self):
        """
        function that plots the interpolated SUNTANS vorticity
        """
	wdr = os.getcwd()

	ROMS_file = wdr+'/download_ROMS/txla_subset_HIS.nc'
        w, lonss, latss, maskss = self.interp(ROMS_file)
        
        timeformat = '%Y%m%d-%H%M'
        south = latss.min(); north =latss.max()
        west = lonss.min(); east = lonss.max()

        
        for i in range(len(self.time)):
            fig = plt.figure(figsize=(10,8))
            basemap = Basemap(projection='merc',llcrnrlat=south,urcrnrlat=north,\
                      llcrnrlon=west,urcrnrlon=east, resolution='h')
            
            basemap.drawcoastlines()
            basemap.fillcontinents(color='coral',lake_color='aqua')
            basemap.drawcountries()
            basemap.drawstates()  
            
            llons, llats=basemap(lonss,latss)   
            con = basemap.pcolormesh(llons,llats,w[i,:,:])
            con.set_clim(vmin=-0.0003, vmax=0.0003)
            cbar = plt.colorbar(con, orientation='vertical')
            cbar.set_label("vorticity")
            #plt.show()
            timestr = datetime.strftime(self.time[i], timeformat)
            plt.title('vorticity at %s'%timestr)
            plt.savefig(wdr+'/vorticity_figure/SUNTANS_figure_average/'+str(i)+'.png')
            print "Saving figure %s to SUNTANS figure directory"%str(i)


    def average(self, vor, lon, lat, mask):
        """
        do the spatial average for vorticity
        """
        from scipy import interpolate        
        
        #mask = self.data['mask'][0:-1,0:-1]
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
        import math
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


    def obc_mask(self, lonss, latss, maskss):
        """
        function that outputs the masked area below the open boundary condition
        """   
        
        ####################################################################
        def findNearset(x,y,lon,lat):
            """
            Return the J,I indices of the nearst grid cell to x,y
            """
                        
            dist = np.sqrt( (lon - x)**2 + (lat - y)**2)
            
            return np.argwhere(dist==dist.min())
        
        tide = [] # obc boundary mark
        river = [] #river boundary mark
        for i in self.mark:
            if i == 3:
                tide.append(i)
            if i == 2:
                river.append(i)
        
        tide_index=[]
        for ind, mk in enumerate(self.mark):
            if mk == 3:
                tide_index.append(ind)
        
        open_boundary = []
        for kk in tide_index:
            open_boundary.append([self.xe[kk], self.ye[kk]])
        
        boundary=[]
        for i in range(len(open_boundary)):
            boundary.append(utm.to_latlon(open_boundary[i][0],open_boundary[i][1],15, 'U'))
            
        #### The main loop to obtain the mask ####
        for i in range(len(boundary)):
            ind = findNearset(boundary[i][1], boundary[i][0], lonss, latss)
            J0 = ind[0][0]; I0 = ind[0][1]
            maskss[0:J0,I0] = 0
            #pdb.set_trace()
                
        return maskss 
             
        
    def readTXT(self,fname,sep=None):
        """
        Reads a txt file into an array of floats
        """
        
        fp = file(fname,'rt')
        data = np.array([map(float,line.split(sep)) for line in fp])
        fp.close()
        
        return data
        
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
        
        
