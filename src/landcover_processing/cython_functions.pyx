from __future__ import print_function
cimport numpy as np
import numpy as np
from scipy import spatial

#"""
#distance_to_mask
#----------------
#This function takes an binary mask and finds the distance of each pixel to the
#nearest pixel with value == 1. If NaN values are present, these pixels are
#ignored.
#Optional arguments dx and dy are there so that the resolution of the raster can
#be accounted for if desired
#"""


def distance_to_mask(np.ndarray[np.int64_t,ndim=2] mask_array,np.ndarray[np.int64_t,ndim=2] nodata_mask,double dx,double dy,trees):
    # get coordinates of pixels in mask==1
    cdef double[:] x,y
    cdef double dist_iter
    cdef np.ndarray[np.int64_t,ndim=1] test_rows,test_cols,mask_rows,mask_cols
    cdef double[:,:] xx,yy,distance
    cdef int ii,jj,Ny,Nx,count,interval,Ntest,Nmask,row_idx,col_idx,print_at

    y = np.arange(mask_array.shape[0])*dy
    x = np.arange(mask_array.shape[1])*dx
    Nx = x.size
    Ny = y.size

    N_trees = len(trees)

    # filter pixels to test only pixels where mask==1
    distance = np.zeros((Ny,Nx))*np.nan
    mask_rows,mask_cols = np.where(np.all((mask_array==1,nodata_mask==0),axis=0))
    Nmask=mask_rows.size
    for ii in range(0,Nmask):
        distance[mask_rows[ii],mask_cols[ii]]=0

    test_rows,test_cols=np.where(np.all((mask_array==0,nodata_mask==0),axis=0))
    Ntest = test_rows.size
    count = 0
    interval = 10000
    print_at = 0
    for ii in range(0,Ntest):
        row_idx=test_rows[ii]
        col_idx=test_cols[ii]
        if count==print_at:
            print('\t{0:.3f}%\r'.format(float(ii)/float((Ntest))*100),end='\r')
            print_at = print_at + interval

        distance[row_idx,col_idx]=trees[0].query([x[col_idx],y[row_idx]],k=1)[0]
        for tt in range(1,N_trees):
            dist_iter = trees[tt].query([x[col_idx],y[row_idx]],k=1)[0]
            if dist_iter<distance[row_idx,col_idx]:
                distance[row_idx,col_idx]=dist_iter
        count=count+1
    return distance
