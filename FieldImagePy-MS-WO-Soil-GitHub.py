#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install shapely


# In[ ]:


pip install fiona


# In[ ]:


pip install geopandas


# In[ ]:


pip install rasterio


# In[ ]:


pip install pyshp


# In[ ]:


pip install rasterstats


# In[ ]:


# Importing necessary libraries
import geopandas as gpd
import rasterio.mask
import fiona
import os
import numpy as np
import shapefile
import rasterstats
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import rasterio
import time


# In[ ]:


#Importing necessary programs from the libraries
from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio import features
from rasterstats import zonal_stats
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


# In[ ]:





# In[ ]:


# Importing plot shapefile for clipping the imahe
# The original image used in this program spanned over a larger region than the plot area (which is quite common ni most cases)
# This could increase computing time
# So the plot shapefiles are being brought here to clip the image according to plot areas

with fiona.open("your file path/shapefile name.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


# In[ ]:





# In[ ]:


# Clipping the image by plot area

# Importing original image, the one to be clipped
nodata=0
with rasterio.open("your file path/image name.tif", 'r') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, nodata, crop=True)
    out_meta = src.meta

# Importing shapefile geometry for clipping the image
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})


# In[ ]:


# Exporting clipped image
# This step is optional, but recommended to cross check whether clipping has been performed properly
with rasterio.open("clipped image name.tif", "w", **out_meta) as raster:
    # If no filepath is added this way, the image would be saved in compiler's (Jupyter Notebook, Spyder, etc.) memory
    # If ony one name is used, the image will be replaced by the same name every time the image is changed
    raster.write(out_image)


# In[ ]:





# In[ ]:


# Importing the clipped image
with rasterio.open("clipped image name.tif.tif", 'r') as raster2:
        Blue16bit = raster2.read(1) #1 stands for Blue band 
        Green16bit = raster2.read(2) #2 stands for Green band 
        Red16bit = raster2.read(3) #3 stands for Red band
        NIR16bit = raster2.read(4) #4 stands for Red band
        RE16bit = raster2.read(5) #5 stands for Red band


# In[ ]:


# Converting digital numbers (DN) to reflectance
# This step is quite important and highly recommended, unless already have been done while image processing
# Images convertd to reflectance values should range between 0 - 1 or 0 - 100%
# The equation for conversion is ((DN/(2^n)) - 1), where n is the bit size of the camera
# Digital cameras generally store imaes as 8 bit or 16 bit
# For this example n = 16, and thus ((DN/(2^16)) - 1) = 65535
Blue=(Blue16bit/65535).astype(float)
Green=(Green16bit/65535).astype(float)
Red=(Red16bit/65535).astype(float)
NIR=(NIR16bit/65535).astype(float)
RE=(RE16bit/65535).astype(float)


# In[ ]:





# In[ ]:


# Calculating vegetation indices
# This example shows with 15 vegetation indices
# Any number of vegetation indices can be used

# Dealing with the situations division by zero
np.seterr(divide='ignore', invalid='ignore')

# Making original calculation
# Later on adjustments can be made to remove soil, or normallizing the vegetation indices, etc.
NDVI_Orig = (NIR.astype(float) - Red.astype(float)) / (NIR.astype(float) + Red.astype(float))
RENDVI_Orig = (RE.astype(float) - Red.astype(float)) / (RE.astype(float) + Red.astype(float))
GNDVI_Orig = (NIR.astype(float) - Green.astype(float)) / (NIR.astype(float) + Green.astype(float))
ENDVI_Orig = (NIR.astype(float) + Green.astype(float) - 2*Blue.astype(float)) / (NIR.astype(float) + Green.astype(float) + 2*Blue.astype(float))
NDRE_Orig = (NIR.astype(float) - RE.astype(float)) / (NIR.astype(float) + RE.astype(float))
NNIR_Orig = NIR.astype(float) / (NIR.astype(float) + (RE.astype(float) + Green.astype(float)))
MCARI_Orig = (RE.astype(float)-Red.astype(float)) - 2*(RE.astype(float) - Green.astype(float))*(RE.astype(float) / Red.astype(float))
SIPI_Orig = (NIR.astype(float)-Blue.astype(float))/(NIR.astype(float) + Red.astype(float))
NGRDI_Orig = ((Green).astype(float) - (Red).astype(float))/((Green).astype(float) + (Red).astype(float))
NLI_Orig = (((NIR.astype(float))**2) - Red.astype(float)) / (((NIR.astype(float))**2) + Red.astype(float))
SR_Orig = NIR.astype(float)/Red.astype(float)
DVI_Orig = NIR.astype(float) - Red.astype(float)
RDVI_Orig = (NIR.astype(float) - Red.astype(float)) / ((NIR.astype(float) + Red.astype(float))**(1/2))
MDD_Orig = (NIR.astype(float) - RE.astype(float)) - (RE.astype(float) - Green.astype(float))
MARI_Orig = ((1/Green.astype(float))-(1/RE.astype(float)))*NIR.astype(float)
HUE = np.arctan((2 * (Red - Green - Blue) )/ (30.5*(Green - Blue)))


# In[ ]:





# In[ ]:


# FINAL VEGETATION INDICES CALCULATION


# In[ ]:


# Separating crop and soil fractions based on NGRDI
# This step is ontional and based on the need of the research
# For this example NGRDI has been used to classify between soil and crop
# Any vegetation indices can be used (VI_For_Classification)
# basic syntax is: 
# VI = np.wnere(VI_For_Classification (symbol(s): > or < or = or !=) (classification criteria), VI_Of_Interest, -9999)
# -9999 is the number for null values

NDVI = np.where(NGRDI_Orig > 0, NDVI_Orig, -9999)
RENDVI = np.where(NGRDI_Orig > 0, RENDVI_Orig, -9999)
GNDVI = np.where(NGRDI_Orig > 0, GNDVI_Orig, -9999)
ENDVI = np.where(NGRDI_Orig > 0, ENDVI_Orig, -9999)
NDRE = np.where(NGRDI_Orig > 0, NDRE_Orig, -9999)
NNIR = np.where(NGRDI_Orig > 0, NNIR_Orig, -9999)
MCARI = np.where(NGRDI_Orig > 0, MCARI_Orig, -9999)
SIPI = np.where(NGRDI_Orig > 0, SIPI_Orig, -9999)
NGRDI = np.where(NGRDI_Orig > 0, NGRDI_Orig, -9999)
NLI = np.where(NGRDI_Orig > 0, NLI_Orig, -9999)
SR = np.where(NGRDI_Orig > 0, SR_Orig, -9999)
DVI = np.where(NGRDI_Orig > 0, DVI_Orig, -9999)
RDVI = np.where(NGRDI_Orig > 0, RDVI_Orig, -9999)
MDD = np.where(NGRDI_Orig > 0, MDD_Orig, -9999)
MARI = np.where(NGRDI_Orig > 0, MARI_Orig, -9999)


# In[ ]:





# In[ ]:


# Import shapefile for calculation
# Shapefile of the plot area
crop_gdf = gpd.read_file("your file path/shapefile name.shp")

# Get list of geometries for all features in vector file
# This will greate a geodatabase with each plot ID
geom = [shapes for shapes in crop_gdf.geometry]


# In[ ]:





# In[ ]:


# Rasterize vector using the shape and coordinate system of the raster
# This ste is required to count the total number of pixels per polygon
rasterized = features.rasterize(geom,
                                out_shape = raster2.shape,
                                fill = 0,
                                out = None,
                                transform = raster2.transform,
                                all_touched = False,
                                default_value = 1,
                                dtype = None)


# In[ ]:





# In[ ]:


# Accessing the transform information from the original raster
affine = raster2.transform


# Counting the number of pixels per plot
rasterized_mean = zonal_stats(crop_gdf, rasterized, affine=affine, geojson_out=True,stats=['count'], nodata = -9999)
Crop_Poly_rasterized_Data = []
i=0
while i<len(rasterized_mean):
        Crop_Poly_rasterized_Data.append(rasterized_mean[i]["properties"])
        i=i+1
Avg_rasterized_Indices = pd.DataFrame(Crop_Poly_rasterized_Data)
Avg_rasterized_Indices.rename({'id': 'ID', 'count':'Count_SHP'}, axis=1, inplace=True)
ID_FINAL_COUNT = Avg_rasterized_Indices[["ID", "Count_SHP"]]



# Extracting one vegetaion index (VI) per plot
# General syntex format:

#  Calculating statistics (mean, median, sum, and count)
#  VI_mean = zonal_stats(crop_gdf, NDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)

#  Extracting ststistics per plot to save into the geodatabase with plot ID's created in the previous step
#  Crop_Poly_VI_Data = []
#  i=0
#  while i<len(VI_mean):
#          Crop_Poly_VI_Data.append(VI_mean[i]["properties"])
#          i=i+1

#  Saving the extracted indices into the geodatabase
#  Avg_VI_Indices = pd.DataFrame(Crop_Poly_VI_Data)


#  Renaming the deafult headers names to names of interest
#  Avg_VI_Indices.rename({'id': 'ID', 'mean': 'VI_mean', 'median':'VI_median', 'sum':'VI_sum', 'count':'VI_count'}, axis=1, inplace=True)


#  Creating final database
#  VI_FINAL = Avg_VI_Indices[["VI_mean", "VI_median", "VI_sum", "VI_count"]]





# Extracting NDVI per plot
NDVI_mean = zonal_stats(crop_gdf, NDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_NDVI_Data = []
i=0
while i<len(NDVI_mean):
        Crop_Poly_NDVI_Data.append(NDVI_mean[i]["properties"])
        i=i+1
Avg_NDVI_Indices = pd.DataFrame(Crop_Poly_NDVI_Data)
Avg_NDVI_Indices.rename({'id': 'ID', 'mean': 'NDVI_mean', 'median':'NDVI_median', 'sum':'NDVI_sum', 'count':'NDVI_count'}, axis=1, inplace=True)
NDVI_FINAL = Avg_NDVI_Indices[["NDVI_mean", "NDVI_median", "NDVI_sum", "NDVI_count"]]



# Extracting RENDVI per plot
RENDVI_mean = zonal_stats(crop_gdf, RENDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_RENDVI_Data = []
i=0
while i<len(RENDVI_mean):
        Crop_Poly_RENDVI_Data.append(RENDVI_mean[i]["properties"])
        i=i+1
Avg_RENDVI_Indices = pd.DataFrame(Crop_Poly_RENDVI_Data)
Avg_RENDVI_Indices.rename({'id': 'ID', 'mean': 'RENDVI_mean', 'median':'RENDVI_median', 'sum':'RENDVI_sum', 'count':'RENDVI_count'}, axis=1, inplace=True)
RENDVI_FINAL = Avg_RENDVI_Indices[["RENDVI_mean", "RENDVI_median", "RENDVI_sum", "RENDVI_count"]]



# Extracting GNDVI per plot
GNDVI_mean = zonal_stats(crop_gdf, GNDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_GNDVI_Data = []
i=0
while i<len(GNDVI_mean):
        Crop_Poly_GNDVI_Data.append(GNDVI_mean[i]["properties"])
        i=i+1
Avg_GNDVI_Indices = pd.DataFrame(Crop_Poly_GNDVI_Data)
Avg_GNDVI_Indices.rename({'id': 'ID', 'mean': 'GNDVI_mean', 'median':'GNDVI_median', 'sum':'GNDVI_sum', 'count':'GNDVI_count'}, axis=1, inplace=True)
GNDVI_FINAL = Avg_GNDVI_Indices[["GNDVI_mean", "GNDVI_median", "GNDVI_sum", "GNDVI_count"]]



# Extracting ENDVI per plot
ENDVI_mean = zonal_stats(crop_gdf, ENDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_ENDVI_Data = []
i=0
while i<len(ENDVI_mean):
        Crop_Poly_ENDVI_Data.append(ENDVI_mean[i]["properties"])
        i=i+1
Avg_ENDVI_Indices = pd.DataFrame(Crop_Poly_ENDVI_Data)
Avg_ENDVI_Indices.rename({'id': 'ID', 'mean': 'ENDVI_mean', 'median':'ENDVI_median', 'sum':'ENDVI_sum', 'count':'ENDVI_count'}, axis=1, inplace=True)
ENDVI_FINAL = Avg_ENDVI_Indices[["ENDVI_mean", "ENDVI_median", "ENDVI_sum", "ENDVI_count"]]



# Extracting NDRE per plot
NDRE_mean = zonal_stats(crop_gdf, NDRE, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_NDRE_Data = []
i=0
while i<len(NDRE_mean):
        Crop_Poly_NDRE_Data.append(NDRE_mean[i]["properties"])
        i=i+1
Avg_NDRE_Indices = pd.DataFrame(Crop_Poly_NDRE_Data)
Avg_NDRE_Indices.rename({'id': 'ID', 'mean': 'NDRE_mean', 'median':'NDRE_median', 'sum':'NDRE_sum', 'count':'NDRE_count'}, axis=1, inplace=True)
NDRE_FINAL = Avg_NDRE_Indices[["NDRE_mean", "NDRE_median", "NDRE_sum", "NDRE_count"]]



# Extracting NNIR per plot
NNIR_mean = zonal_stats(crop_gdf, NNIR, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_NNIR_Data = []
i=0
while i<len(NNIR_mean):
        Crop_Poly_NNIR_Data.append(NNIR_mean[i]["properties"])
        i=i+1
Avg_NNIR_Indices = pd.DataFrame(Crop_Poly_NNIR_Data)
Avg_NNIR_Indices.rename({'id': 'ID', 'mean': 'NNIR_mean', 'median':'NNIR_median', 'sum':'NNIR_sum', 'count':'NNIR_count'}, axis=1, inplace=True)
NNIR_FINAL = Avg_NNIR_Indices[["NNIR_mean", "NNIR_median", "NNIR_sum", "NNIR_count"]]



# Extracting MCARI per plot
MCARI_mean = zonal_stats(crop_gdf, MCARI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_MCARI_Data = []
i=0
while i<len(MCARI_mean):
        Crop_Poly_MCARI_Data.append(MCARI_mean[i]["properties"])
        i=i+1
Avg_MCARI_Indices = pd.DataFrame(Crop_Poly_MCARI_Data)
Avg_MCARI_Indices.rename({'id': 'ID', 'mean': 'MCARI_mean', 'median':'MCARI_median', 'sum':'MCARI_sum', 'count':'MCARI_count'}, axis=1, inplace=True)
MCARI_FINAL = Avg_MCARI_Indices[["MCARI_mean", "MCARI_median", "MCARI_sum", "MCARI_count"]]



# Extracting SIPI per plot
SIPI_mean = zonal_stats(crop_gdf, SIPI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_SIPI_Data = []
i=0
while i<len(SIPI_mean):
        Crop_Poly_SIPI_Data.append(SIPI_mean[i]["properties"])
        i=i+1
Avg_SIPI_Indices = pd.DataFrame(Crop_Poly_SIPI_Data)
Avg_SIPI_Indices.rename({'id': 'ID', 'mean': 'SIPI_mean', 'median':'SIPI_median', 'sum':'SIPI_sum', 'count':'SIPI_count'}, axis=1, inplace=True)
SIPI_FINAL = Avg_SIPI_Indices[["SIPI_mean", "SIPI_median", "SIPI_sum", "SIPI_count"]]



# Extracting NGRDI per plot
NGRDI_mean = zonal_stats(crop_gdf, NGRDI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_NGRDI_Data = []
i=0
while i<len(NGRDI_mean):
        Crop_Poly_NGRDI_Data.append(NGRDI_mean[i]["properties"])
        i=i+1
Avg_NGRDI_Indices = pd.DataFrame(Crop_Poly_NGRDI_Data)
Avg_NGRDI_Indices.rename({'id': 'ID', 'mean': 'NGRDI_mean', 'median':'NGRDI_median', 'sum':'NGRDI_sum', 'count':'NGRDI_count'}, axis=1, inplace=True)
NGRDI_FINAL = Avg_NGRDI_Indices[["NGRDI_mean", "NGRDI_median", "NGRDI_sum", "NGRDI_count"]]



# Extracting NLI per plot
NLI_mean = zonal_stats(crop_gdf, NLI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_NLI_Data = []
i=0
while i<len(NLI_mean):
        Crop_Poly_NLI_Data.append(NLI_mean[i]["properties"])
        i=i+1
Avg_NLI_Indices = pd.DataFrame(Crop_Poly_NLI_Data)
Avg_NLI_Indices.rename({'id': 'ID', 'mean': 'NLI_mean', 'median':'NLI_median', 'sum':'NLI_sum', 'count':'NLI_count'}, axis=1, inplace=True)
NLI_FINAL = Avg_NLI_Indices[["NLI_mean", "NLI_median", "NLI_sum", "NLI_count"]]



# Extracting SR per plot
SR_mean = zonal_stats(crop_gdf, SR, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_SR_Data = []
i=0
while i<len(SR_mean):
        Crop_Poly_SR_Data.append(SR_mean[i]["properties"])
        i=i+1
Avg_SR_Indices = pd.DataFrame(Crop_Poly_SR_Data)
Avg_SR_Indices.rename({'id': 'ID', 'mean': 'SR_mean', 'median':'SR_median', 'sum':'SR_sum', 'count':'SR_count'}, axis=1, inplace=True)
SR_FINAL = Avg_SR_Indices[["SR_mean", "SR_median", "SR_sum", "SR_count"]]



# Extracting DVI per plot
DVI_mean = zonal_stats(crop_gdf, DVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_DVI_Data = []
i=0
while i<len(DVI_mean):
        Crop_Poly_DVI_Data.append(DVI_mean[i]["properties"])
        i=i+1
Avg_DVI_Indices = pd.DataFrame(Crop_Poly_DVI_Data)
Avg_DVI_Indices.rename({'id': 'ID', 'mean': 'DVI_mean', 'median':'DVI_median', 'sum':'DVI_sum', 'count':'DVI_count'}, axis=1, inplace=True)
DVI_FINAL = Avg_DVI_Indices[["DVI_mean", "DVI_median", "DVI_sum", "DVI_count"]]



# Extracting RDVI per plot
RDVI_mean = zonal_stats(crop_gdf, RDVI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_RDVI_Data = []
i=0
while i<len(RDVI_mean):
        Crop_Poly_RDVI_Data.append(RDVI_mean[i]["properties"])
        i=i+1
Avg_RDVI_Indices = pd.DataFrame(Crop_Poly_RDVI_Data)
Avg_RDVI_Indices.rename({'id': 'ID', 'mean': 'RDVI_mean', 'median':'RDVI_median', 'sum':'RDVI_sum', 'count':'RDVI_count'}, axis=1, inplace=True)
RDVI_FINAL = Avg_RDVI_Indices[["RDVI_mean", "RDVI_median", "RDVI_sum", "RDVI_count"]]



# Extracting MDD per plot
MDD_mean = zonal_stats(crop_gdf, MDD, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_MDD_Data = []
i=0
while i<len(MDD_mean):
        Crop_Poly_MDD_Data.append(MDD_mean[i]["properties"])
        i=i+1
Avg_MDD_Indices = pd.DataFrame(Crop_Poly_MDD_Data)
Avg_MDD_Indices.rename({'id': 'ID', 'mean': 'MDD_mean', 'median':'MDD_median', 'sum':'MDD_sum', 'count':'MDD_count'}, axis=1, inplace=True)
MDD_FINAL = Avg_MDD_Indices[["MDD_mean", "MDD_median", "MDD_sum", "MDD_count"]]



# Extracting MARI per plot
MARI_mean = zonal_stats(crop_gdf, MARI, affine=affine, geojson_out=True,stats=['mean', 'median', 'sum', 'count'], nodata = -9999)
Crop_Poly_MARI_Data = []
i=0
while i<len(MARI_mean):
        Crop_Poly_MARI_Data.append(MARI_mean[i]["properties"])
        i=i+1
Avg_MARI_Indices = pd.DataFrame(Crop_Poly_MARI_Data)
Avg_MARI_Indices.rename({'id': 'ID', 'mean': 'MARI_mean', 'median':'MARI_median', 'sum':'MARI_sum', 'count':'MARI_count'}, axis=1, inplace=True)
MARI_FINAL = Avg_MARI_Indices[["MARI_mean", "MARI_median", "MARI_sum", "MARI_count"]]


# In[ ]:





# In[ ]:


# MAKING DATABSE OF VEGETATION INDICES


# In[ ]:


# Combining all collected information into one databse
FINAL_VI2 = pd.concat([ID_FINAL_COUNT, Red_FINAL, Green_FINAL, Blue_FINAL, 
                       NIR_FINAL, RE_FINAL, NDVI_FINAL, RENDVI_FINAL,
                       GNDVI_FINAL, ENDVI_FINAL, NDRE_FINAL, NNIR_FINAL,
                       MCARI_FINAL, SIPI_FINAL, NGRDI_FINAL, NLI_FINAL,
                       SR_FINAL, DVI_FINAL, RDVI_FINAL, MDD_FINAL, MARI_FINAL], axis=1)


# In[ ]:





# In[ ]:


# Saving final databse as CSV
FINAL_VI2.to_csv("your file path/CSV file name.csv")

