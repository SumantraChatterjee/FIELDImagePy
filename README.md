# FIELDimagePy: Tutorial & Applications 

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide1.PNG" width="70%" height="70%">
</p>
 
FIELDimagePy is a computer program designed to extract information from a raster image bounded by a shapefile of multiple polygons. FIELDimagePy extracts information within each polygon of the shapefile. This program generates output quite similar to the program FIELDimageR. Although currently the default version of both FIELDimageR and FIELDimagePy programs have been designed to be applicable for agricultural fields separated by different plots, with some slight modifications both programs can be modified to be applicable to many other disciplines of science, such as geography, geophysics, geology, economics, medical research, etc. Other than the fact that FIELDimageR had been written in R-language and that FIELDimagePy has been written in Python-language, the two programs differ significantly in two other important aspects as well, such as  - (i) FIELDimagePy consumes significantly lesser computing time than FIELDimageR, and (ii) to reduce computing time, FIELDimageR aggregates several pixels into one larger pixel (and thus reduces the original resolution), whereas FIELDimagePy does not reduce the original image resolution. Thus, the benefit of FIELDimagePy is that it calculates based on high-resolution images, and at the same time consumes significantly less computing time. 

	The following tutorial has been shown in Jupyter Notebook. However, it can also be used in Spyder or any other platform.


<div id="menu" />
  
## Resources
  
[Step 1 - Intall libraries](#s1)

[Step 2](#s1)

[Contact](#cont)

---------------------------------------------

<div id="s1" />




## Step 1: Intall libraries

> The following libraries need to be installed. Please note that rasterio needs that gdal to be installed as a pre-requisite.

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide2.PNG"
<br />

[Menu](#menu)

<br />

---------------------------------------------

<div id="s2" />
  



## Step 2: Installing libraries

> Once all the aforementioned libraries have been installed, the following libraries and the associated packages needs to be installed.

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide3.PNG"
<br /> 

<br />

---------------------------------------------

<div id="s2" />




## Step 3: Clipping images according to region of interest (ROI)

> Often images are acquired over a larger area than required. If the processes continued with the original images that would increase the computing times unnecessarily. Thus, the images are suggested to be clipped using an area of study. This area of study can be an outermost boundary of the multi-polygon shapefile defining the plots, or a separate shapefile of one polygon defining the region of interest. To implement this step – 
The shapefile of importance is imported as:

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide4.PNG"
<br /> 

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide5.PNG"
<br />  

> Figure above shows the shapefile, defining the plot polygons. For this example, plot polygons of the study region have been used. However, any polygon of interest can be used at this stage.

> The image is imported and clipped according to the shapefile as:

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide6.PNG"
<br />  

> Once the image is clipped according to the shapefile, the clipped image will be used further in the calculation. It is up to the user’s choice whether to save this clipped image at a permanent location or at a temporary location. The following section of the tutorial shows saving at the temporary location:

 <p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide7.PNG"
<br />  
	 
<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide8.PNG"
<br />  
	
> The above figure shows the orhtomosicked images, before (top left) and after (bottom right) clipping with polygon of interest.




<br />

---------------------------------------------

<div id="s2" />
	


## Step 4: Importing multiband raster image and converting each band into separate NumPy arrays
> The next step is to import the raster image and convert it to a NumPy array, and convert digital numbers to reflectance data. In the following two pictures, the first one represents importing a raster image and converting every raster band into NumPy arrays and the second one is converting digital numbers to reflectance. This example is set for a 16-bit image:

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide9.PNG"
<br />  


<br />

---------------------------------------------

<div id="s2" />


## Step 5: Calculating vegetation indices. 
>Here the example has been set for 15 vegetation indices calculations based on multispectral images. Any changes in vegetation indices calculations can be made by changing with any other vegetation indices, or RGB images, of interest:


<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide10B.PNG"
<br />  


<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide11.PNG"
<br /> 


> The above image is showing NDVI calculated from the image. The bright regions represent healthy green vegetation.


<br />

---------------------------------------------

<div id="s2" />


## Step 6: Classifying between soil and crop covers areas (optional step, depends on the aim of the study)
> Crop soil separation can be made with any vegetation index of interest. Since, from this data, Normalized Green Red Difference Index (NGRDI) being greater or less than zero, can be used to precisely classify between soil and crop. Thus, this example is shown based on NGRDI being the vegetation index used to classify between soil and crop. The number -9999 has been given for “null” values. It can also be replaced by “np.NaN” also, which represents NumPy’s default number for nulls. The syntax is – 
VI of Interest = np.where(NGRDI > 0, VI of Interest, -9999)
Thus, if the user uses any other vegetation index than NGRDI, that should replace the NGRDI in above syntax or the equations below:


 <p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide12.PNG"
<br /> 

 
 <p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide13.PNG"
<br /> 


> In the above figure, the original NDVI image (top left) and the NDVI image after clipping for soil exposed areas (bottom right). The white spots within the bottom right image shows the regions removed as exposed soil areas. 


<br />

---------------------------------------------

<div id="s2" />

## Step 7: Importing shapefiles and converting them into geodatabase
> At this stage the plot polygon shapefile needs to be imported as geopandas geodatabase. This is the geodatabase that would be filled with zonal statistics values. The shapefile can be imported as:
 

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide14.PNG"
<br /> 



<br />

---------------------------------------------

<div id="s2" />
	
## Step 8: Calculating total numper of pixels per polygon 
> To calculate the number of pixels within a polygon, the shapefile needs to be rasterized. This step can be done via the following step:


<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide15.PNG"
<br /> 
 

<br />

---------------------------------------------

<div id="s2" />

## Step 9: The next step is extracting zonal statistics from vegetation index
> The following example has been shown for normalized difference vegetation index (NDVI) only. However, this should be altered for any and all other vegetation indices wherever NDVI appears. These zonal statistics, and other, calculates mean, median, sum, and count (the number of non-null pixels within the polygon). For any other statistics, the changes should be made accordingly. The following program calculates the zonal statistics for all the polygons within the shapefile imported in Step 7, and saves the results in the geodatabase created in Step 7:

 
<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide16.PNG"
<br /> 



<br />

---------------------------------------------

<div id="s2" />
	

## Step 10: Combining all vegetation indices togather
> Once all the above steps have been completed, all the zonal statistics for all the vegetation indices can be assembled together into one geodatabase using the following steps:

<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide18.PNG"
<br /> 


<br />

---------------------------------------------

<div id="s2" />
	

## Step 11: Final database
 
>Figure below shows the geopandas geodatabase final database. The header “Count_SHP” represents total number of pixels within each polygons, “NDVI_count” represents the number of pixels with NDVI values after excluding the soil exposed areas. Thus, the ratio between NDVI_count to Count_SHP represents canopy coverage. As seen for NDVI, RENDVI, “mean” represents mean of the vegetation indices, and “median” represents median of the vegetation indices.


<p align="center">
  <img src="https://github.com/SumantraChatterjee/FIELDImagePy_Images/blob/main/Slide19.PNG"
<br /> 


<br />

---------------------------------------------

<div id="s2" />

	
Step 12: The final geodatabase can be exported as CSV file as follows:
 
 
Figure 6. Example of the final exported CSV file


<div id="cont" />

<br />

### Author

> * [Sumantra Chatterjee](https://www.linkedin.com/in/sumantra-chatterjee-01a3154b/)
> * [Seth Murray](https://soilcrop.tamu.edu/people/murray-seth-c/)

<br />

### Acknowledgments

> * [Corn Breeding and Genetics TAMU](https://soilcrop.tamu.edu/people/murray-seth-c/)
> * []()
> * []()

<br />

[Menu](#menu)

<br />
 

