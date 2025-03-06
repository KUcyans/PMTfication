## 06032025 IceCube Feature League Rookie Draft
> * **I selected the DOMS based on their adjusted $Q_{adj} = log_{10}Q_{total} -2$**
> * They are based on two different projected dimensions: Horizontal(XY-2D) and Vertical(Z-1D) and some hybrid features.
> * XY oriented features are based on the 2D projection of the DOMs on the XY plane.(Thus, XY position of the strings)
> * as the XY points are projected on the XY plane, each point can have a summarising feature of the string it belongs to.  
> e.g. $\sum_i^{60} Q_{\text{total}, i}$ : the sum of the total charge of the DOMs in the string
> * There are two types of XY features: (1)the first type is based only on the boundary points(`scipy.ConvexHull`) of the selected XY points and (2)the second type takes all the selected XY points into account.
> * Z oriented features are based on the vertical location of DOMs within each string.
> * hybrid features are based on the combination of the XY and Z features.

1. **XY boundary**
   1. `extent_max`: the maximum distance between the boundary points, calculate the distance pairwise and take the maximum
   2. `major_PCA`: the major axis of the PCA of the boundary points
   3. `minor_PCA`: the minor axis of the PCA of the boundary points
   4. `eccentricity_PCA`: 
   $$\sqrt{1-\frac{minor^2}{major^2}}$$
   5. `aspect_contrast_PCA`: 
   $$\frac{major - minor}{major + minor}$$
   6. `cos_sq_extent_PCA_major`: the cosine squared of the angle between the major axis and the max extent vector
   7. `area_ratio`: the ratio of the area of the convex hull to the area of the IceCube horizontal plane

2. **XY collective** : Assuming there are two reasonably spherical clusters of $\Sigma Q$ weighted XY points, calculate the spatial separation of the two clusters. This can be susceptible to the impartial density of DeepCore and non DeepCore DOMs. One metric I can think about using the centroids of thesefor individual DOM position is 

$$ s =\frac{\frac{1}{d_1} - \frac{1}{d_2}}{\frac{1}{d_1} + \frac{1}{d_2}}$$ 
or 
$$ f = \frac{max(\frac{1}{d_1},\frac{1}{d_2})}{\frac{1}{d_1} + \frac{1}{d_2}}$$

   1. `gmm_score`: assuming there are two reasonably spherical clusters of $\Sigma Q$ weighted XY points, `GaussianMixture` calculates the spatial separation of the two clusters
       * separation score =   

       $\frac{d}{\sigma}$ ,   

       where $d$ is the distance between the two centres and $\sigma$ is the average of std of `dom_x` and `dom_y` of the $\Sigma Q$ weighted XY points   
       
       $\sigma = \frac{\sigma_x + \sigma_y}{2}$

   2.`kmeans_score`: `KMeans` calculates two centroids of the $\Sigma Q$ weighted XY points by minimising the sum of the squared distances between the points and the centroids. 
      * separation score = $$\frac{d}{\sigma}$$, where $d$ is the distance between the two centroids and $\sigma$ is the average of std of `dom_x` and `dom_y` of the $\Sigma Q$ weighted XY points $$\sigma = \frac{\sigma_x + \sigma_y}{2}$$
    2. `outer_mass_fraction`: Investigate the distribution of the $\Sigma Q$ weighted distance from each XY point to the $\Sigma Q$ weighted centroid. Dividing the histogram into two: inner and outer. Calculate the sum of the total histogram and the sum of the outer histogram. The ratio of the outer histogram to the total histogram is the `outer_mass_fraction`. This gives the relative 'mass'(i.e. charge) of the outer cluster to the total mass of the clusters.
    $$\frac{\sum_{outer} \text{histogram}}{\sum_{total} \text{histogram}}$$

3. **Z oriented**
   1. `stretch_max`: the maximum `dom_z` difference within the string which has any DOMs in the selected DOMs
   2. `stretch_mean`: the mean `dom_z` difference within the string which has any DOMs in the selected DOMs
   3. `stretch_hiqr`: the high interquartile range of the `dom_z` difference within the string which has any DOMs in the selected DOMs
   $$\frac{Quart_{stretch,0.843} - Quart_{stretch,0,175}}{2}$$

4. **Hybrid**
   1. `product`: the product of the `extent_max` and `stretch_max`
   2. `hypotenuse`: the hypotenuse of the `extent_max` and `stretch_max`
   3. `max_extent_stretch`: max(`extent_max`, `stretch_max`)