November 2023

# Abstract

This repository is meant to provide some simplified
models for array patterns from conformal geometry arrays.

# Files

- **arrayfactor.py:** contains the AntennaArray class to model the
  geometry of the array and providing functions to compute the far field
  and near field of the array. In both cases assuming the elemental pattern
  is separable from the array factor. In the far field case the *transfer function*  
  matrix is saved to allow, for example, to invert it to obtain a backward projection operator.
- **arraycollimatortest.py:** Makes use of the AntennaArray class to compute the far field of a
  circular section 2-d array and compare it to the far field of a linear uniform array.
  The broadside collimation condition is applied to the excitation phase and some
  strategies to calculate an amplitude tapering are tested to obtain a radiation pattern
  as similar as possible to the one of the linear array.
- **forward_and_backward_projection.py:** Creates a fictitious super-array made by repeating the circular
  section array of the previous file along the longitudinal (aperture) axis.
  A square transfer function matrix is computed choosing the same number of far field points as the
  number of radiators in the array. The transfer function matrix is then inverted to obtain a backward
  projection operator. The excitation vector of the array is forced to be 0 outside the original array.
  The backward projection is utilized to force the far-field of the conformal array to the one of a uniform
  linear array. The success of this method is found to be highly dependent on the choice of element
  spacing period, curvature radius, and number of elements in the array. Moreover, the excitation vector
  amplitude always presents oscillations at the edge elements. Nonetheless the operators here established
  can find use in future Intersection Approach optimisation routines.
    
