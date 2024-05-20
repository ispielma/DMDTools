"""
DMDTools

The main focus of this module is to compute the optical electric field just above the surface of a DMD.  This
will include the contribution the commanded pixel array, the shape of the individual pixels, and the blaze-angle,
i.e. diffraction, effects.
"""
module DMDTools

import Parameters
# import LazyGrids
# import PaddedViews
# import QGas.NumericalTools.ArrayDimensions as AD

"""
DMD

This struct contains the information required to describe the operation of a DMD.  Currently I will
assume a 2D sensor.
"""
struct DMD
    dmd_size::NTuple{2, Int64} 
    pixel_size::Float64 # assume square pixels
    axis_angle::Float64 # axis about which the mirrors tilt in the x-y plane (usually ±π/4)
    tilt_angle::Float64 # Tilt angle of the micromirrors in the on state
end

end