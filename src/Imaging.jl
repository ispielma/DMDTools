"""
Imaging

A module to model imaging of cold atom systems.  This is ported from a python file to solve the same
task.
"""
module Imaging

import ..Atoms as AT
import ..Atoms: ThomasFermiDensity

import Parameters
import AbstractFFTs as AFFTs
import FFTW
import Distributions as Dist

import QGas.NumericalTools.ArrayDimensions as AD
import QGas.NumericalTools.ImageProcessing as IP
import QGas.AtomicConstants: c, ħ

@Parameters.with_kw struct Config{_ndims}

    sensor_size::NTuple{_ndims, Int64} = (1024,1024)
    ndims::Int32 = length(sensor_size)

    atoms_config::AT.AtomsConfig = AT.AtomsConfig(ndims=ndims)


    debug::Bool = false

    # Imaging constants

    dt::Float64 = 20e-6 # imaging pulse duration
    NA::Vector{Float64} = [0.32, 0.24]
    magnification::Float64 = 36.3
    pixel_size::Float64 = 13e-6

    atom_shot_noise::Bool = true

    PCI_phase::Float64 = -π/2
    QE::Float64 = 0.6 # incude all losses including the quantum efficiency
    EM_noise::Float64 = √2 # added noise from EM stage
    defocus::Vector{Float64} = [-10e-6, 160e-6]
    aberration::Dict{String, Float64} = Dict(
        "2_0" => 27.86,
        "0_2" => -489,
        "1_1" => -40.9,
        "2_2" => -261,
        "0_3" => 0,
        "3_1" => 1695,
        "1_3" => 2694,
        "4_0" => -2012,
        "0_4" => -3402,
        )

    # k-dependent loss terms for example from imperfect AR coatings.
    amplitude::Dict{String, Float64} = Dict(
        "2_0" => -5,
        "0_2" => -5,
        )

    # Use constrtive geometry to define the aperture function, in units of k_0
    aperture_geometry::Tuple = (
        ("fill", (1,)), # fill with 1's
        ("disk",(1, 0, 0, 0.35)), # sign, x, y, rad
        ("rectangle", (0, -0.4, 0.2, 0.4, 0.5)), # sign, x1, y1, x2, y2
        ("rectangle", (0, 0.07, 0.1, 0.24, 0.5)),
        )

    # Simulation options
    photon_shot_noise::String = "atoms" # Or "both" or "none"

    read_noise::Int32 = 5 # in units of photons

    N_ph::Int32 = 737 # gives I/Isat = 2
    waist::Vector{Float64} = [150e-6, 150e-6] # Gaussian beam waists

    #
    # Derived parameters that need computation to evaluate (all set to empty defaults)
    # 

    theta_max::Vector{Float64} = asin.(NA) # maximum acceptance angle of objective lens
    k_NA::Vector{Float64} = atoms_config.k_0 .* NA
    
    dx::Float64 = pixel_size / magnification # demagnified pixel size in object plane
    k_max::Float64 = 2*π / dx # in my fourier basis we will go from -k_max/2 to +k_max/2

    N_sat::Float64 = IntensityToPhotons(atoms_config.I_sat, dt, dx, atoms_config.k_0)
    
    spatial_max::Vector{Float64} = [dx * len for len in sensor_size]

    dk::Vector{Float64} =  2*π ./ spatial_max

    # in this range matrix the -dx makes the array of pixels spatial_max - dx in size.  This is because
    # we are measuring from the center of pixels
    x_vals::AD.NDRange = AD.NDRange(Tuple(range(-sm/2, sm/2-dx, ss) for (sm, ss) in zip(spatial_max, sensor_size))) 
    x_grids::NTuple{_ndims, Array{Float64}} = AD.meshgrid(x_vals)

    k_vals::AD.NDRange = AD.NDRange(Tuple(AFFTs.fftfreq(ss, (2*π)/dx) for ss in sensor_size)) 
    k_grids::NTuple{_ndims, Array{Float64}} = AD.meshgrid(k_vals)
    k2_grids::NTuple{_ndims, Array{Float64}} = Tuple(kg.^2 for kg in k_grids)
    k_mask::IP.WindowConfig = IP.WindowConfig(
        k_grids, 
        "Tukey", 
        k_NA, 
        [0,0],
        2.0,
        0.1,
        false)

    # These extents are really for using imshow, but I can construct them for any number of dimensions
    x_extent::Vector{Float64} = reduce(vcat,[[minimum(ax), maximum(ax)] for ax in x_vals.ranges])
    k_extent::Vector{Float64} = reduce(vcat,[[minimum(ax), maximum(ax)] for ax in k_vals.ranges])
  
    # introduce common lab units for plots
    k_NA_lab::Vector{Float64} = k_NA .* 1e-6/(2*π)

    x_vals_lab::AD.NDRange = x_vals * 1e6
    x_grids_lab::NTuple{_ndims, Array{Float64}} = x_grids .* 1e6
    x_extent_lab::Vector{Float64} = x_extent .* 1e6
    
    k_vals_lab::AD.NDRange = AD.NDRange(Tuple(
                                       AFFTs.fftshift(AFFTs.fftfreq(ss, 1/(dx*1e6) )) for ss in sensor_size
                                       ))
    k_grids_lab::NTuple{_ndims, Array{Float64}} = k_grids .* (1e-6/(2*pi))

    k_extent_lab::Vector{Float64} = k_extent .* (1e-6/(2*pi))
end
(cfg::Config)(args...; kwargs...) = Config(cfg; kwargs...) # make a modified version

#=
##     ## ######## ##       ########  ######## ########     ##     ## ######## ######## ##     ##  #######  ########   ######
##     ## ##       ##       ##     ## ##       ##     ##    ###   ### ##          ##    ##     ## ##     ## ##     ## ##    ##
##     ## ##       ##       ##     ## ##       ##     ##    #### #### ##          ##    ##     ## ##     ## ##     ## ##
######### ######   ##       ########  ######   ########     ## ### ## ######      ##    ######### ##     ## ##     ##  ######
##     ## ##       ##       ##        ##       ##   ##      ##     ## ##          ##    ##     ## ##     ## ##     ##       ##
##     ## ##       ##       ##        ##       ##    ##     ##     ## ##          ##    ##     ## ##     ## ##     ## ##    ##
##     ## ######## ######## ##        ######## ##     ##    ##     ## ########    ##    ##     ##  #######  ########   ######
=#

ThomasFermiDensity(cfg::Config) = ThomasFermiDensity(cfg.x_grids, cfg.dx, cfg.atoms_config)
ThomasFermiNumber(cfg::Config) = ThomasFermiDensity(cfg::Config) ./ cfg.dx^cfg.ndims

#=
#### ##     ##    ###     ######   #### ##    ##  ######      ######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
 ##  ###   ###   ## ##   ##    ##   ##  ###   ## ##    ##     ##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
 ##  #### ####  ##   ##  ##         ##  ####  ## ##           ##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
 ##  ## ### ## ##     ## ##   ####  ##  ## ## ## ##   ####    ######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
 ##  ##     ## ######### ##    ##   ##  ##  #### ##    ##     ##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
 ##  ##     ## ##     ## ##    ##   ##  ##   ### ##    ##     ##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
#### ##     ## ##     ##  ######   #### ##    ##  ######      ##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######
=#

"""
I intensity in mW / cm^2

dx: pixel size
dt: imaging pulse duration
"""
function IntensityToPhotons(I, dt, dx, k_0)

    # convert to energy per pulse
    Energy = (I * (1e-3) * (1e2)^2) * dt * dx^2
    
    # convert to photons
    return Energy / (c*k_0*ħ)
end
IntensityToPhotons(I, cfg::Config) = IntensityToPhotons(I, cfg.dt, cfg.dx, cfg.atoms_config.k_0)

"""
BuildField: Build a simulated image

N : atom NUMBER per pixel, not density.

return the field after and before the atoms
"""
function BuildField(N, cfg::Config)

    # Update atomic number
    if cfg.atom_shot_noise
        ρ = [Float64(Dist.rand(Dist.Poisson(n))) for n in N]
    else
        ρ = [Float64(n) for n in N]
    end
    ρ ./= cfg.dx^cfg.ndims

    # Compute OD from the density
    OD = AT.ODFromDensity(ρ, cfg.atoms_config)
    
    E_minus = fill(convert(ComplexF64, sqrt(cfg.N_ph)), size(OD))

    for j in 1:cfg.ndims
        E_minus .*= exp.( -(cfg.x_grids[j]./cfg.waist[j]).^2 )
    end
    
    E_plus = E_minus .* exp.(-ComplexF64(0.5, cfg.atoms_config.delta).*OD)

    return E_plus, E_minus
end

function DetectImage(field, photon_shot_noise, cfg::Config)
    
    if photon_shot_noise == true
        QEeff = cfg.QE/cfg.EM_noise
        # generate noise for the reduced signal at the detector, but keep working in object plane units.
        image = [convert(Float64, Dist.rand(Dist.Poisson( QEeff .* abs(f)^2))) for f in field]
        image ./= QEeff # So the image is in units of photons
    else
        image = convert.(Float64, abs.(field).^2)
    end
    
    if cfg.read_noise > 0.0
        image .+= Dist.rand(Dist.Gaussian(cfg.read_noise), cfg.sensor_size )
    end

    return image
end

""" 
Return the optical depth from absorption imaging
"""
function AI(E_plus, E_minus, cfg::Config)
 
    if cfg.photon_shot_noise == "atoms"
        N_plus = DetectImage(E_plus, true, cfg)
        N_minus = DetectImage(E_minus, false, cfg)
    elseif cfg.photon_shot_noise == "both"
        N_plus = DetectImage(E_plus, true, cfg)
        N_minus = DetectImage(E_minus, true, cfg)
    else # neither
        N_plus = DetectImage(E_plus, false, cfg)
        N_minus = DetectImage(E_minus, false, cfg)
    end

    return -log.( N_plus ./ N_minus)
end

"""
Return the phase contrast imaging signal, in the large detuning limit

Returns OD if requested
"""
function PCI(E_plus, E_minus, cfg::Config; returnOD=false)
    
    E_minus_pci = E_minus .* exp(complex(0, cfg.PCI_phase))
    E_pci = (E_plus .- E_minus) .+ E_minus_pci # replace the unscattered probe with the phase shifted probe
    
    if cfg.photon_shot_noise == "atoms"
        N_pci = DetectImage(E_pci, true, cfg)
        N_minus = DetectImage(E_minus_pci, false, cfg)
    elseif cfg.photon_shot_noise == "both"
        N_pci = DetectImage(E_pci, true, cfg)
        N_minus = DetectImage(E_minus_pci, true, cfg)
    else
        N_pci = DetectImage(E_pci, false, cfg)
        N_minus = DetectImage(E_minus_pci, false, cfg)
    end

    signal = (1 .- N_pci ./ N_minus)

    return signal
end

#=
########  ##     ## ########  #### ##          ######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##     ## ##     ## ##     ##  ##  ##          ##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##     ## ##     ## ##     ##  ##  ##          ##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
########  ##     ## ########   ##  ##          ######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##        ##     ## ##         ##  ##          ##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##        ##     ## ##         ##  ##          ##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##         #######  ##        #### ########    ##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######
=#

# Add functions for amplitude as well.
# I think I am going to write two functions.  One for the hard-clipping
# of the aperture and the other for modest smoothing.
"""
Create a mask/window using the rules define in rules
    
assumes that we are working in 2D

example syntax:
    ['fill', (1,)], # fill with 1's'
    ['disk', (1, 0, 0, 0.32)], # sign, x, y, rad
    ['rectangle', (0, -0.32, 0.24, 0.32, 0.5)] # sign, x1, y1, x2, y2
    
symmetrize: symmetrize the final mask
"""
function GeometryMask(grids, rules; symmetrize=false)
    

    mask::Array{Int8} = zeros(Int8, size(grids[1]))


    for rule in rules
        shape = rule[1]
        parms = rule[2]
        
        # Init options
        if shape == "fill"
            fill!(mask, parms[1])
        else
            # Geometry options
            if shape == "disk"
                temp_mask = ( (grids[1] .- parms[2]).^2 .+ (grids[2] .- parms[3]).^2) .< parms[4]^2
            elseif shape == "rectangle"
                temp_mask = (grids[1] .> parms[2]) .& (grids[1] .< parms[4]) .& (grids[2] .> parms[3]) .& (grids[2] .< parms[5])
            else
                error("Invalid shape: $(shape)")
            end
                
            if parms[1] == 0
                temp_mask = 1 .- temp_mask
            end

            mask .*= temp_mask
        end
    end
    
    if symmetrize == "Extend"
        mask .+= GeometryMask(-1.0.*grids, rules; symmetrize=False)
        mask = Int8(mask .> 0)
    elseif symmetrize == "Contract"
        mask .*= GeometryMask(-1.0.*grids, rules; symmetrize=False)
    end
    
    return mask
end

"""
PolynomialByOrder

Evaluates a polynomial on a N-D grid based on the coefficients

grids : meshgrid like object of points

coefs: Dict such as for 2D

Dict(
        "2_0" => 27.86,
        "0_2" => -489,
        "1_1" => -40.9,
        "2_2" => -261,
        "0_3" => 0,
        "3_1" => 1695,
        "1_3" => 2694,
        "4_0" => -2012,
        "0_4" => -3402,
        )

"""
function PolynomialByOrder(grids, coefs)

    dim = length(grids)
    poly = zeros(Float64, size(grids[1]))
    temp = zeros(Float64, size(grids[1]))

    for (k, v) in pairs(coefs)
        order = [parse(Int64, i) for i in split(k, "_")]
        
        fill!(temp, v)

        for j in 1:dim
            temp .*= grids[j].^order[j]
        end
        poly .+= temp
    end
            
    return poly
end


"""
Evaluates the magnitude of the gradient of a polynomial on an 
N-D grid based on the coefficients

if dk is passed return the (d_k phi) * dk to give the difference 
"""
function DPolynomialByOrder(grids, coefs, dk)

    dim = length(grids)
    d_poly = Tuple(zeros(Float64, size(grids[1])) for _ in 1:dim)

    temp = zeros(Float64, size(grids[1]))
    
    for (k, v) in pairs(coefs)
        order = [parse(Int64, i) for i in split(k, "_")]

        for i in 1:dim # axes over which we differentiate
            
            if order[i] > 0
                fill!(temp, v*order[i]) # coefficient and factor from derivitive of poly
                for j in 1:dim # polynomial terms
                    temp .*= grids[j].^ (i == j ? order[j]-1 : order[j])
                end
                d_poly[i] .+= temp
            end
            
        end

    end
    
    # Convert to finite differences (I am confused by this code now) and sum
    fill!(temp,  0.0)
    for j in 1:dim
        d_poly[j] .*= dk[j]
        temp .+=d_poly[j].^2
    end

    return sqrt.(temp)
end

"""
Generates the phase variation across the pupil based on a polnomial
expansion.
"""
PupilPhase(cfg::Config) = PolynomialByOrder(cfg.k_grids./cfg.atoms_config.k_0, cfg.aberration) # This is a name left over from python

DPupilPhase(cfg::Config) = DPolynomialByOrder(cfg.k_grids./cfg.atoms_config.k_0, cfg.aberration, cfg.dk ./ cfg.atoms_config.k_0) 

"""
Generates the intensity variation across the pupil based on a polnomial
expansion in the expoent.

returns the pupil amplitude and the k_na window 
"""
function PupilAmplitude(cfg::Config)

    ap_mask = GeometryMask(cfg.k_grids./cfg.atoms_config.k_0, cfg.aperture_geometry)
    gamma = PolynomialByOrder(cfg.k_grids./cfg.atoms_config.k_0, cfg.amplitude)
    
    return (ap_mask .* exp.(gamma), ap_mask)
end

"""
Helper function to convert defocus to aberration.  2D only
"""
DefocusToAberration(defocus, k_0) = Dict("2_0" => -0.5*defocus[1]*k_0, "0_2" => -0.5*defocus[2]*k_0)
DefocusToAberration(cfg::Config) = DefocusToAberration(cfg.defocus, cfg.atoms_config.k_0)

"""
Produces the contrast transfer function for any pupil function
alone.  Uses the machinery of the full pupil function.

Used to include an apodizing function to account for rapid phase variation
Decided to move that to a seperate location to collect all of the
terms that make gamma and beta in one place

aperture indicates if one is to include the apreture in the CTF
"""
function DefocusCTF(cfg::Config; aperture=true, field=false)

    # the OLD function for  fast-phase apodizing function this rejects information 
    # that has wrapped around the grid because of periodic boundary conditions.  
    # I am in addition putting a smooth cutoff in the phase function instead to 
    # deal with this problem for the recovery part of the code.

    # d_xy_phi = -np.einsum('i,ijk->ijk', np.array(defocus), k_grids)/k_0
    # Expression from gaussian integral in Mathematica 
    # Cos(\[Phi]0)/Power(E,((Power(dkx\[Phi],2) + Power(dky\[Phi],2))*Power(\[Delta]k,2))/4.)
    # gamma = -0.25*np.einsum('i,ijk->jk', 1/dk**2, d_xy_phi**2)

    i_phi_p = ComplexF64.(0.0, PupilPhase(cfg)) # i times the phase
    (A_p, ap_mask_p) = PupilAmplitude(cfg)

    # Slow down the phase winding to a max spatial frequency when it is too fast for the grid
    d_phi_kxy = DPupilPhase(cfg)
    i_phi_p ./= sqrt.(1.0 .+ d_phi_kxy.^2 ./ 8.0) 
    
    
    # NOTE: I see that the field version does the amplitude slow-down using the OLD Scheme
    # but the image version uses the new scheme.
    if field
        H = A_p .* exp.(i_phi_p)
    else
        cfg_neg = cfg(k_grids=-1.0 .* cfg.k_grids)
        i_phi_n = ComplexF64.(0.0, PupilPhase(cfg_neg) )
        
        # Slow down the phase winding to a max spatial frequency  when it is too fast for the grid
        d_phi_kxy = DPupilPhase(cfg_neg)
        i_phi_n ./= sqrt.(1.0 .+ d_phi_kxy.^2 ./ 8.0) 

        if aperture
            (A_n, _) = PupilAmplitude(cfg_neg)
            H = 0.5 .* ( A_p .* exp.(i_phi_p ) .+ A_n.*exp.(-i_phi_n) )
        else
            H = 0.5 .* ( exp.(i_phi_p) .+ exp.(-i_phi_n) )
        end
    end

    return H, ap_mask_p
end

"""
Defocus

computes the aberrated density or field
"""

Defocus(ϕ, H) = FFTW.ifft(H.* FFTW.fft(ϕ))
function Defocus(ϕ, cfg::Config; field=false)
            
    (H, _) = DefocusCTF(cfg; field=field)
        
    return Defocus(ϕ, H)
end

end