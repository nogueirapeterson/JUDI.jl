# Options structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: May 2017
#

export Options, subsample

# Object for velocity/slowness models
mutable struct Options
    space_order::Integer
    free_surface::Bool
    abc_type::Bool
    limit_m::Bool
    buffer_size::Real
    save_data_to_disk::Bool
    file_path::String
    file_name::String
    sum_padding::Bool
    optimal_checkpointing::Bool
    num_checkpoints::Union{Integer, Nothing}
    checkpoints_maxmem::Union{Real, Nothing}
    frequencies::Array
    isic::Bool
    subsampling_factor::Integer
    dft_subsampling_factor::Integer
    return_array::Bool
    dt_comp::Union{Real, Nothing}
    f0::Real
end

"""
    Options
        space_order::Integer
        free_surface::Bool
        abc_type::Bool
        limit_m::Bool
        buffer_size::Real
        save_rate::Real
        save_data_to_disk::Bool
        file_path::String
        file_name::String
        sum_padding::Bool
        optimal_checkpointing::Bool
        num_checkpoints::Integer
        checkpoints_maxmem::Real
	    frequencies::Array
	    isic::Bool
        subsampling_factor::Integer
	    dft_subsampling_factor::Integer
        return_array::Bool
        dt_comp::Real
        f0::Real



Options structure for seismic modeling.

`space_order`: finite difference space order for wave equation (default is 8, needs to be multiple of 4)

`free_surface`: set to `true` to enable a free surface boundary condition.

`abc_type`: whether the dampening is a mask or layer, (mask => true) inside the domain and decreases in the layer, (mask => false) inside the domain and increase in the layer.

`limit_m`: for 3D modeling, limit modeling domain to area with receivers (saves memory)

`buffer_size`: if `limit_m=true`, define buffer area on each side of modeling domain (in meters)

`save_data_to_disk`: if `true`, saves shot records as separate SEG-Y files

`file_path`: path to directory where data is saved

`file_name`: shot records will be saved as specified file name plus its source coordinates

`sum_padding`: when removing the padding area of the gradient, sum into boundary rows/columns for true adjoints

`optimal_checkpointing`: instead of saving the forward wavefield, recompute it using optimal checkpointing

`num_checkpoints`: number of checkpoints. If not supplied, is set to log(num_timesteps).

`checkpoints_maxmem`: maximum amount of memory that can be allocated for checkpoints (MB)

`frequencies`: calculate the FWI/LS-RTM gradient in the frequency domain for a given set of frequencies

`subsampling_factor`: compute forward wavefield on a time axis that is reduced by a given factor (default is 1)

`dft_subsampling_factor`: compute on-the-fly DFTs on a time axis that is reduced by a given factor (default is 1)

`isic`: use linearized inverse scattering imaging condition

`return_array`: return data from nonlinear/linear modeling as a plain Julia array.

`dt_comp`: overwrite automatically computed computational time step with this value.

`f0`: define peak frequency.

Constructor
==========

All arguments are optional keyword arguments with the following default values:

    Options(;space_order=8, free_surface=false,
            limit_m=false, abc_type=false, buffer_size=1e3,
            save_data_to_disk=false, file_path="",
            file_name="shot", sum_padding=false,
            optimal_checkpointing=false,
            num_checkpoints=nothing, checkpoints_maxmem=nothing,
            frequencies=[], isic=false,
            subsampling_factor=1, dft_subsampling_factor=1, return_array=false,
            dt_comp=nothing, f0=0.015f0)

"""
Options(;space_order=8,
		 free_surface=false,
         abc_type=false,
         limit_m=false,
		 buffer_size=1e3,
		 save_data_to_disk=false,
		 file_path="",
		 file_name="shot",
         sum_padding=false,
		 optimal_checkpointing=false,
		 num_checkpoints=nothing,
		 checkpoints_maxmem=nothing,
		 frequencies=[],
		 isic=false,
		 subsampling_factor=1,
		 dft_subsampling_factor=1,
         return_array=false,
         dt_comp=nothing,
         f0=0.015f0) =
		 Options(space_order,
		 		 free_surface,
                 abc_type,
		         limit_m,
				 buffer_size,
				 save_data_to_disk,
				 file_path,
				 file_name,
				 sum_padding,
				 optimal_checkpointing,
				 num_checkpoints,
				 checkpoints_maxmem,
				 frequencies,
				 isic,
				 subsampling_factor,
				 dft_subsampling_factor,
                 return_array,
                 dt_comp,
                 f0)

function subsample(options::Options, srcnum)
    if isempty(options.frequencies)
        return options
    else
        opt_out = deepcopy(options)
        floc = options.frequencies[srcnum]
        typeof(floc) <: Array && (opt_out.frequencies = floc)
        return opt_out
    end
end
