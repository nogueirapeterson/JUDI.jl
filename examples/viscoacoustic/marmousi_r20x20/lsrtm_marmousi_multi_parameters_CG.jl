# Multiparameter viscoacoustic least-squeare reverse time migration (M-QLSRTM) using conjugate gradient (CG) method
# Author: nogueirapeterson@gmail.com
# Date: April 2022
#
# Warning: The examples requires ~40 GB of memory per shot if used without optimal checkpointing.
#

using ClusterManagers, Distributed, ArgParse; addprocs_slurm(5)

### Process command line args
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--time_order", "-t"
            help = "Time order"
            arg_type = Int
            default = 2
        "--velmod_dir"
            help = "Velocity model directory"
            default = ""
        "--dobs_dir"
            help = "Observed data directory"
            default = ""
        "--output_dir"
            help = "Outpur data directory"
            default = ""
    end
    return parse_args(s)
end

@everywhere using JUDI, HDF5, PyPlot, JLD, SegyIO, Random, Statistics, LinearAlgebra, Logging
parsed_args = parse_commandline()

time_order = parsed_args["time_order"]
velmod_dir = parsed_args["velmod_dir"]
dobs_dir = parsed_args["dobs_dir"]
output_dir = parsed_args["output_dir"]

# Load migration velocity model
# n, d, o, m0 = read(h5open(velmod_dir, "r"), "n", "d", "o", "m0")
n1, d1, o1, m, m0, rho, rho0, qp, qp0 = read(h5open(velmod_dir, "r"), "n", "d", "o", "m", "m0", "rho", "rho0", "qp", "qp0")

# Set up model structure
# model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)
n=(n1[1], n1[2]);
d=(d1[1], d1[2]);
o=(o1[1], o1[2]);

println("spacing: ", d)
println("n: ", n)

model0 = Model(n, d, o, m0, rho0, qp0);

# Load data
block = segy_read(dobs_dir)
d_lin = judiVector(block)   # linearized observed data

# Set up wavelet
f0 = 0.015
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], f0)    # 15 Hz wavelet
q = judiVector(src_geometry, wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry, d_lin.geometry, model0)  # no. of computational time steps
info = Info(prod(model0.n), d_lin.nsrc, ntComp)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=false, multi_parameters=(1,1), f0=f0)  # ~40 GB of memory per source w/o checkpointing
M = judiModeling(info, model0, q.geometry, d_lin.geometry; options=opt)
J = judiJacobian(M, q)

# Stochastic gradient
x = (zeros(Float32, info.n), zeros(Float32, info.n))
# batchsize = 50
niter = 20
fval = zeros(Float32, niter)

global beta = 0f0

gp = (zeros(Float32, info.n), zeros(Float32, info.n))
dk = (zeros(Float32, info.n), zeros(Float32, info.n))
dkp = (zeros(Float32, info.n), zeros(Float32, info.n))

# Main loop
for j = 1: niter
    println("Iteration: ", j)

    # Compute residual and gradient
    r = J*x - d_lin
    g = J'*r

    g = (reshape(g[1], model0.n), reshape(g[2], model0.n))
    g[1][:, 1:50] .= 0f0; g[2][:, 1:50] .= 0f0

    # Step size and update variable
    fval[j] = .5f0*norm(r)^2

    global dk = g .+ beta .* dkp

    if j == 1

       Lg = J*g

       alfa = dot(g,g)/dot(Lg, Lg)

    else

        dk = (reshape(dk[1], model0.n), reshape(dk[2], model0.n))
	    dk[1][:, 1:50] .= 0f0; dk[2][:, 1:50] .= 0f0
        Ldk = J*dk

	global beta = dot(g, g) / dot(gp, gp)

	alfa = dot(dk, g) / dot(Ldk, Ldk)

    end

    global gp = g
    global dkp = dk

    println("fval: ", fval[j])
    println("alfa: ", alfa)

    global x = x .- alfa .* dk
    x = (reshape(x[1], model0.n), reshape(x[2], model0.n))
    x[1][:, 1:50] .= 0f0; x[2][:, 1:50] .= 0f0

    if j == 1
        x_m = x[1]
        x_m_reshape = reshape(x_m, model0.n)
        imagem_dir = joinpath(output_dir, "xkappa_inicial_CG_r20x20.bin")
        write(imagem_dir, htol.(transpose(x_m_reshape)));

        x_tau = x[2]
        x_tau_reshape = reshape(x_tau, model0.n)
        imagem_dir = joinpath(output_dir, "xtau_inicial_CG_r20x20.bin")
        write(imagem_dir, htol.(transpose(x_tau_reshape)));
    end
end

fhistory_dir = joinpath(output_dir, "fhistory_viscoacoustic_multi_parameter_CG_r20x20.bin")
write(fhistory_dir, htol.(transpose(fval)));

x_m = x[1]
x_m_reshape = reshape(x_m, model0.n)
imagem_dir = joinpath(output_dir, "xkappa_final_CG_r20x20.bin")
write(imagem_dir, htol.(transpose(x_m_reshape)));

x_tau = x[2]
x_tau_reshape = reshape(x_tau, model0.n)
imagem_dir = joinpath(output_dir, "xtau_final_CG_r20x20.bin")
write(imagem_dir, htol.(transpose(x_tau_reshape)));