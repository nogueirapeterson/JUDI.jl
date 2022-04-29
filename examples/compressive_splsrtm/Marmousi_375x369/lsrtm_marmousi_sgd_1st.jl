# LS-RTM of the 2D Marmousi model using stochastic gradient descent
# Author: pwitte.slim@gmail.com
# Date: December 2018
#
# Warning: The examples requires ~40 GB of memory per shot if used without optimal checkpointing.
#

using ClusterManagers, Distributed, ArgParse; addprocs_slurm(2)

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
n1, d1, o1, m, m0, rho, rho0, qp, qp0 = read(h5open(velmod_dir, "r"), "n", "d", "o", "m", "m0", "rho", "rho0", "qp", "qp0")

# Set up model structure
n=(n1[1], n1[2]);
d=(d1[1], d1[2]);
o=(o1[1], o1[2]);

# rho0 = ones(Float32,n)

model0 = Model(n, d, o, m0, rho0, qp0);

# Load data
block = segy_read(dobs_dir)
d_lin = judiVector(block)   # linearized observed data

# Set up wavelet
f0 = 0.015
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
if time_order == 2
    wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], f0)    # 30 Hz wavelet
else
    wavelet = dgauss_wavelet(src_geometry.t[1], src_geometry.dt[1], f0)    # 30 Hz wavelet
end
q = judiVector(src_geometry, wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry, d_lin.geometry, model0)  # no. of computational time steps
info = Info(prod(model0.n), d_lin.nsrc, ntComp)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=false, time_order=time_order, f0=f0)  # ~40 GB of memory per source w/o checkpointing
M = judiModeling(model0, q.geometry, d_lin.geometry; options=opt)
J = judiJacobian(M, q)

# Stochastic gradient
x = zeros(Float32, prod(model0.n))
# batchsize = 50
niter = 20
fval = zeros(Float32, niter)

# Main loop
for j = 1: niter
    println("Iteration: ", j)

    # Compute residual and gradient
    r = J*x - d_lin
    g = adjoint(J)*r

    g = reshape(g, model0.n)
    g[:, 1:50] .= 0f0;

    # Step size and update variable
    fval[j] = .5f0*norm(r)^2

    println("fval: ", fval[j])
    t = norm(r)^2/norm(g)^2

    global x = x .- t .* g
    x = reshape(x, model0.n)
    x[:, 1:50] .= 0f0;
end

fhistory_dir = joinpath(output_dir, "fhistory_viscoacoustic_sgd_sls_1st.bin")
write(fhistory_dir, htol.(transpose(fval)));

x_m = x
x_m_reshape = reshape(x_m, model0.n)
imagem_dir = joinpath(output_dir, "xm_final_sls_1st.bin")
write(imagem_dir, htol.(transpose(x_m_reshape)));
