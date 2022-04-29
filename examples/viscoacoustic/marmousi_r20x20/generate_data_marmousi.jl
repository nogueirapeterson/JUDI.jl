# Generate observed Born data for Gas chimney examples
# Author: nogueirapeterson@gmail.com
# Date: March 2022
#

using ClusterManagers, Distributed, ArgParse; addprocs_slurm(5)

# using Distributed, ArgParse; addprocs(4)

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
            help = "Outpur data directory"
            default = ""
    end
    return parse_args(s)
end

@everywhere using JUDI, SegyIO, HDF5, JLD, LinearAlgebra, Statistics
# using JUDI, SegyIO, HDF5, JLD, LinearAlgebra
parsed_args = parse_commandline()

time_order = parsed_args["time_order"]
velmod_dir = parsed_args["velmod_dir"]
dobs_dir   = parsed_args["dobs_dir"]

# Load migration velocity model
n1, d1, o1, m, m0, rho, rho0, qp, qp0 = read(h5open(velmod_dir, "r"), "n", "d", "o", "m", "m0", "rho", "rho0", "qp", "qp0")
n = (n1[1], n1[2]);

# Inverse of bulk modulus
kappa = m .* (1f0 ./ rho)
kappa0 = m0 .* (1f0 ./ rho0)

# Define model perturbation
dkappa = kappa .- kappa0
dkappa = vec(dkappa)

d = ones(Float32, n) .* 1.0f0;

a = 2f0  ./ qp
b = 1f0 ./ qp
c = (d .+ b .^ 2f0) .^ 0.5f0

a0 = 2f0  ./ qp0
b0 = 1f0 ./ qp0
c0 = (d .+ (b0 .^ 2f0)) .^ 0.5f0

tau = a .* (b .+ c)
tau0 = a0 .* (b0 .+ c0)

dtau = tau .- tau0
dtau = vec(dtau)

x = (dkappa, dtau)

# Set up model structure

n=(n1[1], n1[2]);
d=(d1[1], d1[2]);
o=(o1[1], o1[2]);


# rmse_m = rmsd(m, m0; normalize=false)

println("spacing: ", d)
println("n: ", n)
# println("rmse_m: ", rmse_m)

model = Model(n, d, o, m0, rho0, qp0);

nsrc = 50
xsrc = convertToCell(range(model.o[1], stop=model.n[1]*model.d[1], length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(30f0, stop=30f0, length=nsrc))
# receiver sampling and recording time
timeR = 4000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval
# Source sampling and number of time steps
timeS = 4000f0
dtS = 4f0
# nxrec = 369
nxrec = model.n[1] - 2
# xrec = collect(range(model.o[1], stop=model.n[1]*model.d[1], length=nxrec))
xrec = collect(range(model.d[1], stop=(model.n[1]-2)*model.d[1], length=nxrec))
yrec = collect(range(0f0, stop=0f0, length=nxrec))
zrec = collect(range(30f0, stop=30f0, length=nxrec))
# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.015f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

opt = Options(space_order=16, multi_parameters=(1,1), f0=f0)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F*adjoint(Ps), q)

# Born modeling
dobs = J*x
block_out = judiVector_to_SeisBlock(dobs, q; source_depth_key="SourceDepth")
segy_write(dobs_dir, block_out)

dkappa_reshape = reshape(dkappa, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/dkappa_r20x20.bin", htol.(transpose(dkappa_reshape)));

dtau_reshape = reshape(dtau, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/dtau_r20x20.bin", htol.(transpose(dtau_reshape)));

tau_reshape = reshape(tau, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/tau_r20x20.bin", htol.(transpose(tau_reshape)));

tau0_reshape = reshape(tau0, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/tau0_r20x20.bin", htol.(transpose(tau0_reshape)));

kappa_reshape = reshape(kappa, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/kappa_r20x20.bin", htol.(transpose(kappa_reshape)));

kappa0_reshape = reshape(kappa0, model.n)
write("/scratch/projeto-lde/judi-data/marmousi_r20x20/output/kappa0_r20x20.bin", htol.(transpose(kappa0_reshape)));