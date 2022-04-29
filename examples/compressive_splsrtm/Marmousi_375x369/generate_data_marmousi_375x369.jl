# Generate observed Born data for Marmousi examples
# Author: pwitte.slim@gmail.com
# Date: December 2018
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
            help = "Outpur data directory"
            default = ""
    end
    return parse_args(s)
end

@everywhere using JUDI, SegyIO, HDF5, JLD, LinearAlgebra
parsed_args = parse_commandline()

time_order = parsed_args["time_order"]
velmod_dir = parsed_args["velmod_dir"]
dobs_dir   = parsed_args["dobs_dir"]

# Load migration velocity model
n1, d1, o1, m, m0, rho, rho0, qp, qp0, dm = read(h5open(velmod_dir, "r"), "n", "d", "o", "m", "m0", "rho", "rho0", "qp", "qp0", "dm")

# Set up model structure

dtau = (1f0 ./ qp) .- (1f0 ./ qp0)

n=(n1[1], n1[2]);
d=(d1[1], d1[2]);
o=(o1[1], o1[2]);

model = Model(n, d, o, m0, rho0, qp0);
dm = vec(dm)
dtau = vec(dtau)
x = (dm, dtau)

nsrc = 2
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

opt = Options(multi_parameters=(1,1), f0=f0)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F*adjoint(Ps), q)

# Born modeling
dobs = J*x
block_out = judiVector_to_SeisBlock(dobs, q; source_depth_key="SourceDepth")
segy_write(dobs_dir, block_out)
