using LinearAlgebra
using HDF5, SegyIO
velmod_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/v_r10x10.bin";
velmod0_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/v0_r10x10.bin";
rhomod_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/rho_r40x40.bin";
rhomod0_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/rho0_r40x40.bin";
qpmod_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/qp_r40x40.bin";
qpmod0_dir = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/qp0_r40x40.bin";

ndim = 2
nx = 369
nz = 375
ny = 0
if ndim == 2
    shape = (nz, nx);
    shape2 = (nx, nz);
    dimensions = [nx, nz];
    origin = [0., 0.];
    spacing = [25.0, 8.0];
else
    shape = (nx, ny, nz);
    shape2 = (nx, ny, nz);
    dimensions = [nx, ny, nz];
    origin = [0., 0., 0.];
    spacing = [20., 20., 10.];
end

model = zeros(Float32,shape);
model0 = zeros(Float32,shape);

rho = zeros(Float32,shape);
rho0 = zeros(Float32,shape);

qp = zeros(Float32,shape);
qp0 = zeros(Float32,shape);

read!(velmod_dir, model)
read!(velmod0_dir, model0)

read!(rhomod_dir , rho)
read!(rhomod0_dir , rho0)

read!(qpmod_dir , qp)
read!(qpmod0_dir , qp0)

m = (1f0 ./ model).^2
m0 = (1f0 ./ model0).^2

m_tmp = zeros(Float32,shape2);
m0_tmp = zeros(Float32,shape2);

rho_tmp = zeros(Float32,shape2);
rho0_tmp = zeros(Float32,shape2);

qp_tmp = zeros(Float32,shape2);
qp0_tmp = zeros(Float32,shape2);

m_tmp[:, :] = transpose(m)
m0_tmp[:, :] = transpose(m0)

rho_tmp[:, :] = transpose(rho)
rho0_tmp[:, :] = transpose(rho0)

qp_tmp[:, :] = transpose(qp)
qp0_tmp[:, :] = transpose(qp0)

fname = "/home/peterson.santos/lde/JUDI_Nogueira/JUDI.jl/examples/viscoacoustic/models/marmo/marmo_3.h5"
h5open(fname, "w") do file
    write(file, "n", dimensions)  # alternatively, say "@write file A"
    write(file, "d", spacing)
    write(file, "o", origin)
    write(file, "m", m_tmp)
    write(file, "m0", m0_tmp)
    write(file, "rho", rho_tmp)
    write(file, "rho0", rho0_tmp)
    write(file, "qp", qp_tmp)
    write(file, "qp0", qp0_tmp)
end
