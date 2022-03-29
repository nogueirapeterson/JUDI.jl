import numpy as np
from argparse import ArgumentParser
from scipy import ndimage
from devito.logger import info

from sources import RickerSource, Receiver
from models import Model

from propagators import forward, born, gradient


parser = ArgumentParser(description="Adjoint test args")
parser.add_argument("--fs", default=False, action='store_true',
                    help="Test with free surface")
parser.add_argument('-so', dest='space_order', default=8, type=int,
                    help="Spatial discretization order")
parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")
parser.add_argument('-multi_parameters', dest='multi_parameters',
                    default=None, nargs='+', type=int,
                    help="Define whether the approach is single or multi parameters")

args = parser.parse_args()
so = args.space_order
if args.multi_parameters is not None:
    multi_parameters = tuple(args.multi_parameters)
else:
    multi_parameters = args.multi_parameters

# Model
shape = (301, 151)
spacing = (10., 10.)
origin = (0., 0.)
m = np.empty(shape, dtype=np.float32)
m[:] = 1/1.5**2  # Top velocity (background)
m_i = np.linspace(1/1.5**2, 1/4.5**2, args.nlayer)
for i in range(1, args.nlayer):
    m[..., i*int(shape[-1] / args.nlayer):] = m_i[i]  # Bottom velocity

m0 = ndimage.gaussian_filter(m, sigma=10)
m0[m > 1/1.51**2] = m[m > 1/1.51**2]
m0 = ndimage.gaussian_filter(m0, sigma=3)
rho = (m**(-.5)+.5)/2
rho0 = (m0**(-.5)+.5)/2
# dkappa = (m*(1./rho)) - (m0*(1./rho0))
dkappa = m - m0
# Set up model structures
v0 = m0**(-.5)

qp0 = np.empty(shape, dtype=np.float32)
qp = np.empty(shape, dtype=np.float32)
v = m**(-.5)
qp0[:] = 3.516*((v0[:]*1000.)**2.2)*10**(-6)
qp[:] = 3.516*((v[:]*1000.)**2.2)*10**(-6)
dtau = (1./qp) - (1./qp0)

if multi_parameters is not None:
    if(multi_parameters[0] and multi_parameters[1]):
        x = (dkappa, dtau)
    elif multi_parameters[0]:
        x = dkappa
    elif multi_parameters[1]:
        x = dtau
    else:
        print("No value found for multi_parameters")
else:
    x = dkappa

model = Model(shape=shape, origin=origin, spacing=spacing,
              fs=args.fs, m=m0, vp=v0, rho=rho0, qp=qp0, dm=x, space_order=so,
              abc_type=True, multi_parameters=multi_parameters)

# Time axis
t0 = 0.
tn = 2000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.010
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 20.

# Linearized data
print("Forward J")
dD_hat, u0l, _ = born(model, src.coordinates.data, rec_t.coordinates.data,
                      src.data, save=True, f0=f1, multi_parameters=multi_parameters)

# Forward
print("Forward")
_, u0, _ = forward(model, src.coordinates.data, rec_t.coordinates.data,
                   src.data, save=True, f0=f1)

# gradient
print("Adjoint J")
dx_hat, _ = gradient(model, dD_hat, rec_t.coordinates.data, u0, f0=f1,
                     multi_parameters=multi_parameters)

if multi_parameters is None:
    term1 = np.dot(dx_hat[0].data.reshape(-1), model.dkappa.data.reshape(-1))
else:
    if multi_parameters[0] and multi_parameters[1]:
        term1 = np.dot(dx_hat[0].data.reshape(-1), model.dkappa.data.reshape(-1)) + \
            np.dot(dx_hat[1].data.reshape(-1), model.dtau.data.reshape(-1))
    elif multi_parameters[0]:
        term1 = np.dot(dx_hat[0].data.reshape(-1), model.dkappa.data.reshape(-1))
    elif multi_parameters[1]:
        term1 = np.dot(dx_hat[1].data.reshape(-1), model.dtau.data.reshape(-1))
    else:
        print("No value found for multi_parameters")

term2 = np.linalg.norm(dD_hat.data)**2

# Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
info('<x, J^Ty>: %f, <Jx,y>: %f, difference: %4.4e, ratio: %f'
     % (term1, term2, (term1 - term2)/term1, term1 / term2))
assert np.isclose((term1 - term2)/term1, 0., atol=1.e-5)
