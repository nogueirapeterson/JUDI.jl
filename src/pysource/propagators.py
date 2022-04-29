from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import (wf_as_src, wavefield, otf_dft, extended_src_weights,
                        extented_src, wavefield_subsampled, weighted_norm,
                        particle_velocity)
from sensitivity import grad_expr, lin_src
from utils import weight_fun, opt_op

from devito import Operator, Function
from devito.tools import as_tuple


def name(model):
    if model.is_tti:
        return "tti"
    elif model.is_viscoacoustic:
        return "viscoacoustic"
    else:
        return ""


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, space_order=8, time_order=2,
            save=False, q=None, return_op=False, freq_list=None, dft_sub=None,
            ws=None, t_sub=1, f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, time_order=time_order, save=save, nt=nt,
                  t_sub=t_sub)
    if time_order == 1:
        v = particle_velocity(model, space_order, time_order=time_order, save=save,
                              nt=nt, t_sub=t_sub)
        kwargs.update({'v': v})

    # Expression for saving wavefield if time subsampling is used
    u_save, eq_save = wavefield_subsampled(model, u, nt, t_sub, time_order=time_order)

    # Add extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, time_order=time_order, q=q, f0=f0, **kwargs)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords, nt=nt,
                                rec_coords=rcv_coords, wavelet=wavelet,
                                time_order=time_order)

    # On-the-fly Fourier
    dft, dft_modes = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + dft + geom_expr + eq_save,
                  subs=subs, name="forward"+name(model),
                  opt=opt_op(model))
    op.cfunction
    if return_op:
        return op, u, rcv

    summary = op()

    # Output
    return rcv, dft_modes or (u_save if t_sub > 1 else u), summary


def adjoint(model, y, src_coords, rcv_coords, space_order=8, time_order=2, q=0,
            dft_sub=None, save=False, ws=None, norm_v=False, w_fun=None, freq_list=None,
            f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if y is None else y.shape[0]

    # Setting adjoint wavefield
    v = wavefield(model, space_order, time_order=time_order, save=save, nt=nt, fw=False)
    if time_order == 1:
        va = particle_velocity(model, space_order, time_order=time_order, save=save,
                               nt=nt, fw=False)
        kwargs.update({'v': va})
        kwargs['time_m'] = 0

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, time_order=time_order, q=q, fw=False, f0=f0, **kwargs)

    # On-the-fly Fourier
    dft, dft_modes = otf_dft(v, freq_list, model.critical_dt, factor=dft_sub)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, v, src_coords=rcv_coords, nt=nt,
                                rec_coords=src_coords, wavelet=y, fw=False,
                                time_order=time_order)

    # Extended source
    wsrc, ws_expr = extended_src_weights(model, ws, v)

    # Wavefield norm
    nv_t, nv_s = ([], [])
    if norm_v:
        weights = weight_fun(w_fun, model, src_coords)
        norm_v, (nv_t, nv_s) = weighted_norm(v, weight=weights)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + ws_expr + nv_t + dft + geom_expr + nv_s,
                  subs=subs, name="adjoint"+name(model),
                  opt=opt_op(model))
    op.cfunction
    # Run operator
    summary = op()

    # Output
    if wsrc:
        return wsrc, summary
    if norm_v:
        return rcv, dft_modes or v, norm_v.data[0], summary
    return rcv, v, summary


def gradient(model, residual, rcv_coords, u, return_op=False, space_order=8,
             time_order=2, w=None, freq=None, dft_sub=None, isic=False, f0=0.015,
             **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute the action of the adjoint Jacobian onto a residual J'* δ d.
    """
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, time_order=time_order, fw=False)
    if time_order == 1:
        va = particle_velocity(model, space_order, time_order=time_order, fw=False)
        kwargs.update({'v': va})
        kwargs['time_m'] = 0

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, time_order=time_order, fw=False, f0=f0, **kwargs)

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, v, src_coords=rcv_coords,
                              wavelet=residual, fw=False, time_order=time_order)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, u, v, model, time_order=time_order, w=w, freq=freq,
                       dft_sub=dft_sub, isic=isic)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr,
                  subs=subs, name="gradient"+name(model),
                  opt=opt_op(model))
    try:
        op.cfunction
    except:
        op = Operator(pde + geom_expr + g_expr,
                      subs=subs, name="gradient"+name(model),
                      opt='advanced')
        op.cfunction
    if return_op:
        return op, gradm, v

    summary = op()

    # Output
    return gradm, summary


def born(model, src_coords, rcv_coords, wavelet, space_order=8, time_order=2,
         save=False, q=None, return_op=False, isic=False, freq_list=None,
         dft_sub=None, ws=None, t_sub=1, nlind=False, f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute linearized wavefield U = J(m)* δ m
    and related quantities.
    """
    nt = wavelet.shape[0]
    # Setting wavefield
    u = wavefield(model, space_order, time_order=time_order, save=save, nt=nt,
                  t_sub=t_sub)
    ul = wavefield(model, space_order, time_order=time_order, name="l")
    if time_order == 1:
        v = particle_velocity(model, space_order, time_order=time_order, save=save,
                              nt=nt, t_sub=t_sub)
        kwargs.update({'v': v})

    # Expression for saving wavefield if time subsampling is used
    u_save, eq_save = wavefield_subsampled(model, u, nt, t_sub, time_order=time_order)

    # Extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, time_order=time_order, q=q, f0=f0, **kwargs)
    if model.dm == 0:
        pdel = []
    else:
        if time_order == 1:
            dv = particle_velocity(model, space_order, time_order=time_order, save=save,
                                   nt=nt, t_sub=t_sub, name="d")
            kwargs.update({'v': dv})
        q = lin_src(model, u, time_order, isic=isic)
        pdel = wave_kernel(model, ul, time_order=time_order, q=q, f0=f0, **kwargs)

    # Setup source and receiver
    geom_expr, _, rcvnl = src_rec(model, u, rec_coords=rcv_coords if nlind else None,
                                  src_coords=src_coords, wavelet=wavelet,
                                  time_order=time_order)
    geom_exprl, _, rcvl = src_rec(model, ul, rec_coords=rcv_coords, nt=nt,
                                  time_order=time_order)

    # On-the-fly Fourier
    dft, dft_modes = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + geom_exprl + pdel + dft + eq_save,
                  subs=subs, name="born"+name(model),
                  opt=opt_op(model))
    op.cfunction
    outrec = (rcvl, rcvnl) if nlind else rcvl
    if return_op:
        return op, u, outrec

    summary = op()

    # Output
    return outrec, dft_modes or (u_save if t_sub > 1 else u), summary


# Forward propagation
def forward_grad(model, src_coords, rcv_coords, wavelet, v, space_order=8,
                 time_order=2, q=None, ws=None, isic=False, w=None, freq=None,
                 f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, time_order=time_order, save=False)

    # Add extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, time_order=time_order, q=q, f0=f0, **kwargs)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords, nt=nt,
                                rec_coords=rcv_coords, wavelet=wavelet)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, v, u, model, w=w, isic=isic, freq=freq)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr,
                  subs=subs, name="forward_grad"+name(model),
                  opt=opt_op(model))

    summary = op()

    # Output
    return rcv, gradm, summary
