from devito import Eq, solve, div, grad
from sympy import sqrt

from wave_utils import freesurface, memory_field
from FD_utils import laplacian, sa_tti


def wave_kernel(model, u, time_order=2, fw=True, q=None, f0=0.015, **kwargs):
    """
    Pde kernel corresponding the the model for the input wavefield

    Parameters
    ----------
    model: Model
        Physical model
    u : TimeFunction or tuple
        wavefield (tuple if TTI or Viscoacoustic)
    time_order: Int (optional)
        Time discretization order, defaults to 2
    fw : Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source
    f0 : Peak frequency
    """
    if model.is_tti:
        pde = tti_kernel(model, u[0], u[1], fw=fw, q=q)
    elif model.is_viscoacoustic:
        eq_kernel = stencils[time_order]
        pde = eq_kernel(model, u, fw=fw, q=q, f0=f0, **kwargs)
    else:
        pde = acoustic_kernel(model, u, fw=fw, q=q)
    return pde


def acoustic_kernel(model, u, fw=True, q=None):
    """
    Acoustic wave equation time stepper

    Parameters
    ----------
    model: Model
        Physical model
    u : TimeFunction or tuple
        wavefield (tuple if TTI)
    fw : Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source
    """
    u_n = u.forward if fw else u.backward
    udt = u.dt if fw else u.dt.T
    q = q or 0

    # Set up PDE expression and rearrange
    ulaplace = laplacian(u, model.irho)
    wmr = model.irho * model.m
    damp = model.damp
    stencil = solve(wmr * u.dt2 + damp * udt - ulaplace - q, u_n)

    if 'nofsdomain' in model.grid.subdomains:
        pde = [Eq(u_n, stencil, subdomain=model.grid.subdomains['nofsdomain'])]
        pde += freesurface(model, pde)
    else:
        pde = [Eq(u_n, stencil)]

    return pde


def SLS_1st_order(model, p, fw=True, q=None, f0=0.015, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Blanch and Symes (1995) / Dutta and Schuster (2014).

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    model: Model
        Physical model
    p : TimeFunction
        Pressure field
    time_order: Int (optional)
        Time discretization order, defaults to 2
    fw: Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source as a tuple (one value for each component)
    f0 : Peak frequency
    """
    qp, b, damp, m = model.qp, model.irho, model.damp, model.m

    # Source
    q = q or 0

    # The stress relaxation parameter
    t_s = (sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / b

    # memory variable
    r = memory_field(p)

    # Particle Velocity
    v = kwargs.pop('v')

    if fw:

        # Particle velocity
        pde_v = v.dt + b * grad(p)
        u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

        # Attenuation Memory variable
        pde_r = r.dt + (1. / t_s) * (r + tt * rho * div(v.forward))
        u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

        # Pressure
        pde_p = m * p.dt + rho * (tt + 1.) * div(v.forward) + r.forward - q
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        return [u_v, u_r, u_p]

    else:

        # Attenuation Memory variable
        pde_r = r.dt.T + (1. / t_s) * r + p
        u_r = Eq(r.backward, damp * solve(pde_r, r.backward))

        # Particle velocity
        # Because v is a Vector, `.T` applies a standard matrix transpose
        # so we need to do the derivative transpose by hand with `-*.dtl`
        pde_v = -v.dtl - grad(rho * (1. + tt) * p) - \
            grad((1. / t_s) * rho * tt * r.backward)
        u_v = Eq(v.backward, damp * solve(pde_v, v.backward))

        # Pressure
        pde_p = m * p.dt.T - div(b * v.backward)
        u_p = Eq(p.backward, damp * solve(pde_p, p.backward))

        return [u_r, u_v, u_p]


def SLS_2nd_order(model, p, fw=True, q=None, f0=0.015, **kwargs):
    """
    Viscoacoustic 2nd SLS wave equation.
    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Bulk modulus moved to rhs. The adjoint equation is directly derived
    as the discrete adjoint of the forward PDE which leads to a slightly different
    formulation than in the paper.

    Parameters
    ----------
    model: Model
        Physical model
    p : TimeFunction
        Pressure field
    time_order: Int (optional)
        Time discretization order, defaults to 2
    fw: Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source as a tuple (one value for each component)
    f0 : Peak frequency
    """
    qp, b, damp, m = model.qp, model.irho, model.damp, model.m
    m = m * b
    # Source
    q = q or 0

    # The stress relaxation parameter
    t_s = (sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s) - 1

    # memory variable
    r = memory_field(p)

    if fw:
        # Attenuation Memory variable
        pde_r = b * r.dt - (tt / t_s) * laplacian(p, b) + (b / t_s) * r
        u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

        # Pressure
        pde_p = m * p.dt2 - (1. + tt) * laplacian(p, b) + \
            b * r.forward - q + (1 - damp) * p.dt
        u_p = Eq(p.forward, solve(pde_p, p.forward))

        return [u_r, u_p]
    else:
        # Attenuation Memory variable
        pde_r = r.dt.T + b * p + (1 / t_s) * r
        u_r = Eq(r.backward, damp * solve(pde_r, r.backward))

        # Pressure
        pde_p = m * p.dt2 - laplacian((1. + tt) * p, b) - \
            laplacian((tt/(b*t_s)) * r.backward, b) + (1 - damp) * p.dt.T - q

        u_p = Eq(p.backward, solve(pde_p, p.backward))

        return [u_r, u_p]


def tti_kernel(model, u1, u2, fw=True, q=None):
    """
    TTI wave equation (one from my paper) time stepper

    Parameters
    ----------
    model: Model
        Physical model
    u1 : TimeFunction
        First component (pseudo-P) of the wavefield
    u2 : TimeFunction
        First component (pseudo-P) of the wavefield
    fw: Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source as a tuple (one value for each component)
    """
    m, damp, irho = model.m, model.damp, model.irho
    wmr = (irho * m)
    q = q or (0, 0)

    # Tilt and azymuth setup
    u1_n, u2_n = (u1.forward, u2.forward) if fw else (u1.backward, u2.backward)
    (udt1, udt2) = (u1.dt, u2.dt) if fw else (u1.dt.T, u2.dt.T)
    H0, H1 = sa_tti(u1, u2, model)

    # Stencils
    stencilp = solve(wmr * u1.dt2 + damp * udt1 - H0 - q[0], u1_n)
    stencilr = solve(wmr * u2.dt2 + damp * udt2 - H1 - q[1], u2_n)

    if 'nofsdomain' in model.grid.subdomains:
        # Water at free surface, no anisotrpy
        acout_ttip = [Eq(u1_n, stencilp.subs(model.zero_thomsen))]
        acout_ttir = [Eq(u2_n, stencilr.subs(model.zero_thomsen))]
        pdea = freesurface(model, acout_ttip) + freesurface(model, acout_ttir)
        # Standard PDE in subsurface
        first_stencil = Eq(u1_n, stencilp, subdomain=model.grid.subdomains['nofsdomain'])
        second_stencil = Eq(u2_n, stencilr, subdomain=model.grid.subdomains['nofsdomain'])
    else:
        pdea = []
        first_stencil = Eq(u1_n, stencilp)
        second_stencil = Eq(u2_n, stencilr)

    return [first_stencil, second_stencil] + pdea


stencils = {1: SLS_1st_order, 2: SLS_2nd_order}
