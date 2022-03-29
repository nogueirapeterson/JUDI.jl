# Adjoint test for F and J
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

using Distributed

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
viscoacoustic = parsed_args["viscoacoustic"]
fs =  parsed_args["fs"]

# # Set parallel if specified
nw = parsed_args["parallel"]
if nw > 1 && nworkers() < nw
    addprocs(nw-nworkers() + 1; exeflags=["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
end

@everywhere using JUDI, LinearAlgebra, Test, Distributed

### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["viscoacoustic"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info, f0 = setup_geom(model; nsrc=nw)
dt = srcGeometry.dt[1]

v = sqrt.(1f0 ./ model.m)
v0 = sqrt.(1f0 ./ model0.m)
qp = 3.516f0 .* ((v .* 1000f0).^2.2f0) .* 10f0^(-6f0)
qp0 = 3.516f0 .* ((v0 .* 1000f0).^2.2f0) .* 10f0^(-6f0)
kappa = model.m .* (1f0 ./ model.rho)
kappa0 = model0.m .* (1f0 ./ model0.rho)
dtau = (1f0 ./ qp) .- (1f0 ./ qp0)
dkappa = kappa .- kappa0
dx = (dkappa, dtau)

tol = 5f-4
(tti && fs) && (tol = 5f-3)
###################################################################################################
# Modeling operators
@testset "Adjoint test with $(nlayer) layers and viscoacoustic $(viscoacoustic) and freesurface $(fs) and pertubation=dkappa" begin

    # Define multi-parameters
    multi_parameters = (1, 0)

    opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], f0=f0, multi_parameters=multi_parameters)
    F = judiModeling(model0, srcGeometry, recGeometry; options=opt)

    # Linearized modeling
    J = judiJacobian(F, q)

    ld_hat = J*dkappa
    dkappa_hat = J'*ld_hat

    term1 = dot(dkappa_hat, dkappa)
    term2 = norm(ld_hat.data[1]).^2

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    @printf("<x, J^Ty>: %2.5e, <Jx,y>: %2.5e, difference: %2.5e, ratio: %2.5e \n", term1, term2, (term1 - term2)/term1, term1 / term2)
    @test isapprox((term1 - term2)/term1, 0., atol=tol, rtol=0)

end
###################################################################################################
@testset "Adjoint test with $(nlayer) layers and viscoacoustic $(viscoacoustic) and freesurface $(fs) and pertubation=dtau" begin

    # Define multi-parameters
    multi_parameters = (0, 1)

    opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], f0=f0, multi_parameters=multi_parameters)
    F = judiModeling(model0, srcGeometry, recGeometry; options=opt)

    # Linearized modeling
    J = judiJacobian(F, q)

    ld_hat = J*dtau
    dtau_hat = J'*ld_hat

    term1 = dot(dtau_hat, dtau)
    term2 = norm(ld_hat.data[1]).^2

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    @printf("<x, J^Ty>: %2.5e, <Jx,y>: %2.5e, difference: %2.5e, ratio: %2.5e \n", term1, term2, (term1 - term2)/term1, term1 / term2)
    @test isapprox((term1 - term2)/term1, 0., atol=tol, rtol=0)

end
###################################################################################################
@testset "Adjoint test with $(nlayer) layers and viscoacoustic $(viscoacoustic) and freesurface $(fs) and pertubation=(dkappa,dtau)" begin

    # Define multi-parameters
    multi_parameters = (1, 1)

    opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], f0=f0, multi_parameters=multi_parameters)
    F = judiModeling(model0, srcGeometry, recGeometry; options=opt)

    # Linearized modeling
    J = judiJacobian(F, q)

    ld_hat = J*dx
    dx_hat = J'*ld_hat

    term1 = dot(dx_hat, dx)
    term2 = norm(ld_hat.data[1]).^2

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    @printf("<x, J^Ty>: %2.5e, <Jx,y>: %2.5e, difference: %2.5e, ratio: %2.5e \n", term1, term2, (term1 - term2)/term1, term1 / term2)
    @test isapprox((term1 - term2)/term1, 0., atol=tol, rtol=0)

end