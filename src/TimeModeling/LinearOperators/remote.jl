# Create a solver on every worker
function init_local(name::Symbol, model::Model, options::Options)
    pymodel = devito_model(model, options)
    opts = Dict(Symbol(s) => getfield(options, s) for s in fieldnames(JUDI.Options))
    Core.eval(JUDI, :($name = we."WaveSolver"($pymodel; $(opts)...)))
    nothing
end

function init_solver(model::Model, options::Options)
    solver = make_id()
    @sync for p in workers()
        @async remotecall_wait(init_local, p, solver, model, options)
    end
    return solver
end

# Update model on every worker
function update_local(name::Symbol, m::AbstractArray{T, N}, dm::AbstractArray{T, N}) where {T, N}
    pysolver = getfield(JUDI, name)
    pysolver."model"."dm" = dm
    pysolver."model"."m" = m
    nothing
end

function update_local(name::Symbol, m::AbstractArray{T, N}, dm::Nothing) where {T, N}
    pysolver = getfield(JUDI, name)
    pysolver."model"."m" = m
    nothing
end

function update_model!(m::Model, o::Options, s::Symbol, dm::Union{Nothing, Array{T, N}}) where {T, N}
    !isnothing(dm) && (dm = reshape(dm, m.n))
    @sync for p in workers()
        @async remotecall_wait(update_local, p, s, m.m.data, dm)
    end
    nothing
end

update_model!(m::Model, o::Options, s::Symbol, dm::PhysicalParameter) = update_model!(m, o, s, dm.data)