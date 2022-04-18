
export time_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to devito
function time_modeling(model_full::Model, srcGeometry, srcData, recGeometry, recData, pert, op::Char, mode::Int64, options)
    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(srcGeometry)

    # Reutrn directly for J*0
    if op=='J' && mode == 1
        if norm(pert) == 0 && options.return_array == false
            return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
        elseif norm(pert) == 0 && options.return_array == true
            return vec(zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
        end
    end

    # limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model, pert = limit_model_to_receiver_area(srcGeometry, recGeometry, model, options.buffer_size; pert=pert)
    else
        model = model_full
    end

    # Set up Python model structure
    modelPy = devito_model(model, options; pert=pert)

    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    recGeometry, recData = remove_out_of_bounds_receivers(recGeometry, recData, model)

    # Devito interface
    argout = devito_interface(modelPy, srcGeometry, srcData, recGeometry, recData, pert, options)
    # Extend gradient back to original model size
    if isa(argout, Array)
        if op=='J' && mode==-1 && options.limit_m==true
            argout_ = []
            for i=1:length(argout)
                push!(argout_, extend_gradient(model_full, model, argout[i]))
            end
            argout = argout_
        end
    else
        if op=='J' && mode==-1 && options.limit_m==true
            argout = extend_gradient(model_full, model, argout)
        end
    end

    return argout
end

# Function instance without options
time_modeling(model::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())
