"""
    AugmentedEpsilonConstraint()

Implements the augmented epsilon-constraint method (AUGMECON) for bi-objective optimization. Extends the standard epsilon-constraint method by adding a secondary term to the objective function to eliminate weakly efficient solutions.

## Supported optimizer attributes

- `MOA.EpsilonConstraintStep()`: Step size for partitioning the first objective's domain.
- `MOA.SolutionLimit()`: Maximum number of solutions to generate.
- `MOA.AugmentationFactor()`: Weight (δ) for the secondary objective term (default: 1e-3).
"""
mutable struct AugmentedEpsilonConstraint <: AbstractAlgorithm
    solution_limit::Union{Nothing,Int}
    atol::Union{Nothing,Float64}
    delta::Float64  # Augmentation factor (δ)

    AugmentedEpsilonConstraint() = new(nothing, nothing, 1e-3)
end

MOI.supports(::AugmentedEpsilonConstraint, ::SolutionLimit) = true

function MOI.set(alg::AugmentedEpsilonConstraint, ::SolutionLimit, value)
    alg.solution_limit = value
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, attr::SolutionLimit)
    return something(alg.solution_limit, default(alg, attr))
end

MOI.supports(::AugmentedEpsilonConstraint, ::EpsilonConstraintStep) = true

function MOI.set(alg::AugmentedEpsilonConstraint, ::EpsilonConstraintStep, value)
    alg.atol = value
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, attr::EpsilonConstraintStep)
    return something(alg.atol, default(alg, attr))
end

# Add support for AugmentationFactor
MOI.supports(::AugmentedEpsilonConstraint, ::MOI.AbstractOptimizerAttribute) = false  # Restrict to supported attributes

function MOI.set(alg::AugmentedEpsilonConstraint, ::AugmeconDelta, value)
    alg.delta = value
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, attr::AugmeconDelta)
    return something(alg.delta, default(alg, attr))
end

MOI.supports(::AugmentedEpsilonConstraint, ::ObjectiveAbsoluteTolerance) = true

function MOI.set(alg::AugmentedEpsilonConstraint, ::ObjectiveAbsoluteTolerance, value)
    @warn("This attribute is deprecated. Use `EpsilonConstraintStep` instead.")
    MOI.set(alg, EpsilonConstraintStep(), value)
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, ::ObjectiveAbsoluteTolerance)
    @warn("This attribute is deprecated. Use `EpsilonConstraintStep` instead.")
    return MOI.get(alg, EpsilonConstraintStep())
end

function optimize_multiobjective!(
    algorithm::AugmentedEpsilonConstraint,
    model::Optimizer,
)
    start_time = time()
    if MOI.output_dimension(model.f) != 2
        error("AugmentedEpsilonConstraint requires exactly two objectives")
    end

    # Compute objective bounds
    alg = Hierarchical()
    MOI.set.(Ref(alg), ObjectivePriority.(1:2), [1, 0])
    status, solution_1 = optimize_multiobjective!(alg, model)
    !_is_scalar_status_optimal(status) && return status, nothing

    MOI.set(alg, ObjectivePriority(2), 2)
    status, solution_2 = optimize_multiobjective!(alg, model)
    !_is_scalar_status_optimal(status) && return status, nothing

    a, b = solution_1[1].y[1], solution_2[1].y[1]
    left, right = min(a, b), max(a, b)

    # Compute ε-step
    ε = MOI.get(algorithm, EpsilonConstraintStep())
    n_points = MOI.get(algorithm, SolutionLimit())
    if n_points != default(algorithm, SolutionLimit())
        ε = abs(right - left) / (n_points - 1)
    end

    f1, f2 = MOI.Utilities.eachscalar(model.f)
    r1 = right - left
    r1 = r1 ≈ 0.0 ? 1.0 : r1  # Handle zero range
    δ = algorithm.delta
    T = Float64

    # Create augmented objective: f2 + δ*(f1/r1)
    scaled_f1 = MOI.Utilities.operate(*, typeof(δ/r1), δ/r1, f1)
    augmented_obj = MOI.Utilities.operate(+, T, f2, scaled_f1)

    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(augmented_obj)}(), augmented_obj)

    # Iterate with epsilon constraints
    solutions = SolutionPoint[]
    sense = MOI.get(model.inner, MOI.ObjectiveSense())
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())
    SetType, bound = sense == MOI.MIN_SENSE ? 
        (MOI.LessThan{Float64}, right) : (MOI.GreaterThan{Float64}, left)
    
    constant = MOI.constant(f1, Float64)
    ci = MOI.Utilities.normalize_and_add_constraint(
        model,
        f1,
        SetType(bound);
        allow_modify_function=true,
    )
    bound -= constant

    status = MOI.OPTIMAL
    for _ in 1:n_points
        _time_limit_exceeded(model, start_time) && (status = MOI.TIME_LIMIT; break)
        
        MOI.set(model, MOI.ConstraintSet(), ci, SetType(bound))
        MOI.optimize!(model.inner)
        !_is_scalar_status_optimal(model) && break

        X, Y = _compute_point(model, variables, model.f)
        if isempty(solutions) || !(Y ≈ solutions[end].y)
            push!(solutions, SolutionPoint(X, Y))
        end

        # Update bound
        adjustment = sense == MOI.MIN_SENSE ? (Y[1] - constant - ε) : (Y[1] - constant + ε)
        bound = sense == MOI.MIN_SENSE ? 
            min(adjustment, bound - ε) : 
            max(adjustment, bound + ε)
    end
    MOI.delete(model, ci)

    #return status, filter_nondominated(sense, solutions)
    return status, solutions
end