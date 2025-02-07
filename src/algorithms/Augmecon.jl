"""
    AugmentedEpsilonConstraint()

Implements the augmented epsilon-constraint method (AUGMECON) for bi-objective optimization. Extends the standard epsilon-constraint method by adding a secondary term to the objective function to eliminate weakly efficient solutions.

## Supported optimizer attributes

- `MOA.GridPoints()`: Number of grid points to use in the grid search.
- `MOA.AugmeconDelta()`: Weight (δ) for the secondary objective term (default: 1e-3).
"""
mutable struct AugmentedEpsilonConstraint <: AbstractAlgorithm
    grid_points::Union{Nothing,Int}
    delta::Union{Nothing,Float64} # Penalty factor for slack variables
    slack_vars::Vector{MOI.VariableIndex}  # Slack variables for equality constraints

    AugmentedEpsilonConstraint() = new(nothing, nothing, MOI.VariableIndex[])
end

MOI.supports(::AugmentedEpsilonConstraint, ::GridPoints) = true

function MOI.set(alg::AugmentedEpsilonConstraint, ::GridPoints, value)
    alg.grid_points = value
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, attr::GridPoints)
    return something(alg.grid_points, default(alg, attr))
end

MOI.supports(::AugmentedEpsilonConstraint, ::AugmeconDelta) = true

function MOI.set(alg::AugmentedEpsilonConstraint, ::AugmeconDelta, value)
    alg.delta = value
    return
end

function MOI.get(alg::AugmentedEpsilonConstraint, attr::AugmeconDelta)
    return something(alg.delta, default(alg, attr))
end

function optimize_multiobjective!(
    algorithm::AugmentedEpsilonConstraint,
    model::Optimizer,
)
    start_time = time()
    if MOI.output_dimension(model.f) != 2
        error("AugmentedEpsilonConstraint requires exactly two objectives")
    end

    original_sense = MOI.get(model.inner, MOI.ObjectiveSense())
    convert_min_to_max!(model)

    # Determine payoff table using lexicographic method
    payoff_table = Matrix{Float64}(undef, 2, 2)
    alg = Lexicographic()
    MOI.set(alg, LexicographicAllPermutations(), true)
    status, solutions = optimize_multiobjective!(alg, model)
    !_is_scalar_status_optimal(status) && return status, nothing

    for (i, solution) in enumerate(solutions)
        payoff_table[1, i] = solution.y[1]
        payoff_table[2, i] = solution.y[2]
    end

    n_points = MOI.get(algorithm, GridPoints())
    delta = MOI.get(algorithm, AugmeconDelta())
    sense = MOI.get(model.inner, MOI.ObjectiveSense())

    f1, f2 = MOI.Utilities.eachscalar(model.f)
    epsilon_min = min(payoff_table[2, 1], payoff_table[2, 2])
    epsilon_max = max(payoff_table[2, 1], payoff_table[2, 2])

    r = abs(epsilon_max - epsilon_min)
    epsilon_step = r / (n_points - 1)
    current_epsilon = epsilon_min 

    slack = MOI.add_variable(model.inner)
    MOI.add_constraint(model.inner, slack, MOI.GreaterThan(0.0))

    # Create augmented objective: f1 + (delta / r) * slack
    scaled_slack = MOI.Utilities.operate!(*, Float64, delta / r, slack)
    augmented_obj = sense == MOI.MIN_SENSE ? MOI.Utilities.operate!(-, Float64, f1, scaled_slack) : MOI.Utilities.operate!(+, Float64, f1, scaled_slack)
    MOI.set(model.inner, MOI.ObjectiveFunction{typeof(augmented_obj)}(), augmented_obj)

    # Add epsilon constraint: f2 - slack == epsilon 
    constraint_func = sense == MOI.MIN_SENSE ? MOI.Utilities.operate(+, Float64, f2, slack) : MOI.Utilities.operate(-, Float64, f2, slack)
    ci = MOI.add_constraint(model.inner, constraint_func, MOI.EqualTo(current_epsilon))

    solutions = SolutionPoint[]
    variables = MOI.get(model.inner, MOI.ListOfVariableIndices())

    for i in 1:n_points
        _time_limit_exceeded(model, start_time) && (status = MOI.TIME_LIMIT; break)
        
        MOI.set(model.inner, MOI.ConstraintSet(), ci, MOI.EqualTo(current_epsilon))
        MOI.optimize!(model.inner)
        
        if !_is_scalar_status_optimal(model)
            break
        end

        X, Y = _compute_point(model, variables, model.f)
        if original_sense == MOI.MIN_SENSE
            Y .= -Y
        end

        # Check if this solution is non-dominated
        if isempty(solutions) || !(Y ≈ solutions[end].y)
            push!(solutions, SolutionPoint(X, Y))
        end

        current_epsilon = current_epsilon + epsilon_step + MOI.get(model.inner, MOI.VariablePrimal(), slack)
    end

    MOI.delete(model.inner, ci)
    #MOI.delete(model, slack)
    return status, solutions
end

function convert_min_to_max!(model::MOI.AbstractOptimizer)
    sense = MOI.get(model, MOI.ObjectiveSense())
    
    if sense == MOI.MIN_SENSE
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        
        obj_func = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        neg_obj_func = MOI.Utilities.operate!(*, Float64, -1.0, obj_func)
        MOI.set(model, MOI.ObjectiveFunction{typeof(neg_obj_func)}(), neg_obj_func)
    end
end
