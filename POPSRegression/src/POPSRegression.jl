using LinearAlgebra
using Random
using Statistics
using Plots
# Performs O(N) update for constructing the corrections to each coefficient for each
# data point
function POPS(A, Γ, coeffs, Y)
    H = (transpose(Γ) * Γ + transpose(A) * A)
    dθ = zeros(size(A))
    for i = 1:size(A, 1)
        V        = H \ A[i, :]
        leverage = transpose(A[i, :]) * V
        E        = transpose(A[i, :]) * coeffs
        dy       = Y[i] - E
        dθ[i, :] = (dy / leverage) .* V
    end
    return dθ
end

# Constructs the hypercube of parameter updates to sample
function hypercube(dθ, percentile_clipping)
    U, S, V    = svd(dθ)

    projected = dθ * transpose(V)

    # This is the naive implementation where we just use a percentile clipping to get
    # rid of large parameter bounds from "outliers"
    lower  = [quantile(projected[:, i], percentile_clipping / 100.0) for i in 1:size(projected, 2)]
    upper  = [quantile(projected[:, i], 1.0 - percentile_clipping / 100.0) for i in 1:size(projected, 2)]
    bounds = hcat(lower, upper)

    return bounds, V
end

# Sampling the corrections of the parameters and then returning the committee of 
# parameters
function sample_hypercube(bounds, number_of_committee_members, V, coeffs)
    # Setting up for sampling for committee
    δθ = zeros((number_of_committee_members, size(bounds, 1)))
    
    for j = 1:number_of_committee_members
        U  = rand(Float64, size(bounds, 1))
        δθ[j, :] = (V * (bounds[:, 1] .+ bounds[:, 2] .* U) ) + coeffs
    end
    println(δθ)
    return δθ
end

function curve_fit()
    r_max = 10
    percentile_clipping=25.0
    number_of_committee_members = 1000
    number_of_features = 5
    num = 10
    x = collect(range(- r_max, r_max, num))
    true_func = (x .^ 3 + 0.01 .* x .^ 4) .* 0.1 .+ sin.(x) .* x .* 10.0
    design_matrix = zeros((length(x), number_of_features))
    for i = 1:number_of_features
        design_matrix[:, i] = x .^ (i-1)
    end
    m, n = size(design_matrix)
    Γ = Matrix{Float64}(I, m, n)
    coeffs = design_matrix \ true_func
    dθ = POPS(design_matrix, Γ, coeffs, true_func)
    bounds, V = hypercube(dθ,percentile_clipping)
    println(size(V), size(bounds))
    println(size(design_matrix))
    δθ = sample_hypercube(bounds, number_of_committee_members, V, coeffs)
    println(bounds)
    plot(x, true_func, label = "True", ylim = (-200, 200))
    y_predict = design_matrix * coeffs
    num = 40
    design_matrix = zeros((num, number_of_features))
    x_ = collect(range(-10.1,10.1,num))
    for i = 1:number_of_features
        design_matrix[:, i] = x_ .^ (i-1)
    end
    scatter!(x_, design_matrix * coeffs, label = "Global")
    num = 40
    sin_pred = zeros(number_of_committee_members, num)
    x        = collect(range(-r_max, r_max, num))
    for j = 1:number_of_committee_members
	    x_ = collect(range(-r_max, r_max, num))
        temp_coeffs = δθ[j, :]
        design_matrix = zeros((length(x_), number_of_features))
        for i = 1:number_of_features
            design_matrix[:, i] = x_ .^ (i-1)
        end
        sin_pred[j, :] =design_matrix * temp_coeffs
    end
    for i in 1:number_of_committee_members
        scatter!(x, sin_pred[i,:], primary=false, markercolor=i+1)

    end
    savefig("sin_fit.png")
end
curve_fit()
