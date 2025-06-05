using LinearAlgebra

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
        δθ[j, :] = ((bounds[1] .+ bounds[2] .* U) * V) + coeffs
    end
    return δθ
end
