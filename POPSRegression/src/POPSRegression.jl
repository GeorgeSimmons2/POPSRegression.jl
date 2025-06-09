using LinearAlgebra
using Random
using Statistics
using Plots

function sigma(X; lambda_ = 1.0, alpha_ = 1.0)
    decomp = svd(X; full = false)
    Vh = decomp.Vt                        # (n_components × n_features)
    eigen_vals_ = decomp.S .^ 2
    divisor = eigen_vals_ .+ lambda_ / alpha_

    # divide each row of Vh by corresponding element of divisor
    Vh_scaled = Vh ./ divisor[:, ones(Int, size(Vh, 2))]  # row-wise division

    scaled_sigma_ = Vh' * Vh_scaled                        # (n_features × n_features)
    return scaled_sigma_
end

function corrections(X, Y, scaled_sigma_, coeffs; leverage_percentile=0.0)
    errors = Y .- (X * coeffs) 
    leverage = vec(sum((X * scaled_sigma_) .* X, dims=2))
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = (X * scaled_sigma_)[mask, :]
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])

    return pointwise_corrections
end

function hypercube(pointwise_corrections; percentile_clipping = 0.0)
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals = eig.values
    eigvecs = eig.vectors

    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]

    projections = eigvecs
    projected = pointwise_corrections * projections

    lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
    upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

    bounds = vcat(lower', upper')  # (2 x N)

    return eigvecs, bounds
end

function sample_hypercube(projections, bounds, coeffs; number_of_committee_members = 10)
    lower, upper = bounds[1, :], bounds[2, :]

    U = rand(Float64, (number_of_committee_members, size(lower, 1)))

    committee = projections * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
    δθ        = committee * committee' ./ size(committee, 2)

    committee = coeffs[:,:] .+ committee

    return committee, δθ  
end
