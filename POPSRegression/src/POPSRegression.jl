using LinearAlgebra
using Random
using Statistics

function sigma(X; lambda_ = 1.0, alpha_ = 1.0)
    decomp = svd(X; full = false)
    Vh = decomp.Vt
    eigen_vals_ = decomp.S .^ 2
    divisor = vec(eigen_vals_ .+ lambda_ / alpha_)
    divisor = reshape(divisor, 1, length(divisor))
    scaled_sigma_ = Vh * transpose(Vh) ./ divisor
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
    eigvals, eigvecs = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))

    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]
    projections = deepcopy(eigvecs)
    projected = pointwise_corrections * projections

    lower = mapslices(col -> quantile(col, percentile_clipping / 100), projected; dims=1)
    upper = mapslices(col -> quantile(col, 1.0 - percentile_clipping / 100), projected; dims=1)
    
    bounds = vcat(lower, upper)

    return eigvecs, bounds
end

function sample_hypercube(projections, bounds, coeffs, number_of_committee_members = 10)
    lower, upper = bounds[1, :], bounds[2, :]

    δθ = zeros((number_of_committee_members, size(projections, 1)))
    U = rand(Float64, (number_of_committee_members, size(lower, 1)))
    committee = deepcopy(δθ)'
    for j = 1:number_of_committee_members
        θ = projections * (((upper .- lower ) .* U[j, :] ) .+ lower)
        δθ[j, :] = θ
        committee[:, j] = θ .+ coeffs
    end

    return committee, (δθ' * δθ) ./ size(δθ, 1)  
end
