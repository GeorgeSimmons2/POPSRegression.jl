using POPSRegression
using Test
using Random
using LinearAlgebra

@testset "POPSRegression.jl" begin
    # Write your tests here.
    X = rand(10,10)
    Y = rand(10)
    Γ = I + rand(10,10) .* 0.0 
    C = X \ Y
    point_corrections = corrections(X, Y, Γ)
    eigvecs, bounds   = hypercube(point_corrections)
    committee, δθ     = sample_hypercube(eigvecs, bounds, C)
    @test typeof(point_corrections) <: AbstractMatrix
    @test typeof(eigvecs)           <: AbstractMatrix
    @test typeof(bounds)            <: AbstractMatrix
    @test typeof(committee)         <: AbstractMatrix
    @test typeof(δθ)                <: AbstractMatrix
end
