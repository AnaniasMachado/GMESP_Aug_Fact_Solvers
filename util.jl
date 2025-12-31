using LinearAlgebra

"""
    aug_fact_objective(x, At, t, t_a)

Compute the Aug-Fact objective value

    hatΦ_t(M_{t_a}(x); t_a) = sum_{i=1}^t log(λ_i(M_{t_a}(x)) + t_a)

Inputs:
- x   :: Vector{Float64}   (solution in [0,1]^n)
- At  :: Matrix{Float64}   (A(t_a), upper-triangular Cholesky factor)
- t   :: Int               (number of eigenvalues)
- t_a :: Float64           (augmentation parameter)

Output:
- objective value (Float64)
"""
function aug_fact_objective(
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64
)
    # Build M_{t_a}(x) = A(t_a) * Diag(x) * A(t_a)'
    M = At * Diagonal(x) * At'

    # Eigenvalues (sorted ascending by default)
    λ = eigvals(Symmetric(M))

    # Take the largest t eigenvalues
    λ_top = @view λ[end-t+1:end]

    return sum(log.(λ_top .+ t_a))
end
