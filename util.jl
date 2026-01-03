using LinearAlgebra, Arpack

# ============================================================
# A(t_a) factorization
# ============================================================
function compute_At(C::Matrix{Float64}, t_a::Float64)
    F = cholesky(Symmetric(C - t_a * I))
    return F.U
end

# ============================================================
# M_{t_a}(x) = A(t_a) * Diagonal(x) * A(t_a)'
# ============================================================
function M_ta(x::Vector{Float64}, At::AbstractMatrix{Float64})
    return At * Diagonal(x) * At'
end

# ============================================================
# GAug-Fact Objective Function
# ============================================================
function gaug_fact_objective(
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64
)
    # Build M_{t_a}(x) = A(t_a) * Diag(x) * A(t_a)'
    M = At * Diagonal(x) * At'

    # Compute the t largest eigenvalues using Arpack
    λ, _ = eigs(Symmetric(M), nev=t, which=:LM)

    return sum(log.(λ .+ t_a))
end

# ============================================================
# Spectral Bound
# ============================================================
function spectral_bound(
    C::AbstractMatrix{Float64},
    t::Int64
)
    # Compute the t largest eigenvalues using Arpack
    λ, _ = eigs(Symmetric(C), nev=t, which=:LM)

    # Sum of log of eigenvalues
    return sum(log.(λ))
end
