using LinearAlgebra

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

    # Eigenvalues (sorted ascending by default)
    λ = eigvals(Symmetric(M))

    # Take the largest t eigenvalues
    λ_top = @view λ[end-t+1:end]

    return sum(log.(λ_top .+ t_a))
end
