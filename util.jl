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
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64
)
    # Build M_{t_a}(x) = A(t_a) * Diag(x) * A(t_a)'
    M = At * Diagonal(x) * At'

    # Compute the t largest eigenvalues using Arpack
    λ, _ = eigs(Symmetric(M), nev=t, which=:LM)

    # Sum of log of eigenvalues
    return sum(log.(λ))
end

# ============================================================
# Spectral Subgradient of hatPhi fractional
# ============================================================
function spectral_subgradient_hatPhi_fractional(
    M::Matrix{Float64},
    t::Int;
    tol = 1e-10
)
    eig = eigen(Symmetric(M))
    λ = eig.values
    U = eig.vectors
    n = length(λ)

    # Sort eigenvalues descending
    perm = sortperm(λ, rev=true)
    λs = λ[perm]
    Us = U[:, perm]

    λt = λs[t]

    # Identify k and ℓ
    k = findlast(i -> λs[i] > λt + tol, 1:n)
    k = isnothing(k) ? 0 : k

    ℓ = findfirst(i -> λs[i] < λt - tol, 1:n)
    ℓ = isnothing(ℓ) ? n + 1 : ℓ

    # Build spectral subgradient
    Y = zeros(size(M))

    # Fully active eigenvectors
    for i in 1:k
        ui = view(Us, :, i)
        Y .+= ui * ui'
    end

    # Fractional part on tied eigenspace
    if k + 1 ≤ ℓ - 1
        r = t - k
        m = ℓ - k - 1
        θ = r / m
        for i in k+1:ℓ-1
            ui = view(Us, :, i)
            Y .+= θ * (ui * ui')
        end
    end

    return Y
end

# ============================================================
# GAug-Fact Closed Form Solution
# ============================================================
function gaug_fact_closed_form(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int
)
    n = size(C, 1)

    # Cholesky factor
    At = cholesky(Symmetric(C - t_a * I)).U

    # Start from barycenter (only to build M once)
    x0 = fill(s / n, n)

    # Build M(x0)
    M = At * Diagonal(x0) * At'

    # Fractional spectral subgradient
    Y = spectral_subgradient_hatPhi_fractional(M, t)

    # Scores
    w = diag(At' * Y * At)

    # Optimal x via hypersimplex maximization
    idx = sortperm(w, rev=true)
    x = zeros(n)
    x[idx[1:s]] .= 1.0

    return x, w
end
