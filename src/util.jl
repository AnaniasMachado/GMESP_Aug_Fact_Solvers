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
# Phi Function
# ============================================================
function phi(
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
# GAug-Fact Objective Function
# ============================================================
function gaug_fact_objective(
    x::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64
)
    # Build M_{t_a}(x)
    M = At * Diagonal(x) * At'

    # Full eigen-decomposition (needed for psi_t)
    λ = eigvals(Symmetric(M))

    # Sort eigenvalues in descending order
    λs = sort(λ, rev=true)

    # Form y = λ + t_a * I_t
    y = copy(λs)
    y[1:t] .+= t_a

    # Determine k according to Definition 3
    k = 0
    for i in 0:(t-1)
        avg = sum(y[i+1:end]) / (t - i)
        if i == 0
            if avg >= y[1]
                k = 0
                break
            end
        else
            if y[i] > avg && avg >= y[i+1]
                k = i
                break
            end
        end
    end

    # Compute psi_t(y)
    if k > 0
        val = sum(log.(y[1:k]))
    else
        val = 0.0
    end

    val += (t - k) * log(sum(y[k+1:end]) / (t - k))

    return val
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

# ============================================================
# Simplex Solution (closed-form for t=1)
# ============================================================
function simplex_sol(At::AbstractMatrix{Float64}, s::Int)
    n = size(At, 2)
    
    # Compute squared norm of each column
    col_norms = [norm(At[:, i])^2 for i in 1:n]
    
    # Get indices of the s largest norms
    sorted_indices = sortperm(col_norms, rev=true)
    S_star = sorted_indices[1:s]
    
    # Build solution vector
    x = zeros(n)
    x[S_star] .= 1.0
    
    return x
end
