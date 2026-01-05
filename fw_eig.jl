using LinearAlgebra

function spectral_subgradient_hatPhi_eig(x::Vector{Float64}, λ::Vector{Float64}, U::AbstractMatrix{Float64}, t::Int, t_a::Float64)
    # --- Sort eigenvalues descending ---
    n = length(x)
    λx = zeros(n)
    λx .= x .* λ
    perm = sortperm(λx, rev=true)
    λs = λx[perm]
    Us = U[:, perm]

    # --- Add t_a
    λs[1:t] .+= t_a

    # --- Determine k ---
    k = 0
    for i in 0:(t-1)
        avg = sum(λs[i+1:end]) / (t - i)
        if i == 0
            if Inf > avg && avg >= λs[i+1]
                k = i
                break
            end
        else
            if λs[i] > avg && avg >= λs[i+1]
                k = i
                break
            end
        end
    end

    # --- Construct subgradient matrix ---
    Y = zeros(n, n)   

    # Top k eigenvectors
    if k > 0
        Y .+= Us[:, 1:k] * Diagonal(1 ./ λs[1:k]) * Us[:, 1:k]'
    end

    # Fractional weight for remaining eigenvectors
    if k < n
        r = t - k
        sum_tail = sum(λs[k+1:end])
        weight = r / sum_tail
        U_tail = Us[:, k+1:end]
        Y .+= U_tail * (weight * I(size(U_tail, 2))) * U_tail'
    end

    return Y
end

function fw_exact_line_search_eig(
    x::Vector{Float64},
    d::Vector{Float64},
    λ::Vector{Float64},
    U::AbstractMatrix{Float64},
    At::AbstractMatrix{Float64},
    t::Int,
    t_a::Float64;
    γmax::Float64,
    tol::Float64 = 1e-5,
    maxiter::Int = 50,
)
    γlo, γhi = 0.0, γmax

    for _ in 1:maxiter
        γ = 0.5 * (γlo + γhi)

        xγ = x .+ γ .* d
        Yγ = spectral_subgradient_hatPhi_eig(xγ, λ, U, t, t_a)

        gradγ = diag(At' * Yγ * At)
        deriv = dot(gradγ, d)

        if abs(deriv) ≤ tol
            return γ
        elseif deriv > 0
            γlo = γ
        else
            γhi = γ
        end
    end

    return 0.5 * (γlo + γhi)
end

function fw_gaug_fact_eig(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
)
    # --- Eigen-decomposition ---
    eig = eigen(Symmetric(C - t_a * I))
    λ = eig.values
    U = eig.vectors
    At = sqrt.(λ) .* U'
    n = length(λ)

    # Initial feasible point
    x = fill(s / n, n)
    gap = 1e6
    γ = 0.0

    k = 0
    while true
        k += 1

        # --- Paper subgradient ---
        Y = spectral_subgradient_hatPhi_eig(x, λ, U, t, t_a)

        # --- Gradient ---
        grad = diag(At' * Y * At)

        # --- Linear minimization oracle ---
        idx = sortperm(grad, rev=true)
        v = zeros(n)
        v[idx[1:s]] .= 1.0

        # --- FW gap ---
        gap = dot(grad, v .- x)
        if gap ≤ tol
            return x, gap, k
        end

        # --- Step size (standard) ---
        if !line_search
            γ = 2.0 / (k + 2)
        else
            d = v .- x
            γ = fw_exact_line_search_eig(x, d, λ, U, At, t, t_a; γmax = 1.0)
        end

        # --- Update ---
        x .= (1 - γ) .* x .+ γ .* v
    end
end