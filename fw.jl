using LinearAlgebra

function spectral_subgradient_hatPhi(M::Matrix{Float64}, t::Int)
    # --- Eigen-decomposition ---
    eig = eigen(Symmetric(M))
    λ = eig.values
    U = eig.vectors
    n = length(λ)

    # --- Sort eigenvalues descending ---
    perm = sortperm(λ, rev=true)
    λs = λ[perm]
    Us = U[:, perm]

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

function fw_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # Initial feasible point
    x = fill(s / n, n)
    gap = 1e6

    k = 0
    while true
        k += 1

        # --- Build M(x) ---
        M = At * Diagonal(x) * At'

        # --- Paper subgradient ---
        Y = spectral_subgradient_hatPhi(M, t)

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
        γ = 2.0 / (k + 2)

        # --- Update ---
        x .= (1 - γ) .* x .+ γ .* v
    end
end

function fw_exact_line_search(
    x::Vector{Float64},
    d::Vector{Float64},
    At::AbstractMatrix{Float64},
    t::Int;
    γmax::Float64,
    tol::Float64 = 1e-5,
    maxiter::Int = 20
)
    γlo, γhi = 0.0, γmax

    for _ in 1:maxiter
        γ = 0.5 * (γlo + γhi)

        xγ = x .+ γ .* d
        Mγ = At * Diagonal(xγ) * At'
        Yγ = spectral_subgradient_hatPhi(Mγ, t)

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

function fw_gaug_fact_exact_ls(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    x = fill(s / n, n)
    gap = 1e6

    k = 0
    while true
        k += 1

        M = At * Diagonal(x) * At'
        Y = spectral_subgradient_hatPhi(M, t)
        grad = diag(At' * Y * At)

        # FW vertex
        idx = sortperm(grad, rev=true)
        v = zeros(n)
        v[idx[1:s]] .= 1.0

        d = v .- x
        gap = dot(grad, d)

        if gap ≤ tol
            return x, gap, k
        end

        γ = fw_exact_line_search(x, d, At, t; γmax = 1.0)
        x .= x .+ γ .* d
    end
end
