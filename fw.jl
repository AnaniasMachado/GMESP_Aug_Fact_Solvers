using LinearAlgebra

function spectral_subgradient_hatPhi_paper(
    M::Matrix{Float64},
    t::Int;
    tol = 1e-10
)
    eig = eigen(Symmetric(M))
    U = eig.vectors
    λ = eig.values
    n = length(λ)

    # Sort descending
    perm = sortperm(λ, rev=true)
    λs = λ[perm]
    Us = U[:, perm]

    λt = λs[t]

    # Identify k and ℓ
    k = findlast(i -> λs[i] > λt + tol, 1:n)
    k = isnothing(k) ? 0 : k

    ℓ = findfirst(i -> λs[i] < λt - tol, 1:n)
    ℓ = isnothing(ℓ) ? n+1 : ℓ

    # Construct Y
    Y = zeros(size(M))

    # Full ones
    for i in 1:k
        ui = view(Us, :, i)
        Y .+= ui * ui'
    end

    # Fractional part on tied eigenspace
    if k+1 ≤ ℓ-1
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

function fw_gaug_fact_paper(
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
        Y = spectral_subgradient_hatPhi_paper(M, t)

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
    tol::Float64 = 1e-8,
    maxiter::Int = 50
)
    γlo, γhi = 0.0, γmax

    for _ in 1:maxiter
        γ = 0.5 * (γlo + γhi)

        xγ = x .+ γ .* d
        Mγ = At * Diagonal(xγ) * At'
        Yγ = spectral_subgradient_hatPhi_paper(Mγ, t)

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
        Y = spectral_subgradient_hatPhi_paper(M, t)
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
