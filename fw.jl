using LinearAlgebra

"""
    spectral_subgradient_hatPhi_paper(M, t)

Paper-faithful subgradient of the concave envelope hatΦ_t at M.
Handles eigenvalue ties exactly as in Proposition 2 / Remark 2.
"""
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

"""
    fw_aug_fact_paper(C, t_a, s, t)

Frank–Wolfe for Aug-Fact using the paper's exact subgradient.
"""
function fw_aug_fact_paper(
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
            println("FW converged at iteration $k, gap = $gap")
            return x
        end

        # --- Step size (standard) ---
        γ = 2.0 / (k + 2)

        # --- Update ---
        x .= (1 - γ) .* x .+ γ .* v
    end

    return x
end

# ============================================================
# Away-step Frank–Wolfe for Aug-Fact
# ============================================================
function afw_aug_fact_paper(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # Initial vertex (uniform convex combination not needed for AFW)
    x = fill(s / n, n)

    # Active set: vertices and weights
    active_vertices = Vector{Vector{Float64}}()
    weights = Float64[]

    push!(active_vertices, copy(x))
    push!(weights, 1.0)

    k = 0
    while true
        k += 1

        # --- Build M(x) ---
        M = At * Diagonal(x) * At'

        # --- Paper subgradient ---
        Y = spectral_subgradient_hatPhi_paper(M, t)

        # --- Gradient ---
        grad = diag(At' * Y * At)

        # --- FW vertex ---
        idx = sortperm(grad, rev=true)
        v_fw = zeros(n)
        v_fw[idx[1:s]] .= 1.0

        # --- Away vertex ---
        vals = [dot(grad, v) for v in active_vertices]
        ia = argmin(vals)
        v_away = active_vertices[ia]

        # --- Directions ---
        d_fw = v_fw .- x
        d_away = x .- v_away

        gap_fw = dot(grad, d_fw)
        gap_away = dot(grad, d_away)

        # --- Stopping ---
        if gap_fw ≤ tol
            println("AFW converged at iteration $k, gap = $gap_fw")
            return x
        end

        # --- Choose direction ---
        if gap_fw ≥ gap_away
            d = d_fw
            γmax = 1.0
            step_type = :FW
        else
            d = d_away
            γmax = weights[ia] / (1 - weights[ia] + eps())
            step_type = :Away
        end

        # --- Step size (standard AFW choice) ---
        γ = min(γmax, 2.0 / (k + 2))

        # --- Update x ---
        x .= x .+ γ .* d

        # --- Update active set ---
        if step_type == :FW
            found = false
            for (i, v) in enumerate(active_vertices)
                if norm(v - v_fw) ≤ 1e-12
                    weights[i] += γ
                    found = true
                    break
                end
            end
            if !found
                push!(active_vertices, v_fw)
                push!(weights, γ)
            end
            weights .*= (1 - γ)
        else
            weights[ia] -= γ
            weights .*= (1 + γ)
        end

        # --- Clean up tiny weights ---
        keep = [i for i in eachindex(weights) if weights[i] > 1e-12]
        active_vertices = active_vertices[keep]
        weights = weights[keep]

        # Normalize weights
        weights ./= sum(weights)
    end

    return x
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

function fw_aug_fact_exact_ls(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    x = fill(s / n, n)

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
            println("FW (exact LS) converged at iter $k, gap = $gap")
            return x
        end

        γ = fw_exact_line_search(x, d, At, t; γmax = 1.0)
        x .= x .+ γ .* d
    end

    return x
end

function afw_aug_fact_exact_ls(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    x = fill(s / n, n)

    active_vertices = [copy(x)]
    weights = [1.0]

    k = 0
    while true
        k += 1

        # --- Safety guard ---
        if isempty(active_vertices)
            active_vertices = [copy(x)]
            weights = [1.0]
        end

        # --- Gradient ---
        M = At * Diagonal(x) * At'
        Y = spectral_subgradient_hatPhi_paper(M, t)
        grad = diag(At' * Y * At)

        # --- FW vertex ---
        idx = sortperm(grad, rev=true)
        v_fw = zeros(n)
        v_fw[idx[1:s]] .= 1.0

        d_fw = v_fw .- x
        gap_fw = dot(grad, d_fw)

        if gap_fw ≤ tol
            println("AFW converged at iter $k, gap = $gap_fw")
            return x
        end

        # --- Away vertex ---
        vals = [dot(grad, v) for v in active_vertices]
        ia = argmin(vals)
        v_away = active_vertices[ia]

        d_away = x .- v_away
        gap_away = dot(grad, d_away)

        # --- Direction selection ---
        if gap_fw ≥ gap_away
            d = d_fw
            γmax = 1.0
            step = :FW
        else
            d = d_away
            γmax = weights[ia] / (1 - weights[ia] + eps())
            step = :Away
        end

        # --- Exact line search ---
        γ = fw_exact_line_search(x, d, At, t; γmax = γmax)

        # --- Update x ---
        x .= x .+ γ .* d

        # --- Active set update ---
        if step == :FW
            found = false
            for i in eachindex(active_vertices)
                if norm(active_vertices[i] - v_fw) ≤ 1e-12
                    weights[i] += γ
                    found = true
                    break
                end
            end
            if !found
                push!(active_vertices, v_fw)
                push!(weights, γ)
            end
            weights .*= (1 - γ)
        else
            weights[ia] -= γ
            weights .*= (1 + γ)
        end

        # --- Prune ---
        keep = findall(w -> w > 1e-12, weights)
        active_vertices = active_vertices[keep]
        weights = weights[keep]
        weights ./= sum(weights)
    end

    return x
end
