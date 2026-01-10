using LinearAlgebra

function spectral_subgradient_hatPhi(M::Matrix{Float64}, t::Int, t_a::Float64)
    # --- Eigen-decomposition ---
    eig = eigen(Symmetric(M))
    λ = eig.values
    U = eig.vectors
    n = length(λ)

    # --- Sort eigenvalues descending ---
    perm = sortperm(λ, rev=true)
    λs = λ[perm]
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

function fw_exact_line_search(
    x::Vector{Float64},
    d::Vector{Float64},
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
        Mγ = At * Diagonal(xγ) * At'
        Yγ = spectral_subgradient_hatPhi(Mγ, t, t_a)

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

function fw_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # Initial feasible point
    x = fill(s / n, n)
    gap = 1e6
    γ = 0.0

    k = 0
    while true
        k += 1

        # --- Build M(x) ---
        M = At * Diagonal(x) * At'

        # --- Paper subgradient ---
        Y = spectral_subgradient_hatPhi(M, t, t_a)

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
            γ = fw_exact_line_search(x, d, At, t, t_a; γmax = 1.0)
        end

        # --- Update ---
        x .= (1 - γ) .* x .+ γ .* v
    end
end

function afw_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # ---- Initialize at a true vertex ----
    idx0 = randperm(n)[1:s]
    v0 = zeros(n)
    v0[idx0] .= 1.0

    x = copy(v0)
    active_vertices = [v0]
    active_weights  = [1.0]

    k = 0
    while true
        k += 1

        # ---- Gradient ----
        M = At * Diagonal(x) * At'
        Y = spectral_subgradient_hatPhi(M, t, t_a)
        grad = diag(At' * Y * At)

        # ============================================================
        # Frank–Wolfe (toward) direction
        # ============================================================
        idx_fw = sortperm(grad, rev = true)
        v_fw = zeros(n)
        v_fw[idx_fw[1:s]] .= 1.0

        d_fw   = v_fw - x
        gap_fw = dot(grad, d_fw)

        # ============================================================
        # Away direction
        # ============================================================
        gap_away = -Inf
        idx_away = 1
        v_away   = active_vertices[1]

        for (i, v) in enumerate(active_vertices)
            g = dot(grad, x - v)
            if g > gap_away
                gap_away = g
                idx_away = i
                v_away   = v
            end
        end

        d_away = x - v_away

        # ============================================================
        # Direction selection
        # ============================================================
        if gap_fw >= gap_away
            d = d_fw
            step_type = :FW
            γmax = 1.0
        else
            d = d_away
            step_type = :Away
            α = active_weights[idx_away]
            γmax = α / (1.0 - α + 1e-14)
        end

        gap = dot(grad, d)
        if gap ≤ tol
            return x, gap, k
        end

        # ============================================================
        # Step size
        # ============================================================
        if !line_search
            γ = min(2.0 / (k + 2), γmax)
        else
            γ = fw_exact_line_search(x, d, At, t, t_a; γmax = γmax)
        end

        # ============================================================
        # Update x
        # ============================================================
        x .= x .+ γ .* d

        # ============================================================
        # Update convex decomposition
        # ============================================================
        if step_type == :FW
            # Scale all existing weights
            active_weights .= (1 - γ) .* active_weights

            # Add or merge FW vertex
            found = false
            for i in eachindex(active_vertices)
                if active_vertices[i] == v_fw
                    active_weights[i] += γ
                    found = true
                    break
                end
            end
            if !found
                push!(active_vertices, v_fw)
                push!(active_weights, γ)
            end

        else
            # Away step:
            # α_j ← (1+γ) α_j   for all j
            # α_away ← α_away - γ
            active_weights .= (1 + γ) .* active_weights
            active_weights[idx_away] -= γ

            # Drop vertex if weight vanishes
            if active_weights[idx_away] ≤ 1e-12
                deleteat!(active_vertices, idx_away)
                deleteat!(active_weights, idx_away)
            end
        end

        # Numerical cleanup
        sα = sum(active_weights)
        active_weights ./= sα
    end
end

function pairwise_fw_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
    line_search::Bool = false,
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # ------------------------------------------------------------
    # Initialize at a true vertex
    # ------------------------------------------------------------
    idx0 = randperm(n)[1:s]
    v0 = zeros(n)
    v0[idx0] .= 1.0

    x = copy(v0)
    active_vertices = [v0]
    active_weights  = [1.0]

    k = 0
    while true
        k += 1

        # --------------------------------------------------------
        # Gradient
        # --------------------------------------------------------
        M = At * Diagonal(x) * At'
        Y = spectral_subgradient_hatPhi(M, t, t_a)
        grad = diag(At' * Y * At)

        # --------------------------------------------------------
        # FW vertex v+
        # --------------------------------------------------------
        idx_fw = sortperm(grad, rev = true)
        v_plus = zeros(n)
        v_plus[idx_fw[1:s]] .= 1.0

        # --------------------------------------------------------
        # Away vertex v- (worst active)
        # --------------------------------------------------------
        gap = -Inf
        idx_minus = 1
        v_minus = active_vertices[1]

        for (i, v) in enumerate(active_vertices)
            g = dot(grad, v_plus - v)
            if g > gap
                gap = g
                idx_minus = i
                v_minus = v
            end
        end

        if gap ≤ tol
            return x, gap, k
        end

        # --------------------------------------------------------
        # Pairwise direction
        # --------------------------------------------------------
        d = v_plus - v_minus
        γmax = active_weights[idx_minus]

        # --------------------------------------------------------
        # Step size
        # --------------------------------------------------------
        if !line_search
            γ = min(2.0 / (k + 2), γmax)
        else
            γ = fw_exact_line_search(x, d, At, t, t_a; γmax = γmax)
        end

        # --------------------------------------------------------
        # Update x
        # --------------------------------------------------------
        x .= x .+ γ .* d

        # --------------------------------------------------------
        # Update convex decomposition
        # --------------------------------------------------------
        # Remove weight from v-
        active_weights[idx_minus] -= γ
        if active_weights[idx_minus] ≤ 1e-12
            deleteat!(active_vertices, idx_minus)
            deleteat!(active_weights, idx_minus)
        end

        # Add weight to v+
        found = false
        for i in eachindex(active_vertices)
            if active_vertices[i] == v_plus
                active_weights[i] += γ
                found = true
                break
            end
        end
        if !found
            push!(active_vertices, v_plus)
            push!(active_weights, γ)
        end

        # Numerical normalization
        sα = sum(active_weights)
        active_weights ./= sα
    end
end

using JuMP
using Gurobi

function fw_fully_corrective_gaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    tol::Float64 = 1e-6,
)
    n = size(C, 1)
    At = cholesky(Symmetric(C - t_a * I)).U

    # Initial point
    x = fill(s / n, n)
    # Store active vertices
    X_vertices = [x]

    k = 0
    while true
        k += 1

        M = At * Diagonal(x) * At'
        Y = spectral_subgradient_hatPhi(M, t, t_a)
        grad = diag(At' * Y * At)

        # Linear Frank‑Wolfe direction
        idx_fw = sortperm(grad, rev=true)
        v_fw = zeros(n)
        v_fw[idx_fw[1:s]] .= 1.0

        d = v_fw - x
        gap = dot(grad, d)

        if gap ≤ tol
            return x, gap, k
        end

        # Add the new vertex
        push!(X_vertices, v_fw)
        m = length(X_vertices)

        # Build matrix of active vertices
        Vmat = hcat(X_vertices...)

        # Construct Gurobi model
        model = Model(Gurobi.Optimizer)
        set_silent(model)

        @variable(model, α[1:m] >= 0)
        @constraint(model, sum(α[i] for i=1:m) == 1)

        # Objective: linearized using gradient
        @objective(
            model, Min,
            -sum(grad' * Vmat[:, i] * α[i] for i=1:m)
        )

        optimize!(model)

        α_opt = value.(α)

        # Update x from optimal α
        x .= Vmat * α_opt
    end
end
