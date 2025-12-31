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
# Spectral proximal operator of σ * Φ̂_t^*
# ============================================================
function spectral_prox_hatPhi_star!(
    Ynew::Matrix{Float64},
    W::Matrix{Float64},
    t::Int,
    σ::Float64;
    tol = 1e-12,
    maxiter = 100
)
    eig = eigen!(Symmetric(W))
    U = eig.vectors
    ω = eig.values
    n = length(ω)

    function trace_of_alpha(α)
        s = 0.0
        @inbounds for i in 1:n
            v = ω[i] - α
            if v > -2sqrt(σ)
                s += 0.5 * (v + sqrt(v*v + 4σ))
            end
        end
        return s
    end

    # --- Bracketing ---
    α_hi = maximum(ω)
    α_lo = α_hi - 1.0
    while trace_of_alpha(α_lo) < t
        α_lo -= max(1.0, abs(α_lo))
    end

    # --- Bisection ---
    for _ in 1:maxiter
        α = (α_lo + α_hi) / 2
        tr = trace_of_alpha(α)
        if tr > t
            α_lo = α
        else
            α_hi = α
        end
        if abs(tr - t) < tol
            break
        end
    end
    α = (α_lo + α_hi) / 2

    # --- Eigenvalue update ---
    ν = similar(ω)
    @inbounds for i in 1:n
        v = ω[i] - α
        ν[i] = v > -2sqrt(σ) ? 0.5 * (v + sqrt(v*v + 4σ)) : 0.0
    end

    # --- Reconstruction (NO aliasing) ---
    tmp = similar(Ynew)
    mul!(tmp, U, Diagonal(ν))
    mul!(Ynew, tmp, U')

    return Ynew
end

# ============================================================
# Projection onto box
# ============================================================
proj_box(x) = clamp.(x, 0.0, 1.0)

# ============================================================
# KKT stationarity residual
# ============================================================
function stationarity_residual(x, xnew, grad, τ)
    z = x .- τ .* grad
    return norm(grad .+ (xnew .- z) ./ τ)
end

# ============================================================
# Chambolle–Pock for Aug-Fact
# ============================================================
function cp_aug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int;
    τ::Float64,
    σ::Float64,
    θ::Float64,
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = compute_At(C, t_a)

    x   = fill(s / n, n)
    x̄  = copy(x)
    Y   = zeros(n, n)
    Ynew = similar(Y)
    λ   = 0.0

    while true
        # --- Dual update ---
        W = Y + σ * M_ta(x̄, At)
        spectral_prox_hatPhi_star!(Ynew, W, t, σ)
        λnew = λ + σ * (sum(x̄) - s)

        # --- Gradient ---
        grad = diag(At' * Ynew * At) .+ λnew

        # --- Primal update ---
        xnew = proj_box(x .- τ .* grad)

        # --- Residuals ---
        rp = abs(sum(xnew) - s)
        rd = stationarity_residual(x, xnew, grad, τ)

        if max(rp, rd) ≤ tol
            return xnew
        end

        # --- Extrapolation ---
        x̄ = xnew .+ θ .* (xnew .- x)
        x, Y, λ = xnew, Ynew, λnew
    end
end

# ============================================================
# Chambolle–Pock for CAug-Fact (Ax ≤ b)
# ============================================================
function cp_cgaug_fact(
    C::Matrix{Float64},
    t_a::Float64,
    s::Int,
    t::Int,
    A::Matrix{Float64},
    b::Vector{Float64};
    τ::Float64,
    σ::Float64,
    θ::Float64,
    tol::Float64 = 1e-6
)
    n = size(C, 1)
    At = compute_At(C, t_a)

    x   = fill(s / n, n)
    x̄  = copy(x)
    Y   = zeros(n, n)
    Ynew = similar(Y)
    λ   = 0.0
    μ   = zeros(size(A, 1))

    while true
        # --- Dual updates ---
        W = Y + σ * M_ta(x̄, At)
        spectral_prox_hatPhi_star!(Ynew, W, t, σ)
        λnew = λ + σ * (sum(x̄) - s)
        μnew = max.(μ .+ σ .* (A * x̄ .- b), 0.0)

        # --- Gradient ---
        grad = diag(At' * Ynew * At) .+ λnew .+ A' * μnew

        # --- Primal update ---
        xnew = proj_box(x .- τ .* grad)

        # --- Residuals ---
        rp = sqrt(
            (sum(xnew) - s)^2 +
            norm(max.(A * xnew .- b, 0.0))^2
        )
        rd = stationarity_residual(x, xnew, grad, τ)

        if max(rp, rd) ≤ tol
            return xnew
        end

        # --- Extrapolation ---
        x̄ = xnew .+ θ .* (xnew .- x)
        x, Y, λ, μ = xnew, Ynew, λnew, μnew
    end
end
