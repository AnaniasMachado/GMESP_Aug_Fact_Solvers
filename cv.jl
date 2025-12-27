using LinearAlgebra, Arpack

# -------------------------------------------------------
# Spectral objective and gradient
# -------------------------------------------------------
function spectral_obj_grad(x, a_cols, s, t)
    n = length(x)
    d = length(a_cols[1])

    M = zeros(d, d)
    @inbounds for i in 1:n
        M .+= x[i] * (a_cols[i] * a_cols[i]')
    end

    λ, U = eigs(Symmetric(M), nev=s, which=:LM)

    f = -sum(log.(λ .+ t))

    grad = zeros(n)
    @inbounds for i in 1:n
        ai = a_cols[i]
        for j in 1:s
            grad[i] -= (ai' * U[:, j])^2 / (λ[j] + t)
        end
    end

    return f, grad
end

# -------------------------------------------------------
# Condat–Vũ
# -------------------------------------------------------
function condat_vu_gmesp(
    C, t, ta;
    s::Int,
    τ::Float64,
    σ::Float64,
    tol::Float64 = 1e-6,
)

    n = size(C, 1)

    # Cholesky factor of C - tI
    Ata = cholesky(C - ta * I).L
    a_cols = [Ata[:, i] for i in 1:n]

    # Initialization
    x = fill(s / n, n)
    x_old = similar(x)
    z = similar(x)
    KTy = similar(x)

    y1 = 0.0
    y1_old = 0.0

    k = 0
    while true
        # ----- Primal residual (feasibility violation) -----
        primal_res = sqrt((sum(x) - s)^2)

        # ----- Dual residual (stationarity violation) -----
        _, grad = spectral_obj_grad(x, a_cols, t, ta)

        normal_cone_term = (1 / τ) .* (z .- x)

        dual_res = norm(grad .+ KTy .+ normal_cone_term)

        res = max(primal_res, dual_res)

        if res < tol
            println("CV converged at iter $k, residual = $res")
            return x
        end
        
        # Save previous iterates
        copy!(x_old, x)
        y1_old = y1

        # ----- Dual update -----
        y1 += σ * (sum(x) - s)

        # ----- Primal update -----
        KTy = y1 .* ones(n)
        z = x .- τ .* (grad .+ KTy)
        x .= clamp.(z, 0.0, 1.0)

        # Update iteration counter
        k += 1
    end

    return x
end

function condat_vu_cgmesp(
    C, t, ta;
    s::Int,
    Aineq::Matrix,
    bineq::Vector,
    τ::Float64,
    σ::Float64,
    tol::Float64 = 1e-6,
)

    n = size(C, 1)
    m = size(Aineq, 1)

    # Cholesky factor of C - tI
    Ata = cholesky(C - ta * I).L
    a_cols = [Ata[:, i] for i in 1:n]

    # Initialization
    x = fill(s / n, n)
    x_old = similar(x)
    z = similar(x)
    KTy = similar(x)

    y1 = 0.0
    y2 = zeros(m)
    y1_old = 0.0
    y2_old = similar(y2)

    k = 0
    while true
        # ----- Primal residual (feasibility violation) -----
        primal_res = sqrt(
            (sum(x) - s)^2 +
            norm(max.(Aineq * x .- bineq, 0.0))^2
        )

        # ----- Dual residual (stationarity violation) -----
        _, grad = spectral_obj_grad(x, a_cols, t, ta)

        normal_cone_term = (1 / τ) .* (z .- x)

        dual_res = norm(grad .+ KTy .+ normal_cone_term)

        res = max(primal_res, dual_res)

        if res < tol
            println("CV converged at iter $k, residual = $res")
            return x
        end
        
        # Save previous iterates
        copy!(x_old, x)
        y1_old = y1
        copy!(y2_old, y2)

        # ----- Dual update -----
        y1 += σ * (sum(x) - s)
        y2 .= max.(y2 .+ σ .* (Aineq * x .- bineq), 0.0)

        # ----- Primal update -----
        KTy = y1 .* ones(n) .+ Aineq' * y2
        z = x .- τ .* (grad .+ KTy)
        x .= clamp.(z, 0.0, 1.0)

        # Update iteration counter
        k += 1
    end

    return x
end
