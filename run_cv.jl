using Random

include("cv.jl")

Random.seed!(2)

function random_pd_matrix(n::Int)
    X = randn(n, n)
    return X' * X + 0.1I
end

n = 100
C = random_pd_matrix(n)

s = 5                    # sum(x) = s
t = 3                    # number of greatest eigenvalues to compute
ta = 0.05                # shift for C - tI

# Linear inequality constraints Ax <= b
m = 20
A = rand(m, n)
b = A * rand(n)

L = n + opnorm(A)^2
τ = 0.99 / sqrt(L)
σ = 0.99 / sqrt(L)

cv_runtime = @elapsed begin
    # xsol = condat_vu_cgmesp(C, t, ta; s=s, Aineq=A, bineq=b, τ=τ, σ=σ)
    xsol = condat_vu_gmesp(C, t, ta; s=s, τ=τ, σ=σ)
end

Ata = cholesky(C - ta * I).L
a_cols = [Ata[:, i] for i in 1:n]

g, grad = spectral_obj_grad(xsol, a_cols, s, t)

println("\n================ Condat–Vũ Results ================\n")

println("Objective value:")
println("  f(x) = ", g)

println("\nRuntime:")
println("  elapsed time = ", round(cv_runtime, digits=4), " seconds")

println("\nSolution statistics:")
println("  sum(x)           = ", sum(xsol))
println("  min(x), max(x)   = ", minimum(xsol), ", ", maximum(xsol))

println("\nConstraint violations:")
println("  equality (|sum(x) - s|) = ", abs(sum(xsol) - s))
println("  inequality (max(Ax - b, 0)) = ",
        maximum(max.(A * xsol .- b, 0.0)))

println("\nSolution vector (rounded):")
println(round.(xsol, digits=4))

println("\n===================================================\n")