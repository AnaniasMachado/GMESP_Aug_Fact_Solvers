using Random

include("cp.jl")
include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 100
s = 10
t = 5

Random.seed!(2)
C = randn(n, n)
C = C' * C + I

# Choose t_a < λ_min(C)
t_a = 0.9 * minimum(eigvals(Symmetric(C)))

# -------------------------
# Compute A(t_a)
# -------------------------
At = compute_At(C, t_a)

# -------------------------
# Step-size computation
# -------------------------
# Bound on ||K||^2
L = sum(norm(view(At, :, i))^4 for i in 1:n) + n

τ = 0.99 / sqrt(L)
σ = 0.99 / sqrt(L)

# Chambolle–Pock extrapolation parameter
θ = 1.0

tol = 1e-3

# println("τ = $τ, σ = $σ, θ = $θ")

# -------------------------
# Run Chambolle–Pock
# -------------------------
# runtime = @elapsed begin
#     x = cp_aug_fact(
#         C, t_a, s, t;
#         τ = τ,
#         σ = σ,
#         θ = θ,
#         tol = tol
#     )
# end
# obj = aug_fact_objective(x, At, t, t_a)

# println("--------------------")
# println("Chambolle-Pock stats:")
# println("time: $runtime")
# println("obj: $obj")
# println("rp: $(abs(sum(x) - s))")

# -------------------------
# Run Frank-Wolfe
# -------------------------
runtime = @elapsed begin
    x = fw_aug_fact_paper(
        C, t_a, s, t;
        tol = tol
    )
end
obj = aug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Frank-Wolfe stats:")
println("time: $runtime")
println("obj: $obj")
println("rp: $(abs(sum(x) - s))")

# -------------------------
# Run Away-step Frank-Wolfe
# -------------------------
runtime = @elapsed begin
    x = afw_aug_fact_paper(
        C, t_a, s, t;
        tol = tol
    )
end
obj = aug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Away-step Frank-Wolfe stats:")
println("time: $runtime")
println("obj: $obj")
println("rp: $(abs(sum(x) - s))")

# -------------------------
# Run Frank-Wolfe Exact LS
# -------------------------
runtime = @elapsed begin
    x = fw_aug_fact_exact_ls(
        C, t_a, s, t;
        tol = tol
    )
end
obj = aug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Frank-Wolfe Exact LS stats:")
println("time: $runtime")
println("obj: $obj")
println("rp: $(abs(sum(x) - s))")

# -------------------------
# Run Away-step Frank-Wolfe Exact LS
# -------------------------
runtime = @elapsed begin
    x = afw_aug_fact_exact_ls(
        C, t_a, s, t;
        tol = tol
    )
end
obj = aug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Away-step Frank-Wolfe Exact LS stats:")
println("time: $runtime")
println("obj: $obj")
println("rp: $(abs(sum(x) - s))")