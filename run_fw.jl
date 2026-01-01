using Random

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
# Tolerance
# -------------------------

tol = 1e-3

# -------------------------
# Run Frank-Wolfe
# -------------------------
runtime = @elapsed begin
    x, gap, k = fw_gaug_fact_paper(
        C, t_a, s, t;
        tol = tol
    )
end
obj = gaug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Frank-Wolfe stats:")
println("time: $runtime")
println("obj: $obj")
println("gap: $gap")
println("k: $k")
println("rp: $(abs(sum(x) - s))")

# -------------------------
# Run Frank-Wolfe Exact LS
# -------------------------
runtime = @elapsed begin
    x, gap, k = fw_gaug_fact_exact_ls(
        C, t_a, s, t;
        tol = tol
    )
end
obj = gaug_fact_objective(x, At, t, t_a)

println("--------------------")
println("Frank-Wolfe Exact LS stats:")
println("time: $runtime")
println("obj: $obj")
println("gap: $gap")
println("k: $k")
println("rp: $(abs(sum(x) - s))")
