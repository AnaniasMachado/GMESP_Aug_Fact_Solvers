using Random
using MAT

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
s = 10
t = 5

matfile = matopen("data63.mat")
C = read(matfile, "A")
close(matfile)

# Choose t_a < λ_min(C)
t_a = 1.0 * minimum(eigvals(Symmetric(C)))

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

# -------------------------
# GAug-Fact Closed Form Solution
# -------------------------
# x_star, idx, g = closed_form_gaug_fact(C, t_a, s, t)
# x_star, scores, Sstar = gaug_fact_closed_form(C, t_a, s, t)
x_star, w = gaug_fact_closed_form(C, t_a, s, t)
obj_star = gaug_fact_objective(x_star, At, t, t_a)
sol_res = norm(x_star - x)
obj_res = obj_star - obj
println("--------------------")
println("obj star: $obj_star")
println("sol star rp: $(abs(sum(x_star) - s))")
println("sol res: $sol_res")
println("obj res: $obj_res")