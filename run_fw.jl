using Random
using MAT

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
s = 20
t = 18

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
# Lower Bound
# ------------------------

indexes = [
    1, 7, 10, 15, 18,
    20, 24, 26, 30, 31,
    33, 36, 37, 40, 41,
    43, 44, 46, 47, 48
]

x_ref = zeros(n)
x_ref[indexes] .= 1.0

# -------------------------
# Run Frank-Wolfe
# -------------------------
runtime = @elapsed begin
    x, gap, k = fw_gaug_fact(
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
spectral_bound_val = spectral_bound(C, t)

println("--------------------")
println("Frank-Wolfe Exact LS stats:")
println("time: $runtime")
println("obj: $obj")
println("gap: $gap")
println("k: $k")
println("rp: $(abs(sum(x) - s))")
println("spectral bound: $spectral_bound_val")

# -------------------------
# Run Away-Step Frank-Wolfe
# -------------------------
runtime = @elapsed begin
    x, gap, k = afw_gaug_fact(
        C, t_a, s, t;
        tol = tol,
        use_standard_stepsize = true,
    )
end
obj = gaug_fact_objective(x, At, t, t_a)
spectral_bound_val = spectral_bound(C, t)
obj_ref = gaug_fact_objective(x_ref, At, t, t_a)

println("--------------------")
println("Away-Step Frank-Wolfe stats:")
println("time: $runtime")
println("obj: $obj")
println("gap: $gap")
println("k: $k")
println("rp: $(abs(sum(x) - s))")
println("spectral bound: $spectral_bound_val")
println("obj ref: $obj_ref")
println("rp ref: $(abs(sum(x_ref) - s))")
