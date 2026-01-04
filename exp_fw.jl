using Random
using MAT
using CSV
using DataFrames

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
s = 40
t_vals = [i for i in 1:s]

matfile = matopen("data63.mat")
C = read(matfile, "A")
close(matfile)

C = Matrix{Float64}(C)

# Choose t_a < λ_min(C)
t_a = minimum(eigvals(Symmetric(C)))

# -------------------------
# Compute A(t_a)
# -------------------------
At = compute_At(C, t_a)

# -------------------------
# Tolerance
# -------------------------
tol = 1e-3

# -------------------------
# Warm-up phase (CRUCIAL)
# -------------------------
println("Warming up (compiling methods)...")

t_warm = t_vals[1]

fw_gaug_fact(C, t_a, s, t_warm; tol=tol)
fw_gaug_fact_exact_ls(C, t_a, s, t_warm; tol=tol)
afw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=false)
afw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=true)
pairwise_fw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=false)
pairwise_fw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=true)

# also warm up objective / bounds
x_dummy = fill(s / n, n)
gaug_fact_objective(x_dummy, At, t_warm, t_a)
spectral_bound(C, t_warm)

println("Warm-up complete.\n")

# -------------------------
# Data Collection
# -------------------------
df = DataFrame()
results_filepath = "results_s$s.csv"

for t in t_vals
    println("--------------------")
    println("t: $t")

    # -------------------------
    # Run Frank-Wolfe
    # -------------------------
    fw_runtime = @elapsed begin
        fw_x, fw_gap, fw_k = fw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = false,
        )
    end
    fw_obj = gaug_fact_objective(fw_x, At, t, t_a)

    println("--------------------")
    println("Frank-Wolfe stats:")
    println("time: $fw_runtime")
    println("obj: $fw_obj")
    println("gap: $fw_gap")
    println("k: $fw_k")
    println("rp: $(abs(sum(fw_x) - s))")

    # -------------------------
    # Run Frank-Wolfe Exact LS
    # -------------------------
    fwls_runtime = @elapsed begin
        fwls_x, fwls_gap, fwls_k = fw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = true,
        )
    end
    fwls_obj = gaug_fact_objective(fwls_x, At, t, t_a)
    spectral_bound_val = spectral_bound(C, t)

    println("--------------------")
    println("Frank-Wolfe Exact LS stats:")
    println("time: $fwls_runtime")
    println("obj: $fwls_obj")
    println("gap: $fwls_gap")
    println("k: $fwls_k")
    println("rp: $(abs(sum(fwls_x) - s))")


    # -------------------------
    # Run Away-Step Frank-Wolfe
    # -------------------------
    asfw_runtime = @elapsed begin
        asfw_x, asfw_gap, asfw_k = afw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = false,
        )
    end
    asfw_obj = gaug_fact_objective(asfw_x, At, t, t_a)

    println("--------------------")
    println("Away-Step Frank-Wolfe stats:")
    println("time: $asfw_runtime")
    println("obj: $asfw_obj")
    println("gap: $asfw_gap")
    println("k: $asfw_k")
    println("rp: $(abs(sum(asfw_x) - s))")

    # -------------------------
    # Run Away-Step Frank-Wolfe Exact LS
    # -------------------------
    asfwls_runtime = @elapsed begin
        asfwls_x, asfwls_gap, asfwls_k = afw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = true,
        )
    end
    asfwls_obj = gaug_fact_objective(asfwls_x, At, t, t_a)

    println("--------------------")
    println("Away-Step Frank-Wolfe Exact LS stats:")
    println("time: $asfwls_runtime")
    println("obj: $asfwls_obj")
    println("gap: $asfwls_gap")
    println("k: $asfwls_k")
    println("rp: $(abs(sum(asfwls_x) - s))")
    println("spectral bound: $spectral_bound_val")

    # -------------------------
    # Run Pairwise Frank-Wolfe
    # -------------------------
    pwfw_runtime = @elapsed begin
        pwfw_x, pwfw_gap, pwfw_k = pairwise_fw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = false,
        )
    end
    pwfw_obj = gaug_fact_objective(pwfw_x, At, t, t_a)

    println("--------------------")
    println("Pairwise Frank-Wolfe stats:")
    println("time: $pwfw_runtime")
    println("obj: $pwfw_obj")
    println("gap: $pwfw_gap")
    println("k: $pwfw_k")
    println("rp: $(abs(sum(pwfw_x) - s))")

    # -------------------------
    # Run Pairwise Frank-Wolfe Exact LS
    # -------------------------
    pwfwls_runtime = @elapsed begin
        pwfwls_x, pwfwls_gap, pwfwls_k = pairwise_fw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = true,
        )
    end
    pwfwls_obj = gaug_fact_objective(pwfwls_x, At, t, t_a)

    println("--------------------")
    println("Pairwise Frank-Wolfe Exact LS stats:")
    println("time: $pwfwls_runtime")
    println("obj: $pwfwls_obj")
    println("gap: $pwfwls_gap")
    println("k: $pwfwls_k")
    println("rp: $(abs(sum(pwfwls_x) - s))")
    println("spectral bound: $spectral_bound_val")

    result = DataFrame(
        n = n,
        s = s,
        t = t,
        fw_obj = fw_obj,
        fw_gap = fw_gap,
        fw_k = fw_k,
        fw_runtime = fw_runtime,
        fwls_obj = fwls_obj,
        fwls_gap = fwls_gap,
        fwls_k = fwls_k,
        fwls_runtime = fwls_runtime,
        asfw_obj = asfw_obj,
        asfw_gap = asfw_gap,
        asfw_k = asfw_k,
        asfw_runtime = asfw_runtime,
        asfwls_obj = asfwls_obj,
        asfwls_gap = asfwls_gap,
        asfwls_k = asfwls_k,
        asfwls_runtime = asfwls_runtime,
        pwfw_obj = pwfw_obj,
        pwfw_gap = pwfw_gap,
        pwfw_k = pwfw_k,
        pwfw_runtime = pwfw_runtime,
        pwfwls_obj = pwfwls_obj,
        pwfwls_gap = pwfwls_gap,
        pwfwls_k = pwfwls_k,
        pwfwls_runtime = pwfwls_runtime,
        spectral_bound_val = spectral_bound_val,
    )

    append!(df, result)
end

CSV.write(results_filepath, df)
