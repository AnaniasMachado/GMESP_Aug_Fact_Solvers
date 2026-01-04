using CSV
using DataFrames
using Plots

s = 40

df = CSV.read("results_s$s.csv", DataFrame)

scatter(
    df.t, df.fw_runtime;
    label = "FW",
    yscale = :log10,
    xlabel = "t",
    ylabel = "time (s)"
)

# scatter!(df.t, df.fwls_runtime; label = "FW-LS")

scatter!(df.t, df.asfw_runtime; label = "ASFW")
# scatter!(df.t, df.asfwls_runtime; label = "ASFW-LS")

scatter!(df.t, df.pwfw_runtime; label = "PWFW")
# scatter!(df.t, df.pwfwls_runtime; label = "PWFW-LS")

savefig("scatter_s$s.png")
