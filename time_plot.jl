using CSV
using DataFrames
using Plots

df = CSV.read("results.csv", DataFrame)

scatter(
    df.t, df.time,
    xlabel = "t",
    ylabel = "time (s)",
    legend = false,
    grid = :both,          # :x, :y, :both, or true
    gridalpha = 0.3,       # transparency
    gridlinewidth = 0.8,
)

savefig("scatter.png")
