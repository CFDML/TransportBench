using NPZ

cd(@__DIR__)
ps = npzread("setup.npz")
vars = npzread("cavity_Kn0.95.npz")

using Plots
contourf(ps["x"][:, 1], ps["y"][1, :], vars["w"][:, :, 1]'; xlabel="x", ylabel="y")
