# ------------------------------------------------------------
# Post-processing script to convert jld2 to npz for cavity case
# ------------------------------------------------------------

using KitBase, NPZ
using KitBase.JLD2
using KitBase: ib_cavity

function extract_solution(ks, ctr)
    h = zeros(ks.ps.nx, ks.ps.ny, ks.vs.nu, ks.vs.nv)
    b = zero(h)
    w = zeros(ks.ps.nx, ks.ps.ny, 4)
    P = zeros(ks.ps.nx, ks.ps.ny, 2, 2)
    q = zeros(ks.ps.nx, ks.ps.ny, 2)

    for i in 1:ks.ps.nx
        for j in 1:ks.ps.ny
            w[i, j, :] .= ctr[i, j].w
            h[i, j, :, :] .= ctr[i, j].h
            b[i, j, :, :] .= ctr[i, j].b
            P[i, j, :, :] .=
                stress(ctr[i, j].h, ctr[i, j].prim, ks.vs.u, ks.vs.v, ks.vs.weights)
            q[i, j, :] .= heat_flux(
                ctr[i, j].h,
                ctr[i, j].b,
                ctr[i, j].prim,
                ks.vs.u,
                ks.vs.v,
                ks.vs.weights,
            )
        end
    end

    return h, b, w, P, q
end

begin
    set = Setup(;
        case="cacity",
        space="2d2f2v",
        boundary=["maxwell", "maxwell", "maxwell", "maxwell"],
        limiter="vanleer",
        cfl=0.8,
        maxTime=50,
    )
    ps = PSpace2D(0, 1, 50, 0, 1, 50)
    vs = VSpace2D(-5, 5, 48, -5, 5, 48)
    gas = Gas(; Kn=0.02, K=1.0)
    ib = IB2F(ib_cavity(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
end

cd(@__DIR__)
#Kns = collect(0.02:0.01:1.0)
Kns = collect(0.02:0.01:0.07)

ctrs = []
for i in eachindex(Kns)
    Kn = Kns[i]
    @load "cavity_Kn$(Kn).jld2" ctr
    push!(ctrs, ctr)
end

for i in eachindex(Kns)
    Kn = Kns[i]
    h, b, w, P, q = extract_solution(ks, ctrs[i])
    filename = "cavity_Kn$(Kn).npz"
    npzwrite(filename, Dict("h" => h, "b" => b, "w" => w, "P" => P, "q" => q, "Kn" => Kn))
end
