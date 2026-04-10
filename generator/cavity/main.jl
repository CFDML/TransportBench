using KitBase
using KitBase.JLD2
using KitBase: ib_cavity

function simulator(Kn)
    set = Setup(;
        case="cacity",
        space="2d2f2v",
        boundary=["maxwell", "maxwell", "maxwell", "maxwell"],
        limiter="vanleer",
        cfl=0.8,
        maxTime=50,
    )
    ps = PSpace2D(0, 1, 50, 0, 1, 50)
    vs = VSpace2D(-5, 5, 80, -5, 5, 80)
    gas = Gas(; Kn=Kn, K=1.0)
    ib = IB2F(ib_cavity(set, ps, vs, gas)...)

    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, a1face, a2face = init_fvm(ks)

    dt = timestep(ks, ctr, 0.0)
    nt = ks.set.maxTime ÷ dt |> Int
    res = zeros(4)
    for iter in 1:nt
        evolve!(ks, ctr, a1face, a2face, dt)
        update!(ks, ctr, a1face, a2face, dt, res)

        if iter % 10 == 0
            print("iteration: $iter of $nt, residuals: ", res, "\n")
        end

        if maximum(res) < 1e-6
            break
        end
    end

    return ctr
end

function main()
    cd(@__DIR__)
    Kns = collect(0.02:0.01:1.0)
    for i in eachindex(Kns)
        Kn = Kns[i]
        @info "case $i, Kn = $Kn"
        ctr = simulator(Kn)
        @save "cavity_Kn$(Kn).jld2" ctr
    end
end

main()
