using SimpleEuler
using Test
using OrdinaryDiffEq

dt = 1e-4
abstol = 1e-3
d=100
k=1000
F = 100
autodiff=true
always_new=true

function f0(du,u,p,t)
    F, k, d = p
    
    du[1] = (F - k*u[1]^2)/d
end

function f1(du,u,p,t)
    F, k, d = p
    x, dx = u
    
    du[1] = dx
    du[2] = (d*dx + k*x^2) - (F)
end

function f2(du,u,p,t)
    F, k, d = p
    x, dx, ddx = u
    
    du[1] = dx
    du[2] = ddx
    du[3] = (d*dx + k*x^2) - (F)
end

function f3(du,u,p,t)
    F, k, d = p
    x, dx, ddx, dddx = u
    
    du[1] = dx
    du[2] = ddx
    du[3] = dddx
    du[4] = (d*dx + k*x^2) - (F)
end

fmm0 = ODEFunction(f0)
fmm1 = ODEFunction(f1; mass_matrix=[1 0;0 0])
fmm2 = ODEFunction(f2; mass_matrix=[1 0 0;0 1 0;0 0 0])
fmm3 = ODEFunction(f3; mass_matrix=[1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 0])

prob0 = ODEProblem(fmm0, [0.0], (0.0, 10), [F, k, d])
prob1 = ODEProblem(fmm1, [0.0, F/d], (0.0, 10), [F, k, d])
prob2 = ODEProblem(fmm2, [0.0, F/d, 0.0], (0.0, 10), [F, k, d])
prob3 = ODEProblem(fmm3, [0.0, F/d, 0.0, 0.0], (0.0, 10), [F, k, d])

ref0 = solve(prob0, ImplicitEuler(;autodiff, nlsolve=NLNewton(;always_new)); abstol, dt, adaptive=false, initializealg=NoInit())
ref1 = solve(prob1, ImplicitEuler(;autodiff, nlsolve=NLNewton(;always_new)); abstol, dt, adaptive=false, initializealg=NoInit())
ref2 = solve(prob2, ImplicitEuler(;autodiff, nlsolve=NLNewton(;always_new, check_div=false)); abstol, dt, adaptive=false, initializealg=NoInit()) #always fails
ref3 = solve(prob3, ImplicitEuler(;autodiff, nlsolve=NLNewton(;always_new, check_div=false)); abstol, dt, adaptive=false, initializealg=NoInit()) #always fails

sol0 = solve(prob0, BackwardEuler(;autodiff, always_new); abstol, dt, adaptive=false)
sol1 = solve(prob1, BackwardEuler(;autodiff, always_new); abstol, dt, adaptive=false)
sol2 = solve(prob2, BackwardEuler(;autodiff, always_new); abstol, dt, adaptive=false)
sol3 = solve(prob3, BackwardEuler(;autodiff, always_new); abstol, dt, adaptive=false)

#=
using Plots

dref0 = [F/d; diff(ref0[1,:])./dt]
ddref0 = [0; diff(dref0)]./dt
ddref1 = [0; diff(ref1[2,:])]./dt

dsol0 = [F/d; diff(sol0[1,:])./dt]
ddsol0 = [0; diff(dsol0)]./dt
ddsol1 = [0; diff(sol1[2,:])]./dt

plot(ref0.t, ddref0; ylims=(-4,4))
plot!(ref1.t, ddref1)
plot!(ref2; idxs=3)
plot!(ref3; idxs=3)

plot!(sol0.t, ddsol0)
plot!(sol1.t, ddsol1)
plot!(sol2; idxs=3)
plot!(sol3; idxs=3)
=#