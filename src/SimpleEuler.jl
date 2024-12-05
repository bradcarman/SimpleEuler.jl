module SimpleEuler
using UnPack
using DiffEqBase
using ForwardDiff
using FiniteDiff

export BackwardEuler

@kwdef struct BackwardEuler <: DiffEqBase.AbstractODEAlgorithm
    relax=1.0
    max_iter=100
    order=1
    autodiff=true
    always_new=true
    ζ=0.0
end

DiffEqBase.isadaptive(::BackwardEuler) = false

function get_jacobian!(J::Matrix{T}, jac::Union{Function, Bool}, f::Function, u, t, p) where T <: Number
    if isa(jac, Function)
        jac(J, u, p, t)
    else
        f′ = SciMLBase.UJacobianWrapper(f, t, p)
        autodiff = jac
        if autodiff
            ForwardDiff.jacobian!(J, f′, u)
        else
            FiniteDiff.finite_difference_jacobian!(J, f′, u)
        end
    end
end

get_derivative(u, us, dt, order::Val{1}) = (u .- us[end]) ./ dt

function get_derivative(u, us, dt, order::Val{2})
    if length(us) > 1
        return (3*u .- 4*us[end] .+ us[end-1]) ./ (2*dt)
    else
        return get_derivative(u, us, dt, Val(1))
    end
end

function get_derivative(u, us, dt, order::Val{3})
    if length(us) > 2
        return (11*u .- 18*us[end] .+ 9*us[end-1] .- 2*us[end-2]) ./ (6*dt)
    else
        return get_derivative(u, us, dt, Val(2))
    end
end

function get_derivative(u, us, dt, order::Val{4})
    if length(us) > 3
        return (25*u .- (4*12)*us[end] .+ (3*12)*us[end-1] .- (4*12/3)*us[end-2]) .+ (12/4)*us[end-3] ./ (12*dt)
    else
        return get_derivative(u, us, dt, Val(3))
    end
end





function DiffEqBase.solve(prob::DiffEqBase.AbstractODEProblem{uType,tType,isinplace}, alg::BackwardEuler; dt=(prob.tspan[2] - prob.tspan[1]) / 100, tstops=tType[], abstol=1e-7, reltol=1e-7, kwargs...) where {uType,tType,isinplace} 

    @unpack u0, p, f, tspan = prob
    @unpack f, jac, mass_matrix = f
    @unpack relax, max_iter, order, always_new, ζ = alg

    if isnothing(jac)
        jac = alg.autodiff
    end

    c = length(u0)
    mass_matrix = mass_matrix*ones(c,c) #remove UniformScaling type
    z = du = du_ = rand(c)
    J = J_ = rand(c, c)

    if isempty(tstops)
        tstops = tspan[1]:dt:tspan[2]
    end
    @assert tstops[1] == tspan[1]

    nt = length(tstops)
    us = uType[]
    push!(us, copy(u0))

    # initialize ----------------------------
    u = copy(u0)
    f(du, u, p, tstops[1])
    get_jacobian!(J, jac, f, u, tstops[1], p)

    for i = 2:nt

        t = tstops[i]

        for j = 1:max_iter

            if always_new | (j == 1)
                get_jacobian!(J, jac, f, u, t, p)
            end

            J_ = J .- mass_matrix ./ dt
            f(du, u, p, t)
            du_ = du .- mass_matrix * get_derivative(u, us, dt, Val(order))
            try
                z .= (J_ .+ ζ) \ (du_ .+ ζ)    
            catch er
                @warn "$er on i=$i"
                break
            end
            
            u .-=  z .* relax

            # tmp = 
            # DiffEqBase.calculate_residuals!(tmp, du, u, abstol, reltol, t)

            tol = DiffEqBase.norm(du_, 2)
            if tol <= abstol
                # @show tol abstol
                break
            end

            if j == max_iter
                # @warn "Newton iterations reached max_iter=$(max_iter)"
            elseif any(isnan.(u))
                @warn "Newton steps could not converge"
                break
            end    
        end

        push!(us, copy(u))
    end

    return DiffEqBase.build_solution(prob, alg, tstops, us; ReturnCode.Success)
end


end # module SimpleImplicitEuler
