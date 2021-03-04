export minimize

module WCSCA # module

using ..Mathbrew: fcall, logistic
import Base: ==, isless

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Types
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

abstract type AbstractWCSCA{T}                          end
abstract type AbstractConstraint{T} <: AbstractWCSCA{T} end

mutable struct Water{T} <: AbstractWCSCA{T}
    x::Vector{T}; f::T; v::Bool; c::T
    #=
    x := parameters passed into models
    f := function-value of fn!(x)
    v := viability / feasibility
    c := contravention / violation
    =#

    # @code_warntype ✓
    Water(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real} = new{T}(rain(lb, ub), Inf, false, zero(T))
end

struct BoxBound{T} <: AbstractConstraint{T} a::T; b::T; i::Int end # a * x[i] + b

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Methods for `Water`
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

#### random initialization
# @code_warntype ✓
function rain(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real}
    if @generated
        e = Expr(:vect); a = Vector{Any}(undef, ND)

        @inbounds for i in eachindex(a)
            a[i] = :(lb[$i] + rand() * (ub[$i] - lb[$i]))
        end

        e.args = a

        return quote
            $(Expr(:meta, :inline))
            @inbounds return $e
        end
    else
        x = Vector{T}(undef, ND)

        @inbounds @simd for i in eachindex(x)
            x[i] = lb[i] + rand() * (ub[i] - lb[i])
        end

        return x
    end
end

#### groups and subgroups
# @code_warntype ✓
function waters(lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::Int) where {ND,T<:Real}
    wats = Vector{Water{T}}(undef, NP)

    @inbounds for i in eachindex(wats)
        wats[i] = Water(lb, ub)
    end

    return wats
end

rivers(wats::Vector{Water{T}}, NR::Int)          where T<:Real = view(wats, 1:NR)    # @code_warntype ✓
creeks(wats::Vector{Water{T}}, NR::Int, NP::Int) where T<:Real = view(wats, NR+1:NP) # @code_warntype ✓

#### sorting
# @code_warntype ✓
function ==(w1::Water{T}, w2::Water{T}) where T<:Real
    # w1, w2 are both feasible
    if w1.v && w2.v
        return w1.f == w2.f

    # w1, w2 are both infesasible
    elseif !w1.v && !w2.v
        return w1.c == w2.c

    else
        return false
    end
end

# @code_warntype ✓
function isless(w1::Water{T}, w2::Water{T}) where T<:Real
    # w1, w2 are both feasible
    if w1.v && w2.v
        return w1.f < w2.f

    # w1, w2 are both infesasible
    elseif !w1.v && !w2.v
        return w1.c < w2.c

    # if (w1, w2) = (feasible, infeasible), then w1 < w2
    # if (w1, w2) = (infeasible, feasible), then w2 < w1
    else
        return w1.v
    end
end

# @code_warntype ✓
function swap!(v::AbstractVector, i::Int, j::Int)
    v[i], v[j] = v[j], v[i]

    return nothing # to avoid a tuple-allocation
end

bilast_insert(arr::AbstractVector, val::T) where T = bilast_insert(arr, val, 1, length(arr)) # @code_warntype ✓
# @code_warntype ✓
function bilast_insert(arr::AbstractVector, val::T, ldx::Int, rdx::Int) where T
    if ldx ≥ rdx
        return ldx
    end

    upper_bound = rdx

    @inbounds while ldx < rdx
        mdx = (ldx + rdx) >> 1

        if val < arr[mdx]
            rdx = mdx
        else
            ldx = mdx + 1 # arr[mdx].f == val in this case
        end
    end

    if ldx == upper_bound && arr[ldx] ≤ val
        ldx += 1
    end

    return ldx
end

binary_insertsort!(arr::AbstractVector) = binary_insertsort!(arr, 1, length(arr)) # @code_warntype ✓
# @code_warntype ✓
function binary_insertsort!(arr::AbstractVector, ldx::Int, rdx::Int)
    @inbounds for idx in ldx+1:rdx
        val = arr[idx]
        loc = bilast_insert(arr, val, ldx, idx)

        jdx = idx

        while jdx > loc
            swap!(arr, jdx, jdx - 1); jdx -= 1
        end
    end

    return nothing
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Methods for `Constraint`
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

resolve_lb(lb::Real) = iszero(lb) ? (-1.0, 0.0) : (-abs(inv(lb)),  1.0 * sign(lb)) # @code_warntype ✓
resolve_ub(ub::Real) = iszero(ub) ? ( 1.0, 0.0) : ( abs(inv(ub)), -1.0 * sign(ub)) # @code_warntype ✓

# @code_warntype ✓
function boxBounds(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real}
    if @generated 
        e = Expr(:tuple); a = Vector{Any}(undef, 2 * ND)

        @inbounds for i in 1:ND
            a[i]      = :(BoxBound(resolve_lb(lb[$i])..., $i))
            a[i + ND] = :(BoxBound(resolve_ub(ub[$i])..., $i))
        end

        e.args = a

        return quote
            $(Expr(:meta, :inline))
            @inbounds return $e
        end
    else
        return ntuple(i -> i > ND ? BoxBound(resolve_ub(ub[i - ND])..., i - ND) : BoxBound(resolve_lb(lb[i])..., i), 2ND)
    end
end

eval_violation(x::AbstractVector{T}, bb::BoxBound) where T<:Real = max(0.0, bb.a * x[bb.i] + bb.b) # @code_warntype ✓
# @code_warntype ✓
function eval_violation(x::AbstractVector{T}, box::NTuple{NB,BoxBound}) where {NB,T<:Real}
    if @generated
        e = Expr(:call); a = Vector{Any}(undef, NB + 1); a[1] = :+

        @inbounds for i in 1:NB
            a[i + 1] = :(eval_violation(x, box[$i]))
        end

        e.args = a

        return quote
            $(Expr(:meta, :inline))
            @inbounds return $e
        end
    else
        violation = 0.0

        @inbounds for i in eachindex(box)
            violation += eval_violation(x, box[i])
        end

        return violation
    end
end

# check feasibility of "creeks", @code_warntype ✓
function check!(xnew::Vector{T}, wats::Vector{Water{T}}, rivs::AbstractVector{Water{T}}, crks::AbstractVector{Water{T}}, rdx::Int, cdx::Int, f::Function, cons::NTuple{NB,BoxBound}) where {NB,T<:Real}
    violation = eval_violation(xnew, cons)

    # x[new] is infeasible
    if violation > 0.0
        check!(xnew, violation, crks[cdx])

    # x[new] is feasible
    else
        fnew = fcall(f, xnew)
        check!(xnew, fnew, wats, rivs, crks, rdx, cdx)
    end

    return nothing
end

# Matchup for a feasible x[new] trial in "creeks", @code_warntype ✓
function check!(xnew::Vector{T}, fnew::Real, wats::Vector{Water{T}}, rivs::AbstractVector{Water{T}}, crks::AbstractVector{Water{T}}, rdx::Int, cdx::Int) where T<:Real
    @inbounds begin
        cwat = crks[cdx]

        # x[old] is infeasible
        if !cwat.v
            cwat.f = fnew
            cwat.v = true
            cwat.c = 0.0

            @simd for i in eachindex(xnew)
                cwat.x[i] = xnew[i]
            end

        # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
        elseif fnew ≤ rivs[rdx].f
            cwat.f = fnew

            @simd for i in eachindex(xnew)
                cwat.x[i] = xnew[i]
            end

            swap!(wats, rdx, length(rivs) + cdx)

        # x[old], x[new] are feasible
        elseif fnew ≤ cwat.f
            cwat.f = fnew

            @simd for i in eachindex(xnew)
                cwat.x[i] = xnew[i]
            end
        end
    end

    return nothing
end

# check feasibility of "rivers", @code_warntype ✓
function check!(xnew::Vector{T}, rivs::AbstractVector{Water{T}}, rdx::Int, f::Function, cons::NTuple{NB,BoxBound}) where {NB,T<:Real}
    violation = eval_violation(xnew, cons)

    # x[new] is infeasible
    if violation > 0.0
        check!(xnew, violation, rivs[rdx])

    # x[new] is feasible
    else
        fnew = fcall(f, xnew)

        check!(xnew, fnew, rivs, rdx)
    end

    return nothing
end

# Matchup for a feasible x[new] trial in "rivers", @code_warntype ✓
function check!(xnew::Vector{T}, fnew::Real, rivs::AbstractVector{Water{T}}, rdx::Int) where T<:Real
    @inbounds begin
        rwat = rivs[rdx]

        # x[old] is infeasible
        if !rwat.v
            rwat.f = fnew
            rwat.v = true
            rwat.c = 0.0

            @simd for i in eachindex(xnew)
                rwat.x[i] = xnew[i]
            end

        # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
        elseif fnew ≤ rivs[1].f
            rwat.f = fnew

            @simd for i in eachindex(xnew)
                rwat.x[i] = xnew[i]
            end

            swap!(rivs, 1, rdx)

        # x[old], x[new] are feasible
        elseif fnew ≤ rwat.f
            rwat.f = fnew

            @simd for i in eachindex(xnew)
                rwat.x[i] = xnew[i]
            end
        end
    end

    return nothing
end

# Matchup for an infeasible x[new] trial, here "fnew := violation", @code_warntype ✓
function check!(xnew::Vector{T}, violation::Real, wat::Water{T}) where T<:Real
    # x[old], x[new] are infeasible, compare violation
    # There is no `else` condition, if x[old] is feasible, then a matchup is unnecessary.
    if !wat.v && violation ≤ wat.c
        wat.c = violation

        @inbounds @simd for i in eachindex(xnew)
            wat.x[i] = xnew[i]
        end
    end

    return nothing
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Methods for `trials`
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

# @code_warntype ✓
function trial!(buff::AbstractVector{T}, wat1::AbstractVector{T}, wat2::AbstractVector{T}, ss::T) where T<:Real
    r = rand() * 2.0

    @inbounds @simd for i in eachindex(buff)
        buff[i] = wat1[i] + ss * abs(wat1[i] - wat2[i]) * ifelse(rand() < 0.5, sinpi(r), cospi(r)) # SCA
    end

    return nothing
end

# @code_warntype ✓
function rain!(buff::AbstractVector{T}, sea::AbstractVector{T}) where T<:Real
    r = randn()

    @inbounds @simd for i in eachindex(buff)
        buff[i] = sea[i] + r * 0.31622776601683794 # sqrt(0.1)
    end

    return nothing
end

# @code_warntype ✓
function rain!(buff::AbstractVector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real}
    @inbounds @simd for i in eachindex(buff)
        buff[i] = lb[i] + rand() * (ub[i] - lb[i])
    end

    return nothing
end

# @code_warntype ✓
function euclidean_distance(v1::AbstractVector{T}, v2::AbstractVector{T}, buff::AbstractVector{T}) where T<:Real
    @inbounds @simd for i in eachindex(buff)
        buff[i] = abs2(v1[i] - v2[i])
    end

    return sqrt(sum(buff))
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Methods for the whole searching
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

# @code_warntype ✓
function inits!(f::Function, cons::NTuple{NB,BoxBound}, wats::Vector{Water{T}}) where {NB,T<:Real}
    fmax = -Inf

    @inbounds begin
        for wat in wats
            violation = eval_violation(wat.x, cons)

            # wat is infeasible
            if violation > 0.0
                wat.c = violation

            else
                wat.v = true
                wat.f = fcall(f, wat.x)
                fmax  = max(fmax, wat.f)
            end
        end

        for wat in wats
            # wat is infeasible
            if !wat.v
                wat.f = wat.c + fmax
            end
        end
    end

    return nothing
end

# @code_warntype ✓
function group!(fork::Vector{Int}, wats::Vector{Water{T}}, NR::Int, NC::Int) where T<:Real
    diversity = 0.0; residue = NC; idx = 2

    @inbounds begin
        for i in eachindex(fork)
            diversity += wats[NR + 1].f - wats[i].f
        end

        if iszero(diversity) || isnan(diversity)
            fill!(fork, 1)
        else
            for i in eachindex(fork)
                fork[i] = max(1, round(Int, NC * (wats[NR + 1].f - wats[i].f) / diversity))
            end
        end

        residue -= sum(fork)

        while residue > 0
            fork[idx] += 1; residue -= 1

            idx < NR ? idx += 1 : idx = 2
        end

        while residue < 0
            fork[idx] = max(1, fork[idx] - 1); residue += 1

            idx < NR ? idx += 1 : idx = 2
        end
    end

    return nothing
end

# @code_warntype ✓
function minimize(f::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::Int, NR::Int, imax::Int, dmax::Real) where {ND,T<:Real}
    NC = NP - NR; ix = 0

    wats = waters(lb, ub, NP)
    rivs = rivers(wats, NR)
    crks = creeks(wats, NR, NP)

    cons = boxBounds(lb, ub)

    buff = Vector{T}(undef, ND)
    fork = Vector{Int}(undef, NR)

    inits!(f, cons, wats)
    binary_insertsort!(wats)

    sea = wats[1]

    @inbounds while ix < imax
        ix += 1; ss = logistic(ix, 0.5 * imax, -0.618, 20.0 / imax, 2.0)

        group!(fork, wats, NR, NC)

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Flowing creeks/rivers to rivers/sea
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        rdx = 1; fdx = fork[rdx]

        # flowing creeks to rivers
        for idx in eachindex(crks)
            trial!(buff, rivs[rdx].x, crks[idx].x, ss)
            check!(buff, wats, rivs, crks, rdx, idx, f, cons)

            fdx -= 1

            if iszero(fdx)
                rdx += 1; fdx = fork[rdx]
            end
        end

        # flowing rivers to sea == rivs[1]
        for rdx in 2:NR
            trial!(buff, rivs[1].x, rivs[rdx].x, ss)
            check!(buff, rivs, rdx, f, cons)
        end

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Raining/Evaporation process
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        for idx in 1:fork[1]
            if euclidean_distance(sea.x, crks[idx].x, buff) ≤ dmax
                rain!(buff, sea.x)
                check!(buff, wats, rivs, crks, 1, idx, f, cons)
            end
        end

        for rdx in 2:NR
            if euclidean_distance(sea.x, rivs[rdx].x, buff) ≤ dmax || rand() ≤ 0.1
                rain!(buff, lb, ub)
                check!(buff, rivs, rdx, f, cons)
            end
        end

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Renew process: update the function-value of infeasible candidates
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        fmax = -Inf

        for wat in wats
            if wat.v
                fmax = max(fmax, wat.f)
            end
        end

        for wat in wats
            if !wat.v
                wat.f = wat.c + fmax
            end
        end

        binary_insertsort!(wats); dmax -= dmax / imax
    end

    return sea
end

end # module

import .WCSCA

# @code_warntype ✓
function minimize(f::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}; NP::Int=0, NR::Int=0, imax::Int=0, dmax::Real=NaN) where {ND,T<:Real}
    NP   = iszero(NP)   ? 35 * ND  : NP
    NR   = iszero(NR)   ? ND + 1   : NR
    imax = iszero(imax) ? 210 * ND : imax
    dmax = isnan(dmax)  ? 1e-7     : dmax
    return WCSCA.minimize(f, lb, ub, NP, NR, imax, dmax)
end
