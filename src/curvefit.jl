export decay_fit

abstract type AbstractFit{T} <: Function       end
abstract type AbstractLSQ{T} <: AbstractFit{T} end
abstract type AbstractχSQ{T} <: AbstractFit{T} end

# @code_warntype ✓
function leastsq(f!::Function, p::AbstractVector, x::AbstractVector, y::AbstractVector, Δ::AbstractVector)
    f!(Δ, x, p)
    @inbounds @simd for i in eachindex(Δ)
        Δ[i] = abs2(y[i] - Δ[i])
    end
    return sum(Δ)
end

# @code_warntype ✓
function leastsq(f!::Function, p::AbstractVector, x::AbstractVector, y::AbstractVector, w::AbstractVector, Δ::AbstractVector)
    f!(Δ, x, p)
    @inbounds @simd for i in eachindex(Δ)
        Δ[i] = w[i] * abs2(y[i] - Δ[i])
    end
    return sum(Δ)
end

# @code_warntype ✓
function curve_fit(sq::AbstractFit, lb::NTuple{ND,T}, ub::NTuple{ND,T}; NP::Int=0, NR::Int=0, imax::Int=0, dmax::Real=NaN) where {ND,T<:Real}
    NP   = iszero(NP)   ? 30 * ND  : NP
    NR   = iszero(NR)   ? ND + 1   : NR
    imax = iszero(imax) ? 180 * ND : imax
    dmax = isnan(dmax)  ? 1e-7     : dmax
    return WCSCA.minimize(sq, lb, ub, NP, NR, imax, dmax)
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Multi-Exponential Decay LeastSQ
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
struct DecayLSQ{T} <: AbstractLSQ{T}
    x::Vector{T}; y::Vector{T}; Δ::Vector{T}
    DecayLSQ(x::Vector{T}, y::Vector{T}) where T<:Real = new{T}(x, y, similar(y))
end

struct DecayχSQ{T} <: AbstractχSQ{T}
    x::Vector{T}; y::Vector{T}; w::Vector{T}; Δ::Vector{T}
    DecayχSQ(x::Vector{T}, y::Vector{T}, w::Vector{T}) where T<:Real = new{T}(x, y, w, similar(y))
end

fcall(f::DecayLSQ, p::AbstractVector{<:Real}) = leastsq(decay!, p, f.x, f.y, f.Δ)      # @code_warntype ✓
fcall(f::DecayχSQ, p::AbstractVector{<:Real}) = leastsq(decay!, p, f.x, f.y, f.w, f.Δ) # @code_warntype ✓

# @code_warntype ✓
function decay_fit(x::Vector{T}, y::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real}
    if ND & 1 ≠ 1
        return error("Invalid boundary dimension (odd number for multi-exponential decay).")
    end

    return curve_fit(DecayLSQ(x, y), lb, ub)
end

# @code_warntype ✓
function decay_fit(x::Vector{T}, y::Vector{T}, w::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND,T<:Real}
    if ND & 1 ≠ 1
        return error("Invalid boundary dimension (odd number for multi-exponential decay).")
    end

    return curve_fit(DecayχSQ(x, y, w), lb, ub)
end
