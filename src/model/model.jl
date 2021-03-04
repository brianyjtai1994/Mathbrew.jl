export decay, decay!

fcall(f::Function, x::AbstractVector{<:Real}) = f(x)

# @code_warntype ✓
function ackley(x::AbstractVector{<:Real})
    arg1 = 0.0; arg2 = 0.0; dims = length(x)

    @inbounds for i in eachindex(x)
        arg1 += abs2(x[i]); arg2 += cospi(2.0 * x[i])
    end

    arg1 = -0.2 * sqrt(arg1 / dims); arg2 /= dims

    return -20.0 * exp(arg1) - exp(arg2) + ℯ + 20.0
end

# @code_warntype ✓
logistic(x::Real, x₀::Real, a::Real, k::Real, offset::Real) = a / (1.0 + exp(k * (x₀ - x))) + offset

decay(x::Real, a::Real, τ::Real) = a * exp(- x / τ)

# @code_warntype ✓
function decay(x::Real, p::AbstractVector{<:Real})
    @inbounds begin
        r = p[end]

        for i in 1:length(p) >> 1
            r += decay(x, p[2i-1], p[2i])
        end
    end

    return r
end

# @code_warntype ✓
function decay!(y::AbstractVector{<:Real}, x::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    N = length(p)
    M = N >> 1
    @inbounds begin
        @simd for n in eachindex(y)
            y[n] = p[end]
        end

        for i in 1:M
            j = 2i - 1
            k = 2i
            @simd for n in eachindex(y)
                y[n] += decay(x[n], p[j], p[k])
            end
        end
    end
    return nothing
end
