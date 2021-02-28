fcall(f::Function, x::AbstractVector{T}) where T<:Real = f(x)

# @code_warntype ✓
function ackley!(x::AbstractVector{T}) where T<:Real
    arg1 = 0.0; arg2 = 0.0; dims = length(x)

    @inbounds for i in eachindex(x)
        arg1 += x[i]^2.0; arg2 += cospi(2.0 * x[i])
    end

    arg1 = -0.2 * sqrt(arg1 / dims); arg2 /= dims

    return -20.0 * exp(arg1) - exp(arg2) + ℯ + 20.0
end

# @code_warntype ✓
logistic(x::Real, x₀::Real, a::Real, k::Real, offset::Real) = a / (1.0 + exp(k * (x₀ - x))) + offset
