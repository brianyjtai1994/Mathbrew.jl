module Mathbrew

abstract type AbstractMathbrew{T} end

include("./model/model.jl")
include("./wcsca.jl")
include("./curvefit.jl")

end # module
