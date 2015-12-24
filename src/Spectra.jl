module Spectra

export AbstractMatOp, DenseMatProd
export symeigs

include("hessenqr.jl")
include("symeigs.jl")

end # module
