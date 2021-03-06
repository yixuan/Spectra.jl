## Generic matrix operation
abstract AbstractMatOp{T}

## Dense matrix multiplication
immutable DenseMatProd{T} <: AbstractMatOp{T}
    mat::Matrix{T}
end
## The operation
performop{T}(A::DenseMatProd{T}, x::Vector{T}) = A.mat * x
## Size of the matrix
nrow{T}(A::DenseMatProd{T}) = size(A.mat, 1)



## Arnoldi factorization starting from step-k
function arnoldifac!{T}(k::Int, m::Int, V::Matrix{T}, H::Matrix{T}, f::Vector{T}, A::AbstractMatOp{T}, prec::T)
    if m <= k
        error("m must be greater than k")
    end

    ## Keep the upperleft k x k submatrix of H and
    ## set other elements to 0
    H[:, (k + 1):end] = zero(T)
    H[(k + 1):end, 1:k] = zero(T)

    beta::T = norm(f)

    for i = k:(m - 1)
        ## V{i+1} <- f / ||f||
        V[:, i + 1] = f / beta
        H[i + 1, i] = beta

        ## w <- A * V{i+1}
        w::Vector{T} = performop(A, V[:, i + 1])

        H[i, i + 1] = beta
        H[i + 1, i + 1] = dot(V[:, i + 1], w)

        ## f <- w - V * V' * w
        f[:] = w - beta * V[:, i] - H[i + 1, i + 1] * V[:, i + 1]
        beta = norm(f)

        ## f/||f|| is going to be the next column of V, so we need to test
        ## whether V' * (f/||f||) ~= 0
        Vf::Vector{T} = V[:, 1:(i + 1)]' * f
        count = 0
        while count < 5 && maximum(abs(Vf)) > prec * beta
            ## f <- f - V * Vf
            f[:] -= V[:, 1:(i + 1)] * Vf
            ## h <- h + Vf
            H[i, i + 1] += Vf[i]
            H[i + 1, i] = H[i, i + 1]
            H[i + 1, i + 1] += Vf[i + 1]
            ## beta <- ||f||
            beta = norm(f)

            Vf[:] = V[:, 1:(i + 1)]' * f
        end
    end
end

## Apply shifts on V, H and f
function applyshifts!{T}(k::Int, V::Matrix{T}, H::Matrix{T}, f::Vector{T}, shifts::Vector{T})
    n = size(V, 1)
    ncv = size(V, 2)
    Q::Matrix{T} = eye(T, ncv)

    for i = (k + 1):ncv
        ## QR decomposition of H-mu*I, mu is the shift
        for j = 1:ncv
            H[j, j] -= shifts[i]
        end
        qr = tridiagqr!(H)  ## H -> RQ
        applyright!(qr, Q)  ## Q -> Q * Qi
        for j = 1:ncv
            H[j, j] += shifts[i]
        end
    end

    ## V -> VQ, only need to update the first k+1 columns
    ## Q has some elements being zero
    ## The first (ncv - k + i) elements of the i-th column of Q are non-zero
    # Vs = zeros(T, n, k + 1)
    # for i = 1:k
    #     nnz = ncv - k + i
    #     Vs[:, i] = V[:, 1:nnz] * Q[1:nnz, i]
    # end
    # Vs[:, k + 1] = V * Q[:, k + 1]
    # V[:, 1:(k + 1)] = Vs
    ## However this seems to be slower than a direct multiplication

    ## V -> VQ
    V[:, :] = V * Q
    f[:] = f * Q[ncv, k] + V[:, k + 1] * H[k + 1, k]
end

## Retrieve Ritz values and Ritz vectors
function ritzpairs!{T}(which::Symbol, H::Matrix{T}, ritzval::Vector{T}, ritzvec::Matrix{T}, ritzest::Vector{T})
    ncv = size(ritzvec, 1)
    nev = size(ritzvec, 2)
    ## Eigen decomposition on H, which is symmetric and tridiagonal
    decomp = eigfact(SymTridiagonal(diag(H), diag(H, -1)))

    ## Sort Ritz values according to "which"
    if which == :LM
        trans = abs
        rev = true
    elseif which == :SM
        trans = abs
        rev = false
    elseif which == :LA || which == :BE
        trans = (x -> x)
        rev = true
    else which == :SA
        trans = (x -> x)
        rev = false
    end

    ix = sortperm(decomp.values, by = trans, rev = rev)

    if which == :BE
        ixcp = copy(ix)
        for i = 1:ncv
            if i % 2 == 1
                ix[i] = ixcp[(i + 1) / 2]
            else
                ix[i] = ixcp[ncv - i / 2 + 1]
            end
        end
    end

    ritzval[:] = decomp.values[ix]
    ritzest[:] = decomp.vectors[ncv, ix]
    ritzvec[:, :] = decomp.vectors[:, ix[1:nev]]
end

## Adjusted nev
function nevadjusted{T}(nev::Int, ncv::Int, nconv::Int, ritzest::Vector{T}, prec::T)
    nevnew = nev + sum(abs(ritzest[(nev + 1):ncv]) .< prec)
    nevnew += min(nconv, div(ncv - nevnew, 2))
    if nevnew == 1 && ncv >= 6
        nevnew = div(ncv, 2)
    elseif nevnew == 1 && ncv > 2
        nevnew = 2
    end

    return nevnew
end


function symeigs{T}(A::AbstractMatOp{T};
                    nev::Int = 6, ncv::Int = min(nrow(A), max(2 * nev + 1, 20)),
                    which::Symbol = :LM,
                    tol::T = 1e-8, maxiter::Int = 300,
                    returnvec::Bool = true,
                    v0::Vector{T} = rand(T, nrow(A)) - convert(T, 0.5))
    ## Size of matrix
    n = nrow(A)
    ## Check arguments
    if nev < 1 || nev > n - 1
        error("nev must satisfy 1 <= nev <= n - 1, n is the size of matrix")
    end
    if ncv <= nev || ncv > n
        error("ncv must satisfy nev < ncv <= n, n is the size of matrix")
    end

    ## Number of matrix operations called
    nmatop::Int = 0
    ## Number of restarting iteration
    niter::Int = maxiter
    ## Number of converged eigenvalues
    nconv::Int = 0

    ## Matrices and vectors in the Arnoldi factorization
    V::Matrix{T} = zeros(T, n, ncv)
    H::Matrix{T} = zeros(T, ncv, ncv)
    f::Vector{T} = zeros(T, n)
    ritzval::Vector{T} = zeros(T, ncv)
    ritzvec::Matrix{T} = zeros(T, ncv, nev)
    ritzest::Vector{T} = zeros(T, ncv)
    ritzconv::Vector{Bool} = zeros(Bool, nev)

    ## Precision parameter used to test convergence
    prec::T = eps(T)^(convert(T, 2.0) / 3)

    ## Initialize vectors
    v0norm = norm(v0)
    if v0norm < prec
        error("initial residual vector cannot be zero")
    end
    v0[:] /= v0norm
    w::Vector{T} = performop(A, v0)
    nmatop += 1
    H[1, 1] = dot(v0, w)
    f[:] = w - v0 * H[1, 1]
    V[:, 1] = v0

    ## First Arnoldi factorization
    arnoldifac!(1, ncv, V, H, f, A, prec)
    nmatop += (ncv - 1)
    ritzpairs!(which, H, ritzval, ritzvec, ritzest)
    ## Restarting
    thresh::Vector{T} = zeros(T, nev)
    resid::Vector{T} = zeros(T, nev)
    for i = 1:maxiter
        ## Convergence test
        thresh[:] = tol * max(abs(ritzval[1:nev]), prec)
        resid[:] = abs(ritzest[1:nev]) * norm(f)
        ritzconv[:] = (resid .< thresh)
        nconv = sum(ritzconv)

        if nconv >= nev
            niter = i
            break
        end

        nevadj::Int = nevadjusted(nev, ncv, nconv, ritzest, prec)

        applyshifts!(nevadj, V, H, f, ritzval)
        arnoldifac!(nevadj, ncv, V, H, f, A, prec)
        nmatop += (ncv - nevadj)
        ritzpairs!(which, H, ritzval, ritzvec, ritzest)
    end

    ## Final sorting of Ritz values
    ix = sortperm(ritzval[1:nev], by = abs, rev = true)
    converged = ix[ritzconv[ix]]
    eigval = ritzval[converged]
    if returnvec
        eigvec = V * ritzvec[:, converged]
    else
        eigvec = nothing
    end

    return eigval, eigvec, nconv, niter, nmatop
end
