# Spectra

A [Julia](http://julialang.org/) implementation of the
[Spectra](https://github.com/yixuan/spectra) library.

```jl
using Spectra

srand(123)
n = 1000
M = randn(n, n)
M = M + M'
A = DenseMatProd(M)
symeigs(A; nev = 10, ncv = 30, tol = 1e-10)
```
