using Spectra
using Base.Test

# write your own tests here
srand(123)
n = 100
M = randn(n, n)
M = M + M'
A = DenseMatProd(M)
v0 = rand(n) - 0.5

r1 =    eigs(M; nev = 5, ncv = 12, tol = 1e-10, v0 = v0)
r2 = symeigs(A; nev = 5, ncv = 12, tol = 1e-10, v0 = v0)

@test maximum(abs(r1[1] - r2[1])) < 1e-10

println("Error: ", maximum(abs(r1[1] - r2[1])))

t1 = (@elapsed for i = 1:10    eigs(M; nev = 5, ncv = 12, tol = 1e-10, v0 = v0) end)
t2 = (@elapsed for i = 1:10 symeigs(A; nev = 5, ncv = 12, tol = 1e-10, v0 = v0) end)

println("Time of eigs(): $(t1)")
println("Time of symeigs(): $(t2)")
