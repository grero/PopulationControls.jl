using PopulationControls
using Test
using LinearAlgebra
using Random

@testset "Kroenecker matrix vector product" begin
    RNG = MersenneTwister(1234)
    A = [rand(RNG,n,n) for n in [3,4,5]]
    N = prod([3,4,5])
    b = rand(RNG, N) 
    Ap = reduce(kron, A)
    v1 = Ap*b
    v2 = PopulationControls.kron_mvprod(A,b)
    @test v1 ≈ v2
end

@testset "Kronecker sum of diagonal matrices" begin
    d1 = [1,2,3]
    d2 = [5,6,7,8]
    q1 = PopulationControls.diagkronsum([d1,d2])
    #compare to straight up sum
    q2 = diag(PopulationControls.kronsum(Matrix(Diagonal(d1)), Matrix(Diagonal(d2))))
    @test q1 ≈ q2
end

@testset "Basic" begin
    RNG = MersenneTwister(1234)
    n = 3
    Σ = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        q,r = qr(rand(RNG, 3,3))
        Σ[i] = r'*r
    end
    SS = kron(Σ[1], Σ[2])
    SS2 = reduce(kron, Σ)
    λ0 = rand(RNG, sum(x->size(x,1), Σ))
    Q,_λ = PopulationControls.tensor_covariance(Σ;λ0=λ0)
    U,Λ = PopulationControls.compose(Q, _λ)
    hu =  hash(U)
    @test hu == 0xfb803b574fe2e43e
    hl =  hash(Λ)
    @test hl == 0x28cebb17058950fa
    ΣT = 0.5*U*inv(Λ)*U'
    ht = hash(ΣT)
    @test ht == 0xa1bbaf3af8138f4d
end

@testset "Test s" begin
    λ = cat([rand(n) for n in [3,4,5]]...,dims=1)
    nn = [3,4,5]
    s = PopulationControls.get_s(λ,nn,1,1)
    @show s
end
