using PopulationControls
using Test
using LinearAlgebra
using Random

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
    U,Λ = PopulationControls.tensor_covariance(Σ;λ0=λ0)
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

