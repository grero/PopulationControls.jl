module PopulationControls
using LinearAlgebra
using Random
using Optim

"""
Compute the product between the matrix ⊗ᵢA and the vector b
"""
function kron_mvprod(A::Vector{Matrix{T}}, b::Vector{T}) where T <: Real
    D = length(A)
    N = length(b)
    x = similar(b)
    x .= b
    for d in D:-1:1
        Ad = A[d]
        Gd = size(Ad,1)
        X = reshape(x, Gd, div(N,Gd))
        Z = (Ad*X)'
        x .= vec(Z)
    end
    x
end

function kronsum(A::Matrix{T},B::Matrix{T}) where T <: Real
    IB = Matrix(1.0I,size(B)...)
    IA = Matrix(1.0I,size(A)...)
    kron(A,IB) + kron(IA, B)
end

"""
Returns the covariance matrix of a maximum entropy distribution given the marginal covariance matrices in Σ
"""
function tensor_covariance(Σ::Vector{Matrix{T}};λ0=rand(sum(x->size(x,1), Σ))) where T <: Real
    Q = Vector{Matrix{T}}(undef, length(Σ))
    nn = map(x->size(x,1), Σ)
    S = fill(0.0, sum(nn))
    vv = 0
    for (i,Σi) in enumerate(Σ)
        u,s,v = svd(Σi)
        S[vv+1:vv+nn[i]] = s
        Q[i] = v
        vv += nn[i] 
    end
    U = reduce(kron, Q)
    λ,fmin = find_λ2(S,nn,λ0)
    _λ = Vector{Matrix{T}}(undef, length(nn))
    vv = 0
    for i in 1:length(nn)
        _λ[i] = Matrix(Diagonal(λ[vv+1:vv+nn[i]]))
        vv += nn[i]
    end
    #TODO: Find the Kroencker sum
    Λ = reduce(kronsum, _λ)
    U,Λ
end

function get_s(λ::Vector{T},nn::Vector{Int64}) where T <: Real
    S = similar(λ)
    vv = 0
    for (i,n) in enumerate(nn)
        for d in 1:n
            S[vv+d] = get_s(λ,nn, i,d)
        end
        vv += n
    end
    S
end

function get_s(λ::Vector{T},nn::Vector{Int64}, k::Int64,d::Int64) where T <: Real
    q = 0.0
    vv = 0
    for (i,n) in enumerate(nn)
        v = 0.0
        if i == k
            v += λ[vv+d]
        else
            for j in 1:n
                v += λ[vv+j]
            end
        end
        vv += n
        q += 1/v
    end
    0.5*q
end

function costfunc(λ,S,nn)
    q = 0.0
    vv = 0
    for k in 1:length(nn)
        qq = 0.0
        for d in 1:nn[k] 
            qq = S[vv+d] - get_s(λ,nn,k,d)
            q += qq*qq
        end
        vv += nn[k]
    end
    q
end

function logcostfunc(λ,S,nn)
    q = 0.0
    vv = 0
    for k in 1:length(nn)
        qq = 0.0
        for d in 1:nn[k] 
            qq = log(S[vv+d]) - log(get_s(exp.(λ),nn,k,d))
            q += qq*qq
        end
        vv += nn[k]
    end
    q
end

function find_λ(S::Vector{T},nn::Vector{Int64}) where T <: Real
     q = optimize(x->PopulationControls.costfunc(x,S,nn), rand(length(S)),LBFGS();autodiff=:forward)
     q.minimizer, q.minimum
end

function find_λ2(S::Vector{T},nn::Vector{Int64},λ0=rand(length(S))) where T <: Real
     q = optimize(x->PopulationControls.logcostfunc(x,S,nn), λ0,LBFGS();autodiff=:forward)
     exp.(q.minimizer), q.minimum
 end

end # module
