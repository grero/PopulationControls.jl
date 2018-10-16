module PopulationControls
using LinearAlgebra
using Random
using Optim
using Base.Iterators

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
    kron(A,IB) .+ kron(IA, B)
end

"""
Efficiently computes the kronecker sum of diagonal matrices
"""
function diagkronsum(Λ::Vector{Vector{T}}) where T <: Real
    N = length(Λ)
    kk = fill(0.0, 1,1)
    for λ in Λ 
        kk = vec((kk*fill(1.0, length(λ),1)' .+ fill(1.0, length(kk),1)*λ')')
    end 
    kk
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
    λ,fmin = find_λ2(S,nn,λ0)
    _λ = Vector{Matrix{T}}(undef, length(nn))
    vv = 0
    for i in 1:length(nn)
        _λ[i] = Matrix(Diagonal(λ[vv+1:vv+nn[i]]))
        vv += nn[i]
    end
    Q,_λ
end

function compose(Q,_λ)
    #TODO: Find the Kroencker sum
    U = reduce(kron, Q)
    Λ = reduce(kronsum, _λ)
    U,Λ
end

function get_s(λ::Vector{T},nn::Vector{Int64}) where T <: Real
    S = similar(λ)
    vv = 0
    for (i,n) in enumerate(nn)
        for d in 1:n
            S[vv+d] = get_s2(λ,nn, i,d)
        end
        vv += n
    end
    S
end

function get_s2(λ::Vector{T},nn::Vector{Int64}) where T <: Real
    S = similar(λ)
    vv = 0

    n_iter = Iterators.product([1:n for n in nn]) 
    for pp in n_iter
       for (_k,_d) in enumerate(pp) 
           S[sum(nn[1:_k-1])+_d] = get_s2(λ,nn, _k,_d)
        end
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

function get_s2(λ::Vector{T},nn::Vector{Int64}, k::Int64,d::Int64) where T <: Real
    n_iter = Iterators.product([1:n for n in nn]...) 
    vv = 0.0
    for pp in n_iter
        v = λ[sum(nn[1:k-1]) + d]
       for (_k,_d) in enumerate(pp) 
           if (_k != k)
                v += λ[sum(nn[1:_k-1]) + _d]
            end
       end
       vv += 1/v
    end
    0.5*vv
end

function costfunc(λ,S,nn)
    q = 0.0
    vv = 0
    for k in 1:length(nn)
        qq = 0.0
        for d in 1:nn[k] 
            qq = S[vv+d] - get_s2(λ,nn,k,d)
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

function sample(Q::Vector{Matrix{T}}, Λ::Vector{Matrix{T}},n=1) where T <: Real
    d = 1.0./sqrt.(diagkronsum(map(diag, Λ)))
    N = length(d)
    X = randn(N,n)
    Z = zeros(N,n)
    for i in 1:n
        Y = X[:,i].*d 
        Z[:,i] = kron_mvprod(Q, Y)
    end
    Z
end

end # module
