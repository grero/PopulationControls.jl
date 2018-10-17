# Introduction
This is a julia implementation of the algorithm developed in Elsayed, G. F., & Cunningham, J. P. (2017). Structure in neural population recordings: an expected byproduct of simpler phenomena? Nature Neuroscience, 20(9), 1310–1318. http://doi.org/10.1038/nn.4617

In essence, the code fits a maximum entropy distribution over tensors that preserves marginally covariance matrices along the different tensor dimensions. Sampling from the resulting distribution provides a null distribution for testing whether so called population-level phenomena can be explained by the marginal covariances, or whether highle level features play a part.

# Example

```julia
    #construct marignal covariances along 3 dimensions
    RNG = MersenneTwister(1234)
    n = 3
    Σ = Vector{Matrix{Float64}}(undef, n)
    for i in 1:n
        q,r = qr(rand(RNG, 3,3))
        Σ[i] = r'*r
    end
    #Initial guess for eigenvalues of the joint covariance matrix
    λ0 = rand(RNG, sum(x->size(x,1), Σ))
    #Find the eigenvectors Q and the eigenvalues λ of the joint covariance matrix
    Q,λ = PopulationControls.tensor_covariance(Σ;λ0=λ0)
    #sample 100 trials from the tensor distribution
    Z = PopulationControls.sample(Q, λ, 100)
    #If desired, we can explicitly form the covariance matrix using the  eigenvectors U and the eigvenvalues Λ by taking the kronecker product of Q and the kronecker sum of Λ. Note that this is not necessary, and indeed not in most cases not desired since the resulting matrix can be huge. Instead, we sample from the compnents as above.
    U,Λ = PopulationControls.compose(Q, λ)
```
