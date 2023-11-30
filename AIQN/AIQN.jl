
using Flux
using Flux: DataLoader
using HypothesisTests:ApproximateTwoSampleKSTest, pvalue

function get_quantile_data(x, batch_size)
    """
    This function prepares training data in a DataLoader format
    """

    if size(x, 2) == 1
        return DataLoader((data =  rand(size(x)[1]), label = x) , batchsize = batch_size, shuffle = true)
    else
        @show size(x[:, 1:end-1])
        @show size( rand(size(x, 1)))
        @show size( hcat(x[:, 1:end-1], rand(size(x, 1))))
        return DataLoader((data = hcat(x[:, 1:end-1], rand(size(x, 1)))', label = x[:, end]) , batchsize = batch_size, shuffle = true)
    end
    
end

function find_mean(tau)
    """
        This function returns the index of the input array which is closer to median
    """
    min = 1
    index = 0

    for i = 1:length(tau)
        dif = abs(tau[i] - 0.5)
        if dif < min
            min = dif
            index = i
        end
    end

    return index
end


function find1573(tau) 
    """
        This function returns the index of the input array which is closer to quantile = 0.1573 or quantile = 0.8427.
        This is needed to find the standard deviation using the subtraction of that and quantile = 0.5.
    """
    min = 1
    index = 0
    for i = 1:length(tau)
        dif = abs(tau[i] - 0.1573)
        dif2 = abs(tau[i] - 0.8427)
        if (dif < min)||(dif2 < min)
            min = dif
            index = i
        end
    end
    return index
end


function quantile_loss(x, y, qfind_network)
    
    kappa = 0.1

    

    u = y - qfind_network(x)



    tau = x[end]
 
    condval = abs(tau - (u[1] <= 0.0 ))

    return ((abs(u[1]) <= kappa) * (condval / 2kappa) * u[1]^2)  + ((abs(u[1]) > kappa) * (condval * (abs(u[1]) - 0.5 * kappa))) #+ 0.01 * sum(x->sum(x.^2), Flux.params(qfind_network))
    #FIXME @.
end

function find_score!(first, second, zmat)

    """
        This function returns a score which the higher is representative for the better combination of zmat (which dimension use first for the quantile network)
        The score comes from the combined p-value from Kolmigrov Smirnov test.
        https://daithiocrualaoich.github.io/kolmogorov_smirnov/
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
    """

    Random.seed!(42)
    
    n, z_dimension = size(zmat)

    zmat_copy = hcat(zmat[:, first], zmat[:, second])

    generated_median = nothing
    tau_first = nothing
    tau_second = nothing
    generated_first_dimension_z = nothing
    generated_second_dimension_z = nothing

    for i = 1:2 # i = 1 is for z₁ & i = 2 is for z₂|z₁

        qfind_network = Chain(Dense(i, 3, tanh), Dense(3, 1))

        if i == 1 #FIXME make sure that this is being used 
            data =  get_quantile_data(zmat_copy[repeat(1:n, 10)[randperm(n * 10)], 1], 1)   #FIXME batchsize
        else
            zindex = repeat(1:n, 10)[randperm(n * 10)]
            data = get_quantile_data(zmat_copy[zindex,:], 1) #FIXME add batchsize for quantile
        end

        opt = ADAM(0.001)

        ps = Flux.params(qfind_network)

        training_loss = nothing

        for epoch = 1:10 
            for (x,y) in data
                gs = gradient(ps) do

                    training_loss =  quantile_loss(x, y, qfind_network)
                    
                    return training_loss
                end

                
                update!(opt, ps, gs)
            end
            @info epoch
            @show training_loss
        end

        # Get unconditional z₁ and conditional z₂|z₁
        
        
        if i == 1
            tau_first = rand(n) .* 0.9 .+ 0.05
            generated_median = qfind_network([0.5])
            
            generated_first_dimension_z =  map(val -> qfind_network([val])[1], tau_first) 
        else
            tau_second = rand(n) .* 0.9 .+ 0.05
            # @show generated_median[1:i-1]
            # @show size(hcat(generated_median[1:i-1], tau_second)')
            generated_second_dimension_z = [qfind_network([generated_median[1], tau_second[k] ])[1] for k=1:n]#qfind_network(hcat(generated_median[1:i-1], tau_second)')  #FIXME vcat or hcat
        end
    end


    # normal_z₁ is a dataset with normal distribution sampled from N(μ, σ) with the same μ and σ as generated_first_dimension_z
    normal_z₁ = quantile.(Normal(generated_first_dimension_z[find_mean(tau_first)][1], abs.(generated_first_dimension_z[find_mean(tau_first)] - generated_first_dimension_z[find1573(tau_first)])[1]), tau_first)
    
    # normal_conditional is a dataset with normal distribution sampled from N(μ, σ) with the same μ and σ as generated_second_dimension_z
    normal_conditional = quantile.(Normal(generated_second_dimension_z[find_mean(tau_second)][1], abs.(generated_second_dimension_z[find_mean(tau_second)] - generated_second_dimension_z[find1573(tau_second)])[1]), tau_second)


    p_value_unconditional = ApproximateTwoSampleKSTest(normal_z₁, generated_first_dimension_z)
    p_value_conditional = ApproximateTwoSampleKSTest(normal_conditional, generated_second_dimension_z)

    #FIXME print this part in a file 
    println(p_value_conditional)
    # println(pvalue(p_value_conditional))
    println(p_value_unconditional)
    # println(pvalue(p_value_unconditional))


    print("combined p-value for P(z₁) and p-value for P(z₂|z₁): ")
    comb_z = - 2 * sum(log(pvalue(p_value_conditional)) + log(pvalue(p_value_unconditional)))
    println(comb_z)

    return comb_z
end



function find_order(zmat)

    shape = size(zmat)
    z_dimension = shape[2]

    # K_S is a metrix in each row we have score, first element and second element
    scores_number = Int(((z_dimension ^ 2) - z_dimension)/2)
    K_S = fill(0.0, scores_number, 3)

    # Number of scores which has been added
    cnt = 1
    for i = 1:z_dimension
        for j = i+1:z_dimension
            
            # If the p-value is small, conclude that the two groups were sampled from populations with different distributions. (Better to be big)
            score1 =  find_score!(i, j, zmat)
            
            score2 =  find_score!(j, i, zmat)
            
            if (score1 > score2)
                K_S[cnt, 1] = score1
                K_S[cnt, 2] = i
                K_S[cnt, 3] = j
            else
                K_S[cnt, 1] = score2
                K_S[cnt, 2] = j
                K_S[cnt, 3] = i
            end
            cnt +=1
        end
    end

    order = []
    K_S[sortperm(K_S[:, 1]), :]

    for i = 1:scores_number
        println("count")
        println(K_S[scores_number - i + 1, 2])
        println(K_S[scores_number - i + 1, 3])

        if K_S[scores_number - i + 1, 2] in order || K_S[scores_number-i+1, 3] in order
            continue
        else
            append!(order, Int(K_S[scores_number - i + 1, 2]))
            append!(order, Int(K_S[scores_number - i + 1, 3]))
        end
    end

    @show z_dimension
    if (length(order) == z_dimension)

        println("order: $(order)")

        return order
    else
        for i = 1:z_dimension
            if i in order
                continue
            else
                append!(order, i)
            end
        end

        println("order: $(order)")

        return order
    end

end


function qfind(zmat, order)
    
    n, z_dimension = size(zmat)

    zmat_copy = fill(0.0, n, 1)

    # initialize a zmat generated by quantile network
    generated_z = fill(0.0, n, z_dimension)

    for i = 1:z_dimension
        Random.seed!(11)
        qfind_network = Chain(Dense(i, 3, tanh), Dense(3, 1))
        

        if i ==1
            zmat_copy = zmat[:, order[i]]
            data = get_quantile_data(zmat_copy[repeat(1:n, 10)[randperm(n * 10)], 1], 1) #FIXME batchsize
        else
            zmat_copy = hcat(zmat_copy, zmat[:, order[i]])

            @show size(zmat_copy)
            zindex = repeat(1:n, 10)[randperm(n * 10)]
            data = get_quantile_data(zmat_copy[zindex,:], 1) #FIXME batchsize
        end

        opt = ADAM(0.001)

        ps = Flux.params(qfind_network)

        for epoch = 1:100 
            for (x,y) in data
                gs = gradient(ps) do
                    training_loss =  quantile_loss(x, y, qfind_network)
            
                    return training_loss
                end
                update!(opt, ps, gs)
            end
        
        end

        
        if i ==1
            generated_z[:, 1] = qfind_network((rand(n) .* 0.9 .+ 0.05)')
            
        else
            tau = rand(n) .* 0.9 .+ 0.05
            generated_z[:, i] = qfind_network(hcat(generated_z[:, 1:i-1], tau)')  
            
        end
        
    end

    return generated_z
end
