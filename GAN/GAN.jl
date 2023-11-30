getcode(gan::GAN) = Float64.(gan.pz(size(((Flux.params(gan.g)).order)[1],2)))
getcode(gan::GAN, n::Int) = Float64.(gan.pz(size(((Flux.params(gan.g)).order)[1],2), n))

generate(gan::GAN, data) = binarize(activation_gen(gan.g(getcode(gan)), gan.feature_type), gan.feature_type)
generate(gan::GAN, n::Int) = binarize(activation_gen(gan.g(randn(gan.gen_input_dim,n)), gan.feature_type), gan.feature_type)

Dloss(gan::GAN, X, Z) = (- Float64(0.5)*(Statistics.mean(log.(gan.d(X) .+ eps(Float64))) + Statistics.mean(log.(1 .- gan.d(activation_gen(gan.g(Z), gan.feature_type)) .+ eps(Float64)))))
Gloss(gan::GAN, Z) = (- Statistics.mean(log.(gan.d(activation_gen(gan.g(Z), gan.feature_type)) .+ eps(Float64))))


function random_batch_index(x::AbstractArray, batch_size=1; dims=1)
    n = size(x, dims)
    Iterators.partition(shuffle(1:n), batch_size)
end


function train_GAN!(gan::GAN, x_st, original_data)
    loss_array_gan = [Float64[],Float64[]]
    gen_error = []
    for epoch = 1:gan.epochs
        loss_epoch = [0, 0, 0]
        if epoch % args.verbose_freq == 0
            print("Epoch $epoch: ") 
        end

        training_data = random_batch_index(x_st, gan.batch_size)

        for I in training_data
            

            # sample data and generate codes
            x_µ = Statistics.mean(x_st[I,:], dims=1)

            z = getcode(gan, length(I))

            # discriminator training
            for i = 1:5
                psD = Flux.params(gan.d)

                gs = gradient(psD) do

                    training_loss = Dloss(gan, (x_st[I,:])', z)
                    
                    return training_loss
                end

                opt = ADAM(gan.η)

                Flux.Optimise.update!(opt, psD, gs)

            end
            Dl = Dloss(gan, (x_st[I,:])', z)

            # generator training
            

            for i = 1:1
                psG = Flux.params(gan.g)
                gs = gradient(psG) do
                    training_loss = Gloss(gan, z)
                    # Insert what ever code you want here that needs Training loss, e.g. logging
                    
                    return training_loss
                end
               # insert what ever code you want here that needs gradient
               # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
                opt = ADAM(gan.η)
                Flux.Optimise.update!(opt, psG, gs)
            end
            Gl = Gloss(gan, z)

            loss_epoch += [Dl, Gl, length(I)]
        end

        if epoch % args.verbose_freq == 0
        
            print("Discriminator loss = ", loss_epoch[1])

            println("\t Generator loss = ", loss_epoch[2])

        end

        loss_epoch ./= loss_epoch[3]
        push!(loss_array_gan[1], loss_epoch[1])
        push!(loss_array_gan[2], loss_epoch[2])
        push!(gen_error, eval_gen(x_st, gan))

    end

    save_gan_results(original_data, gan, preprocess_ps, args, loss_array_gan, gen_error)

    return gan, loss_array_gan, gen_error
end

function GAN_output(input, gan::GAN, args::Args, preprocess_ps::preprocess_params, loss_array_gan:: Array, gen_error::Array)

    Random.seed!(11)

    n = size(input, 1)

    output = generate(gan, n)

    if args.scaling
        output = reverse_standardize(preprocess_ps.μ, preprocess_ps.σ, output)
    end
    
    if args.pre_transformation

        output = output'

        for j = 1:size(output, 1)
            output[j,:] = reverse_power(preprocess_ps, output[j, :], dataTypeArray)
        end

        output = reverse_BC(preprocess_ps, output, dataTypeArray)

        return output
    else
        
        return output'

    end
    

end


function activation_gen(x, dataTypeArray)
    temp = fill(0.0, size(x)[1], size(x)[2])
    output = Zygote.Buffer(temp)
    
    for j = 1:size(x)[1]
        if dataTypeArray[j]
            output[j,:] = x[j,:]
        else
            output[j,:] = σ.(x[j,:])
        end
    end

    copy(output)
end


function binarize(x, dataTypeArray)
    
    output = copy(x)

    for j = 1:size(output, 1)
        if .!dataTypeArray[j]
            output[j, :] = Int.(output[j, :] .> 0.5)
        end
    end

    return output
end



function eval_gen(data, gan::GAN)
    sum((Statistics.mean(data, dims=1) .- Statistics.mean(generate(gan, size(data, 1)) .- rand(size(data, 2), size(data, 1)), dims=2)).^2)
end
