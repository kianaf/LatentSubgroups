using Flux
using Flux.Data: DataLoader
using Parameters: @with_kw
using Distributions: Bernoulli, Normal
using Revise 
using TensorBoardLogger: TBLogger, tb_overwrite

includet("transformations.jl")
includet("../AIQN/AIQN.jl")
includet("structs.jl")
includet("report.jl")
includet("GLM.jl")

#FIXME in general you are inconsistent about the functions with ! on the end of the name. Do you need return or not?!


"""
    getparams(m)

parameters
----------
m: the variational autoencoder struct 

Notes
-----
Collects the parameters, i.e., the weights and biases of all neural network layers of the VAE model `m`, using the `Flux.params()` function. 
"""

getparams(m::vanilla_vae) = Flux.params(m.encoder, m.encodedμ, m.encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.decodedπ)
# getparams(m::multimodal_vae) = Flux.params(m.binary_encoder, m.continuous_encoder, m.binary_encodedμ, m.continuous_encodedμ, m.binary_encodedlogσ, m.continuous_encodedlogσ, m.μ_fusion, m.σ_fusion, m.decoder, m.decodedμ, m.decodedlogσ, m.decodedπ)
getparams(m::multimodal_vae) = Flux.params(m.binary_encoder, m.continuous_encoder, m.binary_encodedμ, m.continuous_encodedμ, m.binary_encodedlogσ, m.continuous_encodedlogσ, m.decoder, m.decodedμ, m.decodedlogσ, m.decodedπ)

function init_multimodal_vae(args::Args, feature_type)

    num_continuous_vars = count(feature_type.==true)

    num_binary_vars = count(feature_type.==false)

    vae_dir = string(args.current_run_path, "/vae")

    if !isdir(vae_dir)
        # println("kiana")
        # mkdir(vae_dir)
        tblogger_obj = TBLogger(vae_dir)


        save_struct(args, string(vae_dir, "/args.bson"))

        open(string(vae_dir,"/args.txt"), "a") do file
            print(file, args)
        end
    else
        parent_dir = string(pwd(), "/", args.current_run_path)
        dir_list = readdir(parent_dir)

        vae_list = []

        if length(dir_list) > 0
            for i = 1:length(dir_list)
                if isdir(string(parent_dir, "/",  dir_list[i]))   && contains(dir_list[i], "vae_")                     # checking, if it's directory
                    append!(vae_list, parse(Int, string(dir_list[i])[5:end]))      # print the name of a directory
                end
            end

            if length(vae_list) !=0
                vae_number = (sort(vae_list))[end] + 1
            else
                vae_number = 1
            end
        end

        vae_dir = string(vae_dir, "_$(vae_number)")
        # mkdir(vae_dir)
        tblogger_obj = TBLogger(vae_dir)

        save_struct(args, string(vae_dir, "/args.bson"))

        open(string(vae_dir,"/args.txt"), "a") do file
            print(file, args)
        end
    end


    # seed
    Random.seed!(args.seed)

    # VAE encoder
    # binary_encoder, continuous_encoder, binary_encodedμ, continuous_encodedμ, binary_encodedlogσ, continuous_encodedlogσ, μ_fusion, σ_fusion = Dense(num_binary_vars, num_binary_vars, tanh), Dense(num_continuous_vars, num_continuous_vars, tanh), Dense(num_binary_vars, args.latent_dim), Dense(num_continuous_vars,  args.latent_dim), Dense(num_binary_vars, args.latent_dim), Dense(num_continuous_vars,  args.latent_dim), Dense(2*args.latent_dim, args.latent_dim), Dense(2*args.latent_dim,  args.latent_dim)
    binary_encoder, continuous_encoder, binary_encodedμ, continuous_encodedμ, binary_encodedlogσ, continuous_encodedlogσ = Dense(num_binary_vars, num_binary_vars, tanh), Dense(num_continuous_vars, num_continuous_vars, tanh), Dense(num_binary_vars, args.latent_dim), Dense(num_continuous_vars,  args.latent_dim), Dense(num_binary_vars, args.latent_dim), Dense(num_continuous_vars,  args.latent_dim)

    # VAE decoder
    if args.synthetic_data
        decoder, decodedμ, decodedlogσ, decodedπ = Dense(2 * args.latent_dim, args.hidden_dim, tanh),  Dense(args.hidden_dim, num_continuous_vars, σ), Dense(args.hidden_dim, num_continuous_vars), Dense(args.hidden_dim, num_binary_vars, σ)
        
    else
        decoder, decodedμ, decodedlogσ, decodedπ = Dense(args.latent_dim, args.hidden_dim, tanh),  Dense(args.hidden_dim, num_continuous_vars), Dense(args.hidden_dim, num_continuous_vars), Dense(args.hidden_dim, num_binary_vars, σ)
    
    end


    # model = multimodal_vae(binary_encoder, continuous_encoder, binary_encodedμ, continuous_encodedμ, binary_encodedlogσ, continuous_encodedlogσ, μ_fusion, σ_fusion, decoder, decodedμ,  decodedlogσ, decodedπ, feature_type, num_continuous_vars, num_binary_vars, args.AIQN, args.β)
    model = multimodal_vae(binary_encoder, continuous_encoder, binary_encodedμ, continuous_encodedμ, binary_encodedlogσ, continuous_encodedlogσ, decoder, decodedμ,  decodedlogσ, decodedπ, feature_type, num_continuous_vars, num_binary_vars, args.AIQN, args.β, tblogger_obj)

    return model
end

function init_vanilla_vae(args::Args, feature_type)

    num_continuous_vars = count(feature_type.==true)

    num_binary_vars = count(feature_type.==false)

    vae_dir = string(args.current_run_path, "/vae")

    if !isdir(vae_dir)

        tblogger_obj = TBLogger(vae_dir)

        save_struct(args, string(vae_dir, "/args.bson"))

        open(string(vae_dir,"/args.txt"), "a") do file
            print(file, args)
        end
    else
        parent_dir = string(pwd(), "/", args.current_run_path)
        dir_list = readdir(parent_dir)

        vae_list = []

        if length(dir_list) > 0
            for i = 1:length(dir_list)
                if isdir(string(parent_dir, "/",  dir_list[i]))   && (dir_list[i]!="vae") && (dir_list[i]!="pre_transformation")                    # checking, if it's directory
                    append!(vae_list, parse(Int, string(dir_list[i])[5:end]))      # print the name of a directory
                end
            end

            if length(vae_list) !=0
                vae_number = (sort(vae_list))[end] + 1
            else
                vae_number = 1
            end
        end

        vae_dir = string(vae_dir, "_$(vae_number)")
        
        tblogger_obj = TBLogger(vae_dir)

        save_struct(args, string(vae_dir, "/args.bson"))

        open(string(vae_dir,"/args.txt"), "a") do file
            print(file, args)
        end

    end

    

    # seed
    Random.seed!(args.seed)

    # VAE encoder
    encoder, encodedμ, encodedlogσ = Dense(args.input_dim, args.hidden_dim, tanh), Dense(args.hidden_dim, args.latent_dim), Dense(args.hidden_dim, args.latent_dim)
    
    # VAE decoder
    decoder, decodedμ, decodedlogσ, decodedπ = Dense(args.latent_dim, args.hidden_dim, tanh),  Dense(args.hidden_dim, num_continuous_vars), Dense(args.hidden_dim, num_continuous_vars), Dense(args.hidden_dim, num_binary_vars, σ)

    model = vanilla_vae(encoder, encodedμ, encodedlogσ, decoder, decodedμ,  decodedlogσ, decodedπ, feature_type, num_continuous_vars, num_binary_vars, args.AIQN, args.β, tblogger_obj)
    
    return model
end

# encoding function
encode(x, m::vanilla_vae) = (hidden = m.encoder(x); (m.encodedμ(hidden), m.encodedlogσ(hidden)))
decode(z, m::vanilla_vae) = (hidden = m.decoder(z); (m.decodedμ(hidden), m.decodedlogσ(hidden)))



function encode(x, m::multimodal_vae) 
    (binary_μ, binary_logσ) = (binary_hidden = (m.binary_encoder(x[.!m.feature_type, :])); (m.binary_encodedμ(binary_hidden), m.binary_encodedlogσ(binary_hidden)))
    (continuous_μ, continuous_logσ) = (continuous_hidden = (m.continuous_encoder(x[m.feature_type, :])); (m.continuous_encodedμ(continuous_hidden), m.continuous_encodedlogσ(continuous_hidden)))

    return (binary_μ, binary_logσ), (continuous_μ, continuous_logσ) 

end

# FIXME reparameterization trick
latentz(μ, logσ) = μ .+ exp.(logσ) .* randn(Float32, size(logσ))


# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ.^2 .- 1 .- (2 .* logσ))

logp_x_z(x, z, m) = (hidden = m.decoder(z); sum(logpdf.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden)) .+ eps(Float32)), (x[m.feature_type, :]))))#/sum(m.feature_type)

# The same as Flux.binarycrossentropy
logpdf_bernouli(b::Bernoulli, y) = (y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32)))

logp_x_z_bernouli(x, z, m) = sum(logpdf_bernouli.(Bernoulli.(m.decodedπ(m.decoder(z))), (x[.!m.feature_type, :])))#/sum(.!m.feature_type)


function loss(X, m::multimodal_vae, preprocess_ps)



    (binary_μ, binary_logσ), (continuous_μ, continuous_logσ)  = encode(X, m)


    # latent_μ =  m.μ_fusion(vcat(binary_μ, continuous_μ))
    # latent_σ = m.σ_fusion(vcat(binary_logσ, continuous_logσ))
    # latvals = latentz(latent_μ, latent_σ)

    if args.synthetic_data 
        latvals = vcat(latentz(binary_μ, binary_logσ), latentz(continuous_μ, continuous_logσ))
    else
        latvals = latentz((binary_μ .+ continuous_μ) ./ 2, (binary_logσ .+ continuous_logσ)./2)
    end

    latz = latvals

    len = size(X)[end]

    latent_dim = size(latvals,1)

    if m.num_binary_vars ==0

        log_likelihood = logp_x_z(X, latz, m)
        kld_loss = kl_q_p(continuous_μ, continuous_logσ) #/latent_dim

        lower_bound = (log_likelihood - m.β * kld_loss)/ len  

        loss_value = -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder, m.decodedμ, m.decodedlogσ)) # , m.decodedπ

    elseif m.num_continuous_vars ==0
        log_likelihood = logp_x_z_bernouli(X, latz, m)
        kld_loss = kl_q_p(binary_μ, binary_logσ)#/latent_dim


        lower_bound =  (log_likelihood - m.β * kld_loss)/ len  


        loss_value = -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder,  m.decodedπ))
    
    else

        # continuous_weight = (m.num_continuous_vars + m.num_binary_vars)/ m.num_continuous_vars
        # binary_weight = (m.num_continuous_vars + m.num_binary_vars) / m.num_binary_vars


        continuous_weight = 1
        binary_weight = 1

        log_likelihood = continuous_weight * logp_x_z(X, latz, m) + binary_weight * logp_x_z_bernouli(X, latz, m)

        if args.synthetic_data
            kld_loss = kl_q_p(hcat(binary_μ, continuous_μ), hcat(binary_logσ, continuous_logσ)) #/(2*latent_dim)
        else
            kld_loss = kl_q_p((binary_μ .+ continuous_μ)./2, (binary_logσ .+ continuous_logσ)./2)
        end
        
        lower_bound =  (log_likelihood - m.β * kld_loss)/ len  #
        
        loss_value = -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder, m.decodedμ, m.decodedlogσ,  m.decodedπ)) 
        
    end

    reconstruction_loss = - log_likelihood

    return loss_value, reconstruction_loss/len, kld_loss/len
end



function loss(X, m::vanilla_vae, preprocess_ps)

    (μ, logσ) = encode(X, m)
    
    latvals = latentz(μ, logσ)

    latent_dim = size(latvals,1)

    latz = latvals

    len = size(X)[end]

    

    if m.num_binary_vars ==0

        log_likelihood = logp_x_z(X, latz, m) 
        kld_loss = kl_q_p(μ, logσ)#/latent_dim

        lower_bound =  (log_likelihood - m.β * kld_loss)/ len    

        loss_value = -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder, m.decodedμ, m.decodedlogσ))

    elseif m.num_continuous_vars ==0

        log_likelihood = logp_x_z_bernouli(X, latz, m)
        kld_loss = kl_q_p(μ, logσ)#/latent_dim

        lower_bound =  (log_likelihood - m.β * kld_loss)/ len    

        loss_value = -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder,  m.decodedπ))   #FIXME I removed m.decodedπ since it has laready sigmoid
    
    else

        # continuous_weight = (m.num_continuous_vars + m.num_binary_vars)/ m.num_continuous_vars
        # binary_weight = (m.num_continuous_vars + m.num_binary_vars) / m.num_binary_vars

        continuous_weight = 1
        binary_weight = 1


        log_likelihood = continuous_weight * logp_x_z(X, latz, m) + binary_weight * logp_x_z_bernouli(X, latz, m)
        kld_loss = kl_q_p(μ, logσ)#/latent_dim


        lower_bound = (log_likelihood - m.β * kld_loss)/ len    

        loss_value =  -lower_bound + args.λ * sum(x->sum(x.^2), Flux.params(m.decoder, m.decodedμ, m.decodedlogσ,  m.decodedπ)) #FIXME I removed m.decodedπ since it has laready sigmoid
    end

    reconstruction_loss = - log_likelihood

    return loss_value, reconstruction_loss, kld_loss
end

function trainVAE!(preprocessed_data, original_data, dataTypeArray, preprocess_ps, args; val_data = nothing)

    Random.seed!(args.seed)

    if args.multimodal_encoder
        if length(unique(dataTypeArray)) == 1
            println("Only one modality (data type) exists in the dataset. Switched to vanilla VAE")
            m = init_vanilla_vae(args, dataTypeArray)
        else
            m = init_multimodal_vae(args, dataTypeArray)
        end
    else
        m = init_vanilla_vae(args, dataTypeArray)
    end

    ps = getparams(m)

    
    loss_array_vae = []
    loss_array_reconstruction = [] 
    loss_array_kld = []

    if args.cross_validation_flag
        loss_array_vae_val = []
        loss_array_reconstruction_val = [] 
        loss_array_kld_val = []
    end

    training_data = get_data(preprocessed_data, args.batch_size)

    opt = ADAM(args.η)

    for i = 1:args.epochs 

        if i % args.verbose_freq == 0
            print("Epoch $(i): loss = ") 
        end
        
        for batch in training_data
            gs = gradient(ps) do
                loss_value, reconstruction_loss, kld_loss = loss(batch, m, preprocess_ps)
                loss_value
            end

            update!(opt, ps, gs)
        end

        loss_mean, reconstruction_loss_mean, kld_loss_mean = average_loss!(training_data, m, loss_array_vae, loss_array_reconstruction, loss_array_kld, preprocess_ps)
        loss_mean_val, reconstruction_loss_mean_val, kld_loss_mean_val = average_loss!(val_data, m, loss_array_vae_val, loss_array_reconstruction_val, loss_array_kld_val, preprocess_ps)

        if i % args.verbose_freq == 0
            println(loss_mean)
        end

        if args.tblogger_flag
            !ispath(args.save_path) && mkpath(args.save_path)
    
            with_logger(m.tblogger_object) do


                if args.cross_validation_flag
                    @info "VAE_loss_val" train_and_val = (train=loss_mean,  val=loss_mean_val) 
                    @info "VAE_reconstruction_loss_val" train_and_val = (train=reconstruction_loss_mean, val=reconstruction_loss_mean_val) log_step_increment=0
                    @info "VAE_kld_loss_val" train_and_val = (train=kld_loss_mean, val=kld_loss_mean_val) log_step_increment=0
                else
                    @info "VAE_loss" loss_mean
                    @info "VAE_reconstruction_loss" reconstruction_loss_mean log_step_increment=0
                    @info "VAE_kld_loss" kld_loss_mean log_step_increment=0
                end
            end
        end

    end
    
    save_vae_results(training_data, preprocessed_data, original_data, m, preprocess_ps, args, loss_array_vae)

    if args.cross_validation_flag
        return m, training_data, (loss_array_reconstruction, loss_array_reconstruction_val)
    else
        return m, training_data, loss_array_vae
    end
end




function trainVAE_hyperparams_opt!(preprocessed_data, original_data, dataTypeArray, preprocess_ps, args)

    

    #FIXME later you can add the possibility of saving and reading from the saved model
    

    hyperparams = Hyperparams()

    for args.batch_size = hyperparams.batch_size,
        args.η = hyperparams.learning_rate,
        args.epochs = hyperparams.epochs,
        args.β = hyperparams.beta,
        args.latent_dim = hyperparams.latent_dim,
        args.hidden_dim = hyperparams.hidden_dim, #Int.(floor.(hyperparams.hidden_dim_to_input .* args.input_dim)),
        args.multimodal_encoder = hyperparams.multimodal_encoder,
        args.seed = hyperparams.seed


        Random.seed!(args.seed)

        if args.multimodal_encoder
            if length(unique(dataTypeArray)) == 1
                println("Only one modality (data type) exists in the dataset. Switched to vanilla VAE")
                m = init_vanilla_vae(args, dataTypeArray)
            else
                m = init_multimodal_vae(args, dataTypeArray)
            end
        else
            m = init_vanilla_vae(args, dataTypeArray)
        end

        ps = getparams(m)

        
        loss_array_vae = []
        loss_array_reconstruction = [] 
        loss_array_kld = []


        training_data = get_data(preprocessed_data, args.batch_size)

        opt = ADAM(args.η)

        for i = 1:args.epochs 

            if i % args.verbose_freq == 0
                print("Epoch $(i): loss = ") 
            end
            
            for batch in training_data
                gs = gradient(ps) do
                    loss_value, reconstruction_loss, kld_loss = loss(batch, m, preprocess_ps)
                    loss_value
                end

                update!(opt, ps, gs)
            end

            loss_mean, reconstruction_loss_mean, kld_loss_mean = average_loss!(training_data, m, loss_array_vae, loss_array_reconstruction, loss_array_kld, preprocess_ps)

            if i % args.verbose_freq == 0
                println(loss_mean)
            end

            if args.tblogger_flag
                !ispath(args.save_path) && mkpath(args.save_path)
        
                with_logger(m.tblogger_object) do
                    @info "VAE_loss" loss_mean
                    @info "VAE_reconstruction_loss" reconstruction_loss_mean log_step_increment=0
                    @info "VAE_kld_loss" kld_loss_mean log_step_increment=0
                end
            end

        end
        
        save_vae_results(training_data, preprocessed_data, original_data, m, preprocess_ps, args, loss_array_vae)

    end
end

function get_latent(input, m::vanilla_vae, args::Args, preprocess_ps::preprocess_params)
    (μ, logσ) = encode(input, m)
    latvals = latentz(μ, logσ)

    return latvals
end


function VAE_output(input, m::vanilla_vae, args::Args, preprocess_ps::preprocess_params, sampling_method::String, split_post_fix ="")

    Random.seed!(args.seed)

    data = input.data

    glm_model_death_region = nothing 

    if sampling_method == "prior"

        # use Standard Normal and producing these Zs instead of getting them from Original Data 
        if args.IPW_sampling

            z = get_latent(data, m, args, preprocess_ps)
            
            glm_model_death_region = get_glm_model(args, preprocess_ps, preprocessed_data, z)
        
            weighted_samples, normalized_IPW_grid, propensity_score_average, Z1, Z2 = weighted_sampling(glm_model_death_region, preprocessed_data, z, args)

            latvals = weighted_samples

        else
            # use Standard Normal and producing these Zs instead of getting them from Original Data 
            truesig = fill(0.0, args.latent_dim, args.latent_dim)
            truesig[diagind(truesig)] .= 1.0
            truemu = fill(0.0, args.latent_dim)
            latvals = rand(Distributions.MvNormal(truemu,truesig), size(data, 2))
        end
    

    elseif sampling_method == "posterior"

        latvals = get_latent(data, m, args, preprocess_ps)

    end

    if m.AIQN
        order = find_order(latvals')
        latvals = (qfind(latvals', order))'
    end

    output = zeros(size(data,1), size(data,2))
    

    Random.seed!(args.seed)
    
    hidden = m.decoder(latvals)

    if m.num_binary_vars == 0

        if args.scaling && args.scaling_method == "scaling"
            output = rand.(truncated.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))); lower=0, upper=1))
        else
            output = rand.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))))
        end

        

    elseif m.num_continuous_vars == 0
        output = rand.(Bernoulli.(m.decodedπ(hidden)))

    else
        
        if args.scaling && args.scaling_method == "scaling"
            
            output[m.feature_type, :] = rand.(truncated.(Normal.(m.decodedμ(convert.(Float64, hidden)), exp.(m.decodedlogσ(convert.(Float64,hidden)))); lower=0, upper=1))
        else
            output[m.feature_type, :] = rand.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))))
        end

        output[.!m.feature_type, :] = rand.(Bernoulli.(m.decodedπ(hidden)))
    end


    

    if sampling_method == "posterior"
        mkdir(string(logdir(m.tblogger_object), "/posterior_sampling$(split_post_fix)"))
        writedlm(string(logdir(m.tblogger_object), "/posterior_sampling$(split_post_fix)/VAE_output.csv"),  output, ',')
    else
        mkdir(string(logdir(m.tblogger_object), "/prior_sampling$(split_post_fix)"))
        writedlm(string(logdir(m.tblogger_object), "/prior_sampling$(split_post_fix)/VAE_output.csv"),  output, ',')
    end

    if args.scaling
        output = reverse_standardize(preprocess_ps.μ, preprocess_ps.σ, output)
    end

    
   

    if args.pre_transformation

        if preprocess_ps.pre_transformation_type == "power"

            output = output'

            output = reverse_power(preprocess_ps, output, dataTypeArray)
        
            output = reverse_BC(preprocess_ps, output, dataTypeArray)
        
        else
            output = output'

            
            output = inverse_quantile_transform(preprocess_ps.qt_array, output, dataTypeArray)  # Inverse transform the data


        end

        return output, glm_model_death_region 
    else

        return output', glm_model_death_region

    end
end



function get_data(x, batch_size)
    DataLoader(x, batchsize=batch_size, shuffle=true)
end


function get_latent(input, m::multimodal_vae, args::Args, preprocess_ps::preprocess_params)
    (binary_μ, binary_logσ), (continuous_μ, continuous_logσ)  = encode(input, m)

    if args.synthetic_data
        latvals = vcat(latentz(binary_μ, binary_logσ), latentz(continuous_μ, continuous_logσ))
    else
        latvals = latentz((binary_μ .+ continuous_μ) ./ 2, (binary_logσ .+ continuous_logσ)./2)
    end

    return latvals
end
function VAE_output(input, m::multimodal_vae, args::Args, preprocess_ps::preprocess_params, sampling_method::String, split_post_fix = "")

    Random.seed!(args.seed)

    data = input.data

    glm_model_death_region = nothing 

    if sampling_method == "prior"

        # use Standard Normal and producing these Zs instead of getting them from Original Data 
        if args.IPW_sampling

            z = get_latent(data, m, args, preprocess_ps)

            glm_model_death_region = get_glm_model(args, preprocess_ps, preprocessed_data, z)
        
            weighted_samples, normalized_IPW_grid, propensity_score_average, Z1, Z2 = weighted_sampling(glm_model_death_region, preprocessed_data, z, args)

            latvals = weighted_samples
        else
            if args.synthetic_data
                truesig = fill(0.0, 2 * args.latent_dim, 2 * args.latent_dim)
                truesig[diagind(truesig)] .= 1.0
                truemu = fill(0.0, 2 * args.latent_dim)
                zmat_normal = (collect(rand(Distributions.MvNormal(truemu,truesig), size(data, 2))))
                latvals = zmat_normal 
            else
                # use Standard Normal and producing these Zs instead of getting them from Original Data 
                truesig = fill(0.0, args.latent_dim, args.latent_dim)
                truesig[diagind(truesig)] .= 1.0
                truemu = fill(0.0, args.latent_dim)
                zmat_normal = collect(rand(Distributions.MvNormal(truemu,truesig), size(data, 2)))
                latvals = zmat_normal 
            end
        end

    elseif sampling_method == "posterior"

        (binary_μ, binary_logσ), (continuous_μ, continuous_logσ)  = encode(data, m)


        if args.synthetic_data
            latvals = vcat(latentz(binary_μ, binary_logσ), latentz(continuous_μ, continuous_logσ))
        else
            latvals = latentz((binary_μ .+ continuous_μ) ./ 2, (binary_logσ .+ continuous_logσ)./2)
        end
    end

    # display(Plots.scatter(latvals[1,:], latvals[2,:], title = "Latent Space", xlabel = "z1", ylabel = "z2", legend = false, size = (500, 500)))

    if m.AIQN
        order = find_order(latvals')
        latvals = (qfind(latvals', order))'
    end

    # display(Plots.scatter(latvals[1,:], latvals[2,:], title = "Latent Space", xlabel = "z1", ylabel = "z2", legend = false, size = (500, 500)))

    output = zeros(size(data,1), size(data,2))


    Random.seed!(args.seed)
    
    hidden = m.decoder(latvals)

    if m.num_binary_vars == 0
        if args.scaling_method == "scaling"
            output = rand.(truncated.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))); lower = 0, upper = 1))
        else
            output = rand.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))))
        end

    elseif m.num_continuous_vars == 0
        output = rand.(Bernoulli.(m.decodedπ(hidden)))

    else
        if args.scaling_method == "scaling"
            output[m.feature_type, :] = rand.(truncated.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))); lower = 0, upper = 1))
        else
            output[m.feature_type, :] = rand.(Normal.(m.decodedμ(hidden), exp.(m.decodedlogσ(hidden))))
        end
        output[.!m.feature_type, :] = rand.(Bernoulli.(m.decodedπ(hidden)))
    end

    if sampling_method == "posterior"
        mkdir(string(logdir(m.tblogger_object), "/posterior_sampling$(split_post_fix)"))
        writedlm(string(logdir(m.tblogger_object), "/posterior_sampling$(split_post_fix)/VAE_output.csv"),  output, ',')
    else
        mkdir(string(logdir(m.tblogger_object), "/prior_sampling$(split_post_fix)"))
        writedlm(string(logdir(m.tblogger_object), "/prior_sampling$(split_post_fix)/VAE_output.csv"),  output, ',')
    end

    if args.scaling
        output = reverse_standardize(preprocess_ps.μ, preprocess_ps.σ, output)
    end

    
    if args.pre_transformation

        if preprocess_ps.pre_transformation_type == "power"

            output = output'

            output = reverse_power(preprocess_ps, output, dataTypeArray)


            output = reverse_BC(preprocess_ps, output, dataTypeArray)

        else

            output = output'

            
            output = inverse_quantile_transform(preprocess_ps.qt_array, output, dataTypeArray)  # Inverse transform the data


        end

        
        return output, glm_model_death_region 
    else

        return output', glm_model_death_region

    end
    
end


function average_loss!(data, m, loss_array_vae, loss_array_reconstruction, loss_array_kld, preprocess_ps)
    loss_value, reconstruction_loss, kld_loss = loss(data.data, m, preprocess_ps)
    loss_mean = mean(loss_value)
    reconstruction_loss_mean = mean(reconstruction_loss)
    kld_loss_mean = mean(kld_loss)
    
    append!(loss_array_vae, loss_mean)
    append!(loss_array_reconstruction, reconstruction_loss_mean)
    append!(loss_array_kld, kld_loss_mean)

    return loss_mean, reconstruction_loss_mean, kld_loss_mean
end


function weighted_sampling(glm_model_death_region, preprocessed_data, z, args)
    probabilities_death_region = predict_probability_region(glm_model_death_region, preprocessed_data', args)

    Z1, Z2, _, propensity_score_average = grid_for_heatmap(z, args.grid_point_size, probabilities_death_region)

    # include only propesity_scores which |propensity_score - 0.5| < args.δ or 0.05
    
    if args.subpopulation_mode == 2
        propensity_score_average[findall(x -> abs(x - 0.5) > args.δ, propensity_score_average)] .= -Inf
    elseif args.subpopulation_mode == 0
        propensity_score_average[findall(x -> x > (0.5 + args.δ), propensity_score_average)] .= -Inf
    else 
        propensity_score_average[findall(x -> x < (0.5 - args.δ), propensity_score_average)] .= -Inf
    end

    normalized_IPW_grid = normalize_IPW(compute_IPW.(propensity_score_average))

    kernel = Kernel.gaussian(1)
    normalized_IPW_grid = imfilter(normalized_IPW_grid , kernel) 

    num_samples =  length(probabilities_death_region)

    weighted_samples_Bernouli = sample_form_prior_with_IPW_grid_with_Bernouli(Z1, Z2, normalized_IPW_grid, args.grid_point_size, num_samples)

    return weighted_samples_Bernouli, normalized_IPW_grid, propensity_score_average, Z1, Z2
end


