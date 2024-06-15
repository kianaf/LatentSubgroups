using GLM, DataFrames
using KernelDensity
using StatsModels
using CSV
using MLBase:predict 


function fit_logistic_regression_exposure(preprocessed_data, x6_inclusion)

    Random.seed!(42)

    
    
    df = DataFrame(preprocessed_data, :auto)

    if x6_inclusion
        colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
        fm = @formula(Exposure ~ x1 + x2 + x3 + x4a + x4b + x5 + x6 + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
        features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x6", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18"]
    else
        colnames = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18", "Exposure", "y"]
        fm = @formula(Exposure ~ x1 + x2 + x3 + x4a + x4b + x5  + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
        features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18"]
    end
    
    rename!(df, colnames)




    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    
    selected_features = features[(coeftable(model).cols[4].< 0.05)[2:end]]

    
    formula_str = "Exposure ~ "

    for feature in selected_features
        if feature == features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, "+", feature)
        end
    end
    

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())


    return selected_model, Set(selected_features)
end




function fit_logistic_regression_outcome(preprocessed_data, x6_inclusion)
    Random.seed!(42)

    df = DataFrame(preprocessed_data, :auto)

    if x6_inclusion
        
        colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
        fm = @formula(y ~ x1 + x2 + x3 + x4a + x4b + x5 + x6 + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
        features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x6", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18"]
    else
        
        colnames = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18", "Exposure", "y"]
        fm = @formula(y ~ x1 + x2 + x3 + x4a + x4b + x5  + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
        features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18"]
    end
    


    rename!(df,colnames)
    

    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    @show model

    selected_features = features[(coeftable(model).cols[4].< 0.05)[2:end]]


    return Set(selected_features)
end

function fit_logistic_regression_selected_features_both(preprocessed_data, selected_features_e, selected_features_y, x6_inclusion)

    Random.seed!(42)

    df = DataFrame(preprocessed_data, :auto)

    if x6_inclusion
        colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
    else
        
        colnames = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18", "Exposure", "y"]
    end
    


    rename!(df,colnames)

    select_features = selected_features_e
    features = collect(intersect(select_features, selected_features_y))

    # @show features  

    formula_str = "Exposure ~ "

    for feature in features
        if feature == features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, "+", feature)
        end
    end
    

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())




    return selected_model, features
end



function fit_logistic_regression_selected_features_either(preprocessed_data, selected_features_e, selected_features_y, x6_inclusion)

    Random.seed!(42)


    df = DataFrame(preprocessed_data, :auto)

    if x6_inclusion
        
        colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
    else
        
        colnames = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18", "Exposure", "y"]
    end
    

    rename!(df,colnames)

    select_features = selected_features_e
    features = collect(union(select_features, selected_features_y))

    # @show features  

    formula_str = "Exposure ~ "

    for feature in features
        if feature == features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, "+", feature)
        end
    end
    

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())




    return selected_model, features
end




function select_features(X, y, alpha= 0.02)
    # Fit a logistic regression model with the given alpha value
    model = glm(@formula(y ~ X), Binomial(), X, y, alpha = alpha)

    # Select the features with non-zero coefficients
    selected_features = X[:, coefficients(model) .!= 0]

    return selected_features
end





function predict_probability(model, preprocessed_data, x6_inclusion)

    df = DataFrame(preprocessed_data, :auto)
    if x6_inclusion
        colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
    else
        
        colnames = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18", "Exposure", "y"]
    end

    rename!(df,colnames)

    GLM.predict(model, df)

end


function latent_matching(latentz, E, max_distance)
    npoints = size(latentz, 2)
    latent_index = vcat(latentz, collect(1:npoints))
    matching_dict = Dict()

    for i = 1:npoints
        if E[i] == 1
            closest_distance = max_distance
            for j in 1
                if E[j] == 1
                    break
                else
                    distance = sum((latent_index[1:2, i] .- latent_index[1:2, j]) .^2)
                    
                    if distance < closest_distance
                        closest_distance = distance
                        matching_dict[latent_index[3,i]] = latent_index[3,j]
                    end
                end
            end

        end
    end

    return matching_dict
end



function fit_logistic_regression_region(preprocessed_data, args)

    Random.seed!(42)

    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-2])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    REGION = DataFrame(REGION = (CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end] .== "EU-NORTH"))

   
    df = hcat(df, REGION)
    

    formula_string = string("REGION ~ ",  join(["$(colnames[i])" for i in 1:length(colnames)], " + "))

    

    fm = @eval(@formula($(Meta.parse(formula_string))))

    
    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    # # @show model
    # # @show coeftable(model).cols[4]

    selected_features = colnames[(coeftable(model).cols[4].< 0.05)[2:end]]

    
    formula_str = "REGION ~ FDEAD +"

    for feature in selected_features
        if feature == selected_features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, " + ", feature)
        end
    end
    
    
    # @show formula_str

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())


    return selected_model , Set(selected_features)
end




function fit_logistic_regression_death(preprocessed_data, args)

    Random.seed!(42)

    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-3])

    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    FDEAD = DataFrame(FDEAD = CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end-2])

   
    df = hcat(df, FDEAD)
    

    formula_string = string("FDEAD ~ ",  join(["$(colnames[i])" for i in 1:length(colnames)], " + "))

    fm = @eval(@formula($(Meta.parse(formula_string))))

    
    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    # # @show model
    # # @show coeftable(model).cols[4]

    selected_features = colnames[(coeftable(model).cols[4].< 0.05)[2:end]]

    
    formula_str = "FDEAD ~ "

    for feature in selected_features
        if feature == selected_features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, " + ", feature)
        end
    end


    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())


    return selected_model , Set(selected_features)
end





function fit_logistic_regression_death_given_region(preprocessed_data, args)

    Random.seed!(42)

    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-3])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    FDEAD = DataFrame(FDEAD = CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end-2])
    REGION = DataFrame(REGION = (CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end] .== "EU-NORTH"))

   
    df = hcat(df, FDEAD, REGION)
    

    formula_string = string("FDEAD ~ REGION + ",  join(["$(colnames[i])" for i in 1:length(colnames)], " + "))

    

    fm = @eval(@formula($(Meta.parse(formula_string))))

    
    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    # # @show model
    # # @show coeftable(model).cols[4]

    features = vcat("REGION", colnames)

    selected_features = features[(coeftable(model).cols[4].< 0.05)[2:end]]

    
    formula_str = "FDEAD ~ "

    for feature in selected_features
        if feature == selected_features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, " + ", feature)
        end
    end
    
    # @show formula_str

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())


    return selected_model , Set(selected_features)
end





function predict_probability_region(model, preprocessed_data, args)

    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-2])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    REGION = DataFrame(REGION = (CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end] .== "EU-NORTH"))

   
    df = hcat(df, REGION)
    

    GLM.predict(model, df)

end



function predict_probability_death(model, preprocessed_data, args)
    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-3])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    FDEAD = DataFrame(FDEAD = CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end-2])

   
    df = hcat(df, FDEAD)
    

    GLM.predict(model, df)

end




function predict_probability_death_given_region(model, preprocessed_data, args)
    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-3])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    FDEAD = DataFrame(FDEAD = CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end-2])

    REGION = DataFrame(REGION = (CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end] .== "EU-NORTH"))

   
    df = hcat(df, FDEAD, REGION)
    

    GLM.predict(model, df)

end



function fit_logistic_regression_selected_features_region_death(preprocessed_data, selected_features_region, selected_features_death)

    df = DataFrame(preprocessed_data, :auto)

    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, 1:end-2])
    rename!(df, colnames)
    # @show CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end]
    FDEAD = DataFrame(FDEAD = CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end-2])
    REGION = DataFrame(REGION = (CSV.read("./data/$(args.data_string).csv" , DataFrame)[:, end] .== "EU-NORTH"))

   
    # df = hcat(df, FDEAD, REGION)
    df = hcat(df, REGION)
    
    features = collect(intersect(selected_features_region, selected_features_death))

    formula_str = "REGION ~ FDEAD +"

    for feature in features
        if feature == features[1]
            formula_str = string(formula_str, feature)
        else
            formula_str = string(formula_str, "+", feature)
        end
    end
    

    fm = @eval(@formula($(Meta.parse(formula_str))))
    selected_model = glm(fm, df, Binomial(), LogitLink())




    return selected_model, features
end




# Compute Inverse Probability Weights
function compute_IPW(propensity_score)

    if propensity_score == -Inf
        IPW = 0
    elseif propensity_score >= 0.5 
        IPW = 1 / propensity_score 
    else
        IPW = (1 / (1 - propensity_score))
    end

    return IPW
end


# Normalize Inverse Probability Weights
function normalize_IPW(IPW)
    IPW = IPW ./ sum(IPW)
    return IPW
end


# # Fit a kernel density estimator to the latent space and the IPW list 
# function fit_kernel_density(latentz, probabilities)
#     Random.seed!(42)


#     # truesig = fill(0.0, 2, 2)
#     # truesig[diagind(truesig)] .= 1.0
#     # truemu = fill(0.0, 2)
#     # latentz = (collect(rand(Distributions.MvNormal(truemu,truesig), length(probabilities))))
    
#     bivariate_kde = kde(latentz', weights=probabilities) #kde(([(latentz[1,i], latentz[2,i]) for i = 1:length(probabilities)], probabilities))

#     @show(bivariate_kde.y)
#     # bivariate_kde.density =  probabilities

#     # sample(bivariate_kde, 1000)
#     cs = cgrad([colorant"#1D4A91", colorant"white",  colorant"#AE232F"])
#     display( Plots.contourf(bivariate_kde.x, bivariate_kde.y, bivariate_kde.density, color=cs))
#     return bivariate_kde
#     # bivariate_kde
# end


function sample_point_from_IPW_grid(X, Y, Z)

    # Flatten and compute cumulative probabilities
    flat_Z = vec(Z)
    cumulative_probs = cumsum(flat_Z)

    # Sample a random number
    r = rand()

    # Find the corresponding index
    index = findfirst(cumulative_probs .>= r)

    # Compute the boundaries of the cell
    rows, cols = size(X)
    row = div(index - 1, cols) + 1
    col = rem(index - 1, cols) + 1

    # Sample a point uniformly within the cell
    x = rand(Uniform(X[row, col], X[row, col + 1]))
    y = rand(Uniform(Y[row, col], Y[row + 1, col]))

    return (x, y)
end




function sample_form_prior_with_IPW_grid(Z1, Z2, normalized_IPW_grid, grid_point_size, num_samples)

    Random.seed!(42)

    truesig = fill(0.0, 2, 2)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0, 2)

    samples = []

    

    num_samples_grid = ceil.(normalized_IPW_grid .* num_samples)

    num_samples_sampled = zeros(size(num_samples_grid))

    # @show num_samples_sampled

    for i= 1:num_samples
        sample_not_found_flag = true
        while(sample_not_found_flag)
            s = rand(Distributions.MvNormal(truemu,truesig))
            # @show sample

            z1_index = findfirst((Z1.+grid_point_size./2) .> s[1])
            if isnothing(z1_index)
                if s[1]>= Z1[end]
                    z1_index = length(Z1)
                elseif s[1] <= Z1[1]
                    z1_index = 1
                end

            end
            
            z2_index = findfirst((Z2.+grid_point_size./2) .> s[2])
            if isnothing(z2_index)
                if s[2]>= Z2[end]
                    z2_index = length(Z2)
                elseif s[2] <= Z2[2]
                    z2_index = 1
                end

            end

            # @show(z1_index, z2_index)

            if num_samples_grid[z1_index, z2_index] > num_samples_sampled[z1_index, z2_index]
                num_samples_sampled[z1_index, z2_index] += 1
                sample_not_found_flag = false
                push!(samples, s)
            end
        end
        

    end

    # @show size(num_samples_sampled)

    hcat(samples...)
end



# with Bernoulli




function sample_form_prior_with_IPW_grid_with_Bernouli(Z1, Z2, normalized_IPW_grid, grid_point_size, num_samples)

    Random.seed!(42)

    truesig = fill(0.0, 2, 2)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0, 2)

    samples = []

    

    # num_samples_grid = ceil.(normalized_IPW_grid .* num_samples)

    # num_samples_sampled = zeros(size(num_samples_grid))

    # @show num_samples_sampled

    for i= 1:num_samples
        sample_not_found_flag = true
        while(sample_not_found_flag)
            s = rand(Distributions.MvNormal(truemu,truesig))
            # @show sample

            z1_index = findfirst((Z1.+grid_point_size./2) .> s[1])
            if isnothing(z1_index)
                if s[1]>= Z1[end]
                    z1_index = length(Z1)
                elseif s[1] <= Z1[1]
                    z1_index = 1
                end

            end
            
            z2_index = findfirst((Z2.+grid_point_size./2) .> s[2])
            if isnothing(z2_index)
                if s[2]>= Z2[end]
                    z2_index = length(Z2)
                elseif s[2] <= Z2[2]
                    z2_index = 1
                end

            end

            # @show(z1_index, z2_index)

            # if num_samples_grid[z1_index, z2_index] > num_samples_sampled[z1_index, z2_index]
            #     num_samples_sampled[z1_index, z2_index] += 1
            #     sample_not_found_flag = false
            #     push!(samples, s)
            # end

            if !isnothing(z1_index) && !isnothing(z2_index) && s[1]>=Z1[1] && s[2]>=Z2[1] && s[1]<=Z1[end] && s[2]<=Z2[end]

                if rand(Bernoulli(normalized_IPW_grid[z1_index, z2_index]))
                    sample_not_found_flag = false
                    push!(samples, s)
                
                end
            end
        end
        

    end

    # @show size(num_samples_sampled)

    hcat(samples...)
end


# make the grid for standard normal distribution with the same grid as propensity score grid


# Define the range and step size
# function get_prior_grid_for_custom_range(Z1, Z2, grid_point_size)
#     grid_range_Z1 = collect(Z1[1]:grid_point_size:Z1[end]) .+ grid_point_size/2   #Z1[1]:grid_point_size:Z1[end]#(Z1[1] - (grid_point_size/2)):grid_point_size: (Z1[end] + (grid_point_size/2))
#     grid_range_Z2 = collect(Z2[1]:grid_point_size:Z2[end]) .+ grid_point_size/2 #(Z2[1] - (grid_point_size/2)):grid_point_size: (Z2[end] + (grid_point_size/2))

#     # Create an empty grid
#     grid_size = length(grid_range)
#     pdf_grid = zeros(grid_size, grid_size)

#     # Fill the grid with the PDF values
#     for (i, x) in enumerate(grid_range_Z1)
#         for (j, y) in enumerate(grid_range_Z2)
#             pdf_grid[i, j] = (1 / (2 * π)) * exp(-0.5 * (x^2 + y^2))
#         end
#     end
#     pdf_grid ./= sum(pdf_grid)
# end



function get_glm_model(args, preprocess_ps, preprocessed_data, z, load = false, vae_dir = nothing)

    #FIXME why the load struct is not working?
    if load 
        model = load_model(string(vae_dir, "/prior_sampling/IPW_sampling/glm_model_death_region.bson"))
        # @show BSON.parse(string(vae_dir, "/prior_sampling/IPW_sampling/glm_model_death_region.bson"))
        # model = BSON.parse(string(vae_dir, "/prior_sampling/IPW_sampling/glm_model_death_region.bson"))
        return model
    else
        
    
        REGION = (load_region(args.data_string).=="EU-NORTH")

        glm_model_death, selected_features_death = fit_logistic_regression_death(preprocessed_data'[:,1:end-1], args)
        
        glm_model_region, selected_features_region = fit_logistic_regression_region(preprocessed_data'[:,1:end], args)
        
        glm_model_death_region, selected_features_death_region = fit_logistic_regression_selected_features_region_death(preprocessed_data'[:,1:end], selected_features_region, selected_features_death)

        return glm_model_death_region
    end

    
end
