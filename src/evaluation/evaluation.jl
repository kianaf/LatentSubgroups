using DecisionTree
using DataFrames
using CSV
using Random
using Statistics: mean, std
using GLM
# using KernelDensity
using StatsModels
using MLBase:predict 

####################################### Propensity Score ############################################
function utility(run_path, x = nothing, syn = nothing)
    Random.seed!(11)


    if !isnothing(x) && !isnothing(syn)
        comb = vcat(syn, x)
        n = size(x,1)
    elseif isnothing(x) && isnothing(syn)
        x = Matrix(CSV.read(string(run_path, "/original_data.csv"), DataFrame))
        n = size(x,1)
        syn = Matrix(CSV.read(string(run_path, "/vae/prior_sampling/synthetic_data_prior.csv"), DataFrame))
        comb = vcat(syn, x)
    elseif isnothing(x) && !isnothing(syn)
        x = Matrix(CSV.read(string(run_path, "/original_data.csv"), DataFrame))
        n = size(x,1)
        comb = vcat(syn, x)
    else
        println("Please specify a path or a matrix for original data!")
    end

    labelTsyn = fill(1,  size(comb)[1]-size(x)[1],1)
    labelTx = fill(0, size(x)[1],1)
    labelT =  vcat(labelTsyn, labelTx)

    model = DecisionTreeClassifier(max_depth=15, min_samples_leaf = 20)  # maximum depth should be tuned using cv

    DecisionTree.fit!(model, comb, labelT[:,1])
    println(model)
    
    P = [predict_proba(model, comb[i,:]) for i=1:n+n]

    pMSE = (1/(n + n)) * sum([((P[i][2].- (n/(n+n))).^2) for i=1:n+n])
    # Up = (1/(n + n)) * sum([(abs(P[i][2].- (n/(n+n)))) for i=1:n+n])
    
    println("pMSE: ", pMSE)
    
    numberOfPer = 100
    pMSE_per = fill(0.0, numberOfPer)
    
    #permute labels 
    for i = 1:numberOfPer
        PerLabelT = labelT[randperm(length(labelT))]
        model = DecisionTreeClassifier(max_depth=15, min_samples_leaf = 20)  # maximum depth should be tuned using cv
        # fit!(model, comb.data, labelT[:,1])
        DecisionTree.fit!(model, comb, PerLabelT[:,1])
        P = [predict_proba(model, comb[j,:]) for j=1:n+n]
        # predictedLabels = [predict(model, comb.data[i,:])> 0.50 ? 1.0 : 0.0 for i=1:n+n]
        pMSE_per[i] = (1/(n + n)) * sum([((P[j][2].- (n/(n+n))).^2) for j=1:n+n])
    end
    
    pMSE_per_mean = mean(pMSE_per)
    pMSE_per_std = std(pMSE_per)
    
    pMSE_ratio = pMSE / pMSE_per_mean
    Standardize_pMSE = (pMSE - pMSE_per_mean)/pMSE_per_std

    println("pMSE_ratio: " ,pMSE_ratio)
    println("Standardize_pMSE: ", Standardize_pMSE)
end

####################################### Logistic Regression ############################################

function logistic_regression(run_path, x = nothing, syn = nothing)
    Random.seed!(11)


    if !isnothing(x) && !isnothing(syn)
        comb = vcat(syn, x)
        n = size(x,1)
    elseif isnothing(x) && isnothing(syn)
        x = Matrix(CSV.read(string(run_path, "/original_data.csv"), DataFrame, header = false))
        n = size(x,1)
        syn = Matrix(CSV.read(string(run_path, "/vae/prior_sampling/synthetic_data_prior.csv"), DataFrame, header = false))
        comb = vcat(syn, x)
    elseif isnothing(x) && !isnothing(syn)
        x = Matrix(CSV.read(string(run_path, "/original_data.csv"), DataFrame))
        n = size(x,1)
        comb = vcat(syn, x)
    else
        println("Please specify a path or a matrix for original data!")
    end

    labelTsyn = fill(1,  size(comb)[1]-size(x)[1],1)
    labelTx = fill(0, size(x)[1],1)
    labelT =  vcat(labelTsyn, labelTx)


end



# function fit_logistic_regression_outcome(preprocessed_data)

#     df = DataFrame(preprocessed_data, :auto)

#     colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])

#     @show colnames

#     rename!(df, colnames)

#     fm = @formula(y ~ x1 + x2 + x3 + x4a + x4b + x5 + x6 + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
    
#     # fit a logistic regression model to the data
#     model = glm(fm, df, Binomial(), LogitLink())

#     features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x6", "x7", "x8", "x9a", "x9b", "x10","x11","x12", "x13", "x15", "x16", "x17", "x18"]

#     selected_features = features[coeftable(model).cols[4].<0.05]

    
#     formula_str = "y ~ "

#     for feature in selected_features
#         if feature == features[1]
#             formula_str = string(formula_str, feature)
#         else
#             formula_str = string(formula_str, "+", feature)
#         end
#     end
    

#     fm = @eval(@formula($(Meta.parse(formula_str))))
#     selected_model = glm(fm, df, Binomial(), LogitLink())


#     return selected_model, Set(selected_features)
# end



function read_data(run_path, original = true)

    args = load_struct(string(run_path, "/args.bson"))

    if original
    
        data = Matrix(CSV.read(string(run_path, "/../original_data.csv"), DataFrame, header = false))
        
        if args.IPW_sampling
            
            if args.subpopulation_mode == 0
                data = data[load_region(args.data_string).!="EU-NORTH", :]
            elseif args.subpopulation_mode == 1
                data = data[load_region(args.data_string).=="EU-NORTH", :]
            end
        end

        
    
    else
        data = Matrix(CSV.read(string(run_path, "/prior_sampling/synthetic_data_prior.csv"), DataFrame, header = false))
    end

    return data
end


function fit_logistic_regression_outcome_simulation(data)
    Random.seed!(11)


    df = DataFrame(data, :auto)
    colnames = names(CSV.read("./data/simulation.csv" , DataFrame, header = true))
    rename!(df,colnames)

    
    # insertcols!(df, size(df,2)+1, :E_O => exposure_outcome)
    fm = @formula(y ~ x1 + x2 + x3 + x4a + x4b + x5 + x6 + x7 + x8 + x9a + x9b + x10 + x11 + x12 + x13 + x15 + x16 + x17 + x18)
                 
    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    features = ["x1", "x2", "x3", "x4a", "x4b", "x5", "x6", "x7", "x8", "x9a", "x9b", "x10", "x11", "x12", "x13", "x15", "x16", "x17", "x18"]

    selected_features = features[vec(coeftable(model).cols[4].< 0.05)[2:end]]


    # This is the Logistic regression-based model which selects the features based on the p-value score of the feature. 
    # The features with p-value less than 0.05 are considered to be the more relevant feature.

    return selected_features, model
end




function fit_logistic_regression_outcome_ist(data)
    Random.seed!(11)


    df = DataFrame(convert.(Float64,data), :auto)
    data_string = "ist_randomization_data_smaller_no_west_no_south_aug5"
    colnames = names(CSV.read("./data/ist_randomization_data_smaller_no_west_no_south_aug5.csv" , DataFrame, header = true))[1:end-2]  # no region and country

    rename!(df,colnames)


    formula_str = "FDEAD ~ "

    for feature in colnames
        if feature != "FDEAD"
        
            if feature == colnames[1]
                formula_str = string(formula_str, feature)
            else
                formula_str = string(formula_str, "+", feature)
            end
        end
    end


    # fm = @formula(FDEAD ~ RCONSC1 + RCONSC2 + RDELAY + SEX + AGE + RSLEEP + RATRIAL + RCT + RVISINF + RHEP24 + RASP3 + RSBP + RXASP + RXHEP)


    fm = @eval(@formula($(Meta.parse(formula_str))))

       
    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())



    selected_features = colnames[1:end-1][vec(coeftable(model).cols[4].< 0.05)[2:end]]


    # This is the Logistic regression-based model which selects the features based on the p-value score of the feature. 
    # The features with p-value less than 0.05 are considered to be the more relevant feature.

    return selected_features, model
end




function fit_logistic_regression_outcome_ist(data, args)
    Random.seed!(11)


    df = DataFrame(data, :auto)
    colnames = names(CSV.read("./data/$(args.data_string).csv" , DataFrame, header = true))[1:end-2]  # no region and country


    @show colnames

    # colnames[1] = "RCONSC2"
    # colnames= vcat("RCONSC1", colnames)
    rename!(df,colnames)



    formula_string = string("FDEAD ~ ",  join(["$(colnames[i])" for i in 1:length(colnames)], " + "))

    fm = @eval(@formula($(Meta.parse(formula_string))))

    @show fm

    # formula_str = "FDEAD ~ "

    # for feature in colnames
    #     if feature == colnames[1]
    #         formula_str = string(formula_str, feature)
    #     else
    #         formula_str = string(formula_str, "+", feature)
    #     end
    # end


    # fm = @eval(@formula($(Meta.parse(formula_str))))


    # fit a logistic regression model to the data
    model = glm(fm, df, Binomial(), LogitLink())

    selected_features = colnames[vec(coeftable(model).cols[4].< 0.05)[2:end]]


    # This is the Logistic regression-based model which selects the features based on the p-value score of the feature. 
    # The features with p-value less than 0.05 are considered to be the more relevant feature.

    return selected_features, model
end



# function fit_logistic_regression_selected_features(preprocessed_data, selected_features_e, selected_features_y)

#     df = DataFrame(preprocessed_data, :auto)
#     colnames = names(CSV.read("./data/data_scenario1.csv" , DataFrame)[:, 2:end])
#     rename!(df,colnames)

#     select_features = selected_features_e
#     features = collect(intersect(select_features, selected_features_y))

#     formula_str = "Exposure ~ "

#     for feature in features
#         if feature == features[1]
#             formula_str = string(formula_str, feature)
#         else
#             formula_str = string(formula_str, "+", feature)
#         end
#     end
    

#     fm = @eval(@formula($(Meta.parse(formula_str))))
#     selected_model = glm(fm, df, Binomial(), LogitLink())




#     return selected_model, features
# end



function select_features(X, y, alpha= 0.02)
    # Fit a logistic regression model with the given alpha value
    model = glm(@formula(y ~ X), Binomial(), X, y, alpha = alpha)

    # Select the features with non-zero coefficients
    selected_features = X[:, coefficients(model) .!= 0]

    return selected_features
end





function predict_probability_outcome(model, data, data_string)
    df = DataFrame(data, :auto)
    
    if contains(data_string,"ist")
        colnames = names(CSV.read("./data/$(data_string).csv" , DataFrame))[1:end-2]  # no region and country
 
        rename!(df,colnames)
    elseif data_string == "sim"
        df = DataFrame(data, :auto)
        colnames = names(CSV.read("./data/simulation.csv" , DataFrame))
        rename!(df,colnames)
    end

    prediction = round.(GLM.predict(model, df))

    if contains(data_string,"ist")
        accuracy = sum(df.FDEAD .== prediction)/length(prediction)
    else
        accuracy = sum(df.y .== prediction)/length(prediction)
    end
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






function round_discrete(synthetic, x)
    n = size(x,1)
    p = size(x,2)
    output = fill(0.0, n,p)
    for i = 1:p
        if count(x[:,i].%1 .!=0) ==0
            output[:,i] = round.(synthetic[:,i])
        else
            output[:,i] = synthetic[:,i]
        end
    end
    return output
end
