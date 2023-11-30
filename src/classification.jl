using DecisionTree
using DataFrames
using CSV
using Random
using Statistics: mean, std
using GLM
# using KernelDensity
using StatsModels
using MLBase:predict 

using Symbolics

# abstract type PivotingStrategy end
# struct NoPivot <: PivotingStrategy end
# struct RowNonZero <: PivotingStrategy end
# struct RowMaximum <: PivotingStrategy end
# struct ColumnNorm <: PivotingStrategy end
# abstract type AbstractNode{T} end

# using Econometrics


    
    
function unzip(input)
    fist_element, second_element = zip(input...) 

    return fist_element, second_element
end


function label_region(data_string)

    labels = load_region(data_string)

    unique_labels = unique(labels)

    region_label_dict = Dict()

    label_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    for i in 1:length(unique_labels)

        region_label_dict[unique_labels[i]] = label_letters[i]

    end

    println(region_label_dict)
    return map(x->region_label_dict[x], labels)

end

####################################### Region Classification ############################################
function classify_regions(model_path, data_path, classification_method)

    args = load_struct(string(model_path, "/args.bson"))
    Random.seed!(11)

    preprocess_ps = load_struct(string(data_path, "/pre_transformation/preprocess_params.bson"))
    preprocessed_data = Matrix(CSV.read(string(data_path, "/pre_transformation/scaling/scaled_data.csv"), DataFrame, header = false))'

    # column_names = Matrix(CSV.read(string(model_path, "../original_data.csv"), DataFrame, header = false))[1, :]
 
    vae_model = load_model(string(model_path, "/vae.bson"))


    

    z = get_latent(preprocessed_data, vae_model, args, preprocess_ps)

    # z = x'

    sample_size = size(z', 1)

    labels =  label_region(load_struct(string(data_path, "/vae/args.bson")).data_string)



    data = [z'[i, :] for i = 1: size(z', 1)]
   
    zipped_vals = [zip(data[i,:], labels[i]) for i in 1:sample_size]
    
    shuffle!(zipped_vals)

    # @show 
    data = reduce(hcat, collect(unzip(zipped_vals[i])[1])[1] for i = 1:sample_size)'



    labels = [unzip(zipped_vals[i])[2][1] for i = 1:sample_size]

    
    if size(data, 2) == 2

        df = DataFrame(z1 = data[:,1], z2 = data[:,2], REGION = labels)
    elseif size(data, 2) == 3
        df = DataFrame(z1 = data[:,1], z2 = data[:,2], z3 = data[:,3], REGION = labels)
    end

    # @show df

    df_train = df[1:Int(floor(0.8 * size(df,1))), :]

    df_test = df[Int(floor(0.8 * size(df,1)))+1:end, :]

    if classification_method == "glm"
        glm_model = fit(EconometricModel,
            @formula(REGION ~ z1 + z2),
            df_train,
            contrasts = Dict(:REGION => DummyCoding(base = "AFRICA")))


        predictions = predict(glm_model, df_test)

        @show predictions


    elseif classification_method == "lm"

        model, predictions = multinomial_logistic_regression(df_train, df_test)

    elseif classification_method =="dt"

        model = DecisionTreeClassifier(max_depth=25, min_samples_leaf = 10)  # maximum depth should be tuned using cv   , 


        if size(df_train,2) == 3

            DecisionTree.fit!(model, Matrix(df_train[!,1:2]), vec(df_train[!,3])) 

            # predictions = DecisionTree.predict(model, Matrix(df_test[!,1:2]))
            predictions = DecisionTree.predict(model, Matrix(df_train[!,1:2]))

        elseif size(df_train,2) == 4
            DecisionTree.fit!(model, Matrix(df_train[!,1:3]), vec(df_train[!,4])) 

            # predictions = DecisionTree.predict(model, Matrix(df_test[!,1:3]))
            predictions = DecisionTree.predict(model, Matrix(df_train[!,1:3]))
        end


        
    end
    # true_values = df_test.REGION
    true_values = df_train.REGION

    return true_values, predictions, model


end

####################################### Logistic Regression ############################################

function multinomial_logistic_regression(df_train, df_test)

    unique_labels = unique(df_train.REGION)

    z_dims = size(df_train, 2) - 1
    features = ["z$(i)" for i = 1:z_dims]

    # @show df

    # @variables fm[1:length(unique_labels)]

    # @variables model[1:length(unique_labels)]
    # println(df)

    models = []
    for i in 1:length(unique_labels)

        println("Train model for $(unique_labels[i])")

        df_train[:, string(unique_labels[i])] = convert.(Int, df_train.REGION .== unique_labels[i])
        df_test[:, string(unique_labels[i])] = convert.(Int, df_test.REGION .== unique_labels[i])

        if z_dims == 2

            formula_str = "$(unique_labels[i]) ~ z1 + z2"
        elseif z_dims == 3
            formula_str = "$(unique_labels[i]) ~ z1 + z2 + z3"
        end
        fm = @eval(@formula($(Meta.parse(formula_str))))
        model = glm(fm, df_train, Binomial(), LogitLink())


        selected_variables =  features[coeftable(model).cols[4][2:end].<0.05]
        
        if (length(selected_variables) != z_dims) && (length(selected_variables) != 0)
            formula_str = "$(unique_labels[i]) ~ "
            for i = 1:length(selected_variables)
                if i ==1
                    formula_str = string(formula_str, selected_variables[i])
                else
                    formula_str = string(formula_str, "+ ", selected_variables[i])
                end
            end

            fm = @eval(@formula($(Meta.parse(formula_str))))
            
            
            model = glm(fm, df_train, Binomial(), LogitLink())

        end

        @show model
        push!(models, model)
    end

    # prob_predictions = reduce(hcat, [GLM.predict(models[i], df_test) for i = 1: length(unique_labels)])
    prob_predictions = reduce(hcat, [GLM.predict(models[i], df_train) for i = 1: length(unique_labels)])

    mxval, mxindx = findmax( prob_predictions; dims=2)

    indexes = [mxindx[i][2] for i = 1:size(prob_predictions, 1)]

    predictions = map(x->unique_labels[x], indexes)


    return models, predictions
end