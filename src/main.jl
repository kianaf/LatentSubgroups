
using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end

Pkg.instantiate()

# #FIXME make the including and using clean. you are including usings in irrelevant files. also including with revise you can check it including only one file and see it is running independently and then continue
# using IJulia


using Revise

includet("structs.jl")
includet("transformations.jl")
includet("AIQN/AIQN.jl")
includet("VAE.jl")
includet("visualization/plotting_paper.jl")
includet("visualization/visualization.jl")
includet("load_data.jl")
includet("src/evaluation/evaluation.jl")
includet("classification.jl")


x, dataTypeArray, args = load_dataset()

# x = x[:,dataTypeArray]
# dataTypeArray = dataTypeArray[dataTypeArray]

# args.input_dim = length(dataTypeArray)
# args.hidden_dim = length(dataTypeArray)

# args.user_description_on_run::String = "This run has been done on $(today()) at $(Dates.format(now(), "HH:MM"))... just binary + correlation idea"



Random.seed!(11)


# preprocess_ps = load_struct("./runs/run_209/pre_transformation/preprocess_params.bson")
# preprocessed_data = Matrix(CSV.read("./runs/run_209/pre_transformation/standardization/standardized_data.csv", DataFrame, header = false))'

preprocess_ps = preprocess_params(input_dim = args.input_dim)
preprocessed_data, preprocess_ps = preprocess!(args, preprocess_ps, x)





if args.hyperopt_flag
    trainVAE_hyperparams_opt!(preprocessed_data, x, dataTypeArray, preprocess_ps, args)
else
    model, training_data, loss_array_vae = trainVAE!(preprocessed_data, x, dataTypeArray, preprocess_ps, args)
end