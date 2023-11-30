cd("../.") 
using Pkg
if isfile("Project.toml") && isfile("Manifest.toml")
    Pkg.activate(".")
end

Pkg.instantiate()

#FIXME make the including and using clean. you are including usings in irrelevant files. also including with revise you can check it including only one file and see it is running independently and then continue
using IJulia
using Revise

includet("src/structs.jl")
includet("GAN/GAN.jl")
includet("src/transformations.jl")
includet("src/visualization/plotting_paper.jl")
includet("src/visualization/visualization.jl")
includet("src/load_data.jl")
includet("src/report.jl")


x, dataTypeArray, args = load_dataset()

args.pre_transformation = false
args.AIQN = false 
args.scaling = true
args.user_description_on_run = "This run has been done on $(today()) at $(Dates.format(now(), "HH:MM"))... It is a GAN experiment"

preprocess_ps = preprocess_params(input_dim = args.input_dim)

preprocessed_data, preprocess_ps = preprocess!(args, preprocess_ps, x)


# FIXME you can merge args and GAN here. The hyperparameter tuning part is a bit weird
if args.data_string == "toy"
    gen_input_dim = 2
    gen_hidden_dim = 10
    dis_hidden_dim = 10
    batch_size = 50
    epochs = 1000
    η = 1e-4
elseif args.data_string == "sim"
    gen_input_dim = 15
    gen_hidden_dim = 30
    dis_hidden_dim = 25
    batch_size = 50
    epochs = 1000
    η = 1e-5
else
    gen_input_dim = 15
    gen_hidden_dim = 30
    dis_hidden_dim = 25
    batch_size = 50
    epochs = 1000
    η = 1e-5
end

Random.seed!(11)

# define GAN
generator = Chain(Dense(gen_input_dim, gen_hidden_dim, leakyrelu), Dense(gen_hidden_dim, args.input_dim, σ))
discriminator = Chain(Dense(args.input_dim, dis_hidden_dim, leakyrelu), Dense(dis_hidden_dim, 1, σ))
gan = GAN(generator, discriminator, randn, gen_input_dim, η, dataTypeArray, batch_size, epochs)

# train gan
gan, loss_array_gan, gen_error = train_GAN!(gan, preprocessed_data', x)
