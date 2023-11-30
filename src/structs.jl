
using Parameters: @with_kw
using TensorBoardLogger: TBLogger, logdir, tb_overwrite, set_step_increment!
using TensorBoardLogger
using Logging: with_logger, NullLogger
using Dates
using BSON: @save, @load
using Revise


@with_kw mutable struct preprocess_params

    input_dim::Int = 9                                   # number of features

    λ2::Array{Float32} = fill(0.0, input_dim)             # shifting value for box cox transformation
    λ1::Array{Float32} = fill(1.0, input_dim)             # power value for box cox transformation
    box_cox_epochs::Int = 5000                           # number of epochs for training box cox transformation parameters
    box_cox_η::Float32 = 1e-3                            # learning rate for training box cox power

    μ::Array{Float32} = fill(0.0, input_dim)             # μ for scaling by μ and 2σ is
    σ::Array{Float32} = fill(0.5, input_dim)             # σ for scaling by μ and 2σ is

    shift::Array{Float32} = fill(0.0, input_dim)         # shifting value for power transformation
    peak1::Array{Float32} = fill(1.0, input_dim)         # peak1 value for power transformation
    peak2::Array{Float32} = fill(1.0, input_dim)         # peak2 value for power transformation
    peak_rng::Array{Float32} = fill(1.0, input_dim)      # the value which decides where to put the -1 and 1 regarding the peaks
    min::Array{Float32} = fill(0.0, input_dim)           # minimum value for power transformation
    max::Array{Float32} = fill(0.0, input_dim)           # maximum value for power transformation
    power::Array{Float32} = fill(1.0, input_dim)         # power value for power transformation
    power_epochs::Int = 50000                             # number of epochs for training power transformation parameters  
    power_η::Float32 = 1e-3                              # learning rate for training power transformation parameters
end



#FIXME where to put learning rate and the others? it is VAE related but some are data related the reason I did not seperate is that some are being used as inputs one solution can be hyperparameters and data structs
@with_kw mutable struct Args

    data_string::String = "ist_randomization_data_smaller_no_west_no_south_aug5" #"sim" #"ist_randomization_data_smaller_no_west_no_south_aug5" #"sim" # "data_scenario1" # "ist_randomization_data_smaller_no_west_no_south_aug5"
    #"ist_randomization_data_smaller_POLA_SWED_aug15"   ##"ist_randomization_data_smaller_no_west_no_south_no_treatment_aug8"#"ist_randomization_data_smaller_aug3" #"ist_randomization_data_aug3"# "ist_new" #"ist_more_features_no_west"#"ist_randomization_data_july14"  # "ist2d_subset" #  "ist_more_features_2"#"ist_more_features_2" #"ist_more_features"  #"data_scenario1"#"hnscc_my_version" #"sc_dec_19"                                                   # specify the dataset you want to use "toy", "sim" or "ist" (in case of having your own csv file use the name before .csv) 
    η::Float32 = 1e-4                                                                                                # learning rate for training VAE
    λ::Float32 = 0.01f0
    β::Float64 = 0.5                                                                                            # regularization paramater
    batch_size::Int = 128                                                                                             # batch size
    epochs::Int = 500                                                                                                 # number of epochs for training VAE
    seed::Int = 42                                                                                                   # random seed
    input_dim::Int = 9                                                                                               # number of features
    latent_dim::Int = 2                                                                                               # latent dimension
    hidden_dim::Int = 9                                                                                              # hidden dimension
    verbose_freq::Int = 100                                                                                           # logging for every verbose_freq iterations #FIXME

    tblogger_flag::Bool = true 
    save_path::String = "runs"                                                                                       # results path
    current_run_path::String = ""                                                # path with the run number assigned to this run
    
    hyperopt_flag::Bool = false                                                                                      # True if we do hyper parameter optimization False if we read from args                                                                            
    tblogger_object::Union{TBLogger, Nothing} = nothing                   # TBLogger object

    pre_transformation::Bool =  true                                                                                 # PTVAE (pre_transformation = true) or Standard VAE
    bimodality_score_threshold::Float32 = 0                                                                          # When zero, bimodality_flag becomes always true.
    
    scaling::Bool = true                                                                                              # true when scaling is requested
    scaling_method::String = "scaling" # or "standardization"  # or "scaling"
                                                                                    
    AIQN::Bool = false                                                                                               # true if autoregressive implicit quantile networks method is used for training VAE
    multimodal_encoder::Bool = true                                                                                 # true if we train different encoders for different modalities in the VAE                                                                                           

    synthetic_data::Bool = false                                                                                     # if true, we generate synthetic data and if false we are interested in latent structure (2 dimensional)
    
    IPW_sampling::Bool = true                                                                                        # if true, we use IPW sampling for training VAE
    subpopulation_mode::Int = 2                                                                                         # 0 which is class 0 and 1 which is class 1 and 2 which is both individuals common to both                                                           
    grid_point_size::Float32 = 0.2
    δ::Float32 = 0.1                                                                                                     
    user_description_on_run::String = "This run has been done on $(today()) at $(Dates.format(now(), "HH:MM"))..."   # user description on the Run. The default is the date and the time of the run
end

#FIXME decide if you want to move the scaling and pre-transformation flag out of the model or not
mutable struct vanilla_vae
    encoder
    encodedμ
    encodedlogσ
    decoder
    decodedμ
    decodedlogσ
    decodedπ
    feature_type    # feature type true is continuous and false is binary
    num_continuous_vars
    num_binary_vars
    AIQN
    β     # multiplier for the KL divergence 
    tblogger_object                   # TBLogger object
end

mutable struct multimodal_vae
    binary_encoder
    continuous_encoder
    binary_encodedμ
    continuous_encodedμ
    binary_encodedlogσ
    continuous_encodedlogσ
    decoder
    decodedμ
    decodedlogσ
    decodedπ
    feature_type    # feature type true is continuous and false is binary
    num_continuous_vars
    num_binary_vars
    AIQN
    β     # multiplier for the KL divergence 
    tblogger_object
end


mutable struct GAN
    g                       # generator
    d                       # discriminator
    pz                      # code distribution
    gen_input_dim           # generator input dimension
    η                       # learning rate
    feature_type            # feature type true is continuous and false is binary
    batch_size              # batch size for training the GAN
    epochs                  # number of iterations to train the GAN
end


function save_struct(object, path)
    @save path object
    return 
end

#IST
# @with_kw mutable struct Hyperparams
#     batch_size = [16, 32, 64]
#     learning_rate = [1e-3, 1e-4, 1e-5]
#     epochs = [1000] 
#     beta = [0.5, 0.4]
#     latent_dim = [2] 
#     hidden_dim = [15]#[20, 30, 40]
#     seed = [11, 42] 
#     multimodal_encoder = [false, true]
# end

# simulation Hyperparameter
@with_kw mutable struct Hyperparams
    multimodal_encoder = [true, false]
    batch_size = [128]
    learning_rate = [1e-3, 1e-4]
    epochs = [1000] 
    beta = [0.1, 0.2, 0.5, 1]
    latent_dim = [2] 
    hidden_dim = [12,18, 21, 24]
    seed = [42] 
    
end


function load_struct(path)
    @load path object
    return object
end


function save_model(model, path)
    @save path model
end


#FIXME:
# UndefVarError: TensorBoardLogger not defined

function load_model(path)
    @load path model
    return model
end