using CSV
using DataFrames
using Revise
using Random


includet("structs.jl")
# includet("report.jl")

function load_dataset(method = "VAE")

    args = Args()

    if method =="GAN"
        args.save_path = "GAN/$(args.save_path)"

    elseif args.AIQN
        args.save_path = "AIQN/$(args.save_path)"
    end

    !isdir(string(pwd(), "/", args.save_path)) && mkdir(string(pwd(), "/", args.save_path))
    args.current_run_path = get_log_dir( args.save_path, args.tblogger_flag) 
    args.tblogger_object = args.current_run_path == nothing ? nothing : TBLogger(args.current_run_path)

    if (args.data_string == "ist_new") || args.data_string == "ist2d"|| args.data_string == "ist2d"|| args.data_string == "ist2d_subset"|| (args.data_string == "ist_more_features")||(args.data_string == "ist_more_features_2") ||(args.data_string == "ist_more_features_no_west") || (args.data_string =="ist_randomization_data_march15")||(contains(args.data_string, "ist_randomization"))
        x = convert.(Float64, Matrix(CSV.read("./data/$(args.data_string).csv", DataFrame, header = true))[:, 1:end-2])

    elseif args.data_string == "ist" 
        
        #### preprocess of IST removing NA and redifinition of RCONSC
        Random.seed!(11)
        # m = 200

        x_temp = Matrix(CSV.read("./data/$(args.data_string).csv", DataFrame))
        # data_withheader = Matrix(CSV.read("./data/ist.csv", header=false))

        p = size(x_temp)[2] +1 # because we have "RCONSC1", "RCONSC2" instead of RCONSC

        n = size(x_temp)[1]
        # header = data_withheader[1, 2:p-1]
        # header = vcat("RCONSC1", "RCONSC2", header)

        cnt = 1

        x = fill(0, n-count(x_temp[:,p-1].=="NA"), p)
        for i = 1:n
            if x_temp[i,p-1]!="NA"
                if x_temp[i,1]==0
                    x[cnt,1] = 0
                    x[cnt,2] = 0
                elseif x_temp[i,1]==1
                    x[cnt,1] = 1
                    x[cnt,2] = 0
                else
                    x[cnt,1] = 1   # FIXME ichanged it from zero to one.
                    x[cnt,2] = 1
                end
                
                x[cnt,3:p-1] = x_temp[i,2:p-2]

                x[cnt,p] = Base.parse(Int64, x_temp[i,p-1])
                cnt+=1
            end    
        end

        n = size(x)[1]

    elseif args.data_string == "data_scenario1" || args.data_string == "data_scenario2"
        x = Matrix(CSV.read("./data/$(args.data_string).csv" , DataFrame))[:, 2:end]
        
        x = hcat(x[:,1:6], x[:,8:end-2])
        
        
    elseif args.data_string == "sim" 
        Random.seed!(11)
        n=2500
        p=21
        # m = 50

        x = Matrix(CSV.read("./data/simulation.csv" , DataFrame))

        # data_withheader = convert(Array, CSV.read("C:\\Users\\farhadyar\\Desktop\\Simulation Design\\data_scenario1_Bimodal_Binary.csv", header=false))
        # header = data_withheader[1, 1:p]


    elseif args.data_string == "toy"
        x = generate_toy_example()
    elseif contains(args.data_string, "sc_dec")
        # your own data
        x = convert.(Float32, Matrix(transpose(Matrix(CSV.read(string("./data/$(args.data_string).csv") , DataFrame))[:,2:end])))

    elseif args.data_string == "cancer"

         #### preprocess of IST removing NA and redifinition of RCONSC
         Random.seed!(11)
         m = 200
 
         x_temp = Matrix(CSV.read("./data/cancer.csv", DataFrame))[:, 3:end]
         # data_withheader = Matrix(CSV.read("./data/ist.csv", header=false))

         x_temp[x_temp[:,2].== 2, 2] .= 0
 
         p = size(x_temp)[2] +1
 
         n = size(x_temp)[1]
         # header = data_withheader[1, 2:p-1]
         # header = vcat("RCONSC1", "RCONSC2", header)
 
 
         x = fill(0, n-count(x_temp[:,p-1].=="NA"), p)
         x[:, 1:end-2] = x_temp[:, 1:end-1]
         for i = 1:n
            if x_temp[i,end]=="Low"
                x[i, end-1] = 0
                x[i, end] = 0
            elseif x_temp[i,end]=="Medium"
                x[i, end-1] = 0
                x[i, end] = 1
            else
                x[i, end-1] = 1
                x[i, end] = 1
            end
         end
         

         n = size(x)[1]
    else
        x = CSV.read("./data/$(args.data_string).csv" , header = true, DataFrame)

        x = filter(row -> (row.GFR != "NA"),  x)
        x = filter(row -> (row.Nodes != "NA"),  x)
        # x = filter(row -> (!contains.(row.Einzeldosis, "/" ) && !contains.(row.Einzeldosis, " ") && !contains.(row.Einzeldosis, "HART")),  x) #FIXME I am removing rows with complicated dosis
        x = select(x, Not([:Einzeldosis, :DurchlaufendeNummer, :Zentrum]))

        types = eltype.(eachcol(x))

        for i in 1:length(types)
            if ((types[i] != Int64) && (types[i] != Float64))

                x[!, i] = parse.(Float64,string.(x[!, i]))
            end

        end

        x = Matrix(x)
 
    end

    n = size(x)[1]
    p = size(x)[2]

    dataTypeArray = fill(false, p)

    for i = 1:p
        for j = 1:n
            if x[j,i]!=0 && x[j,i]!=1
                dataTypeArray[i] = true
                break
            end
        end
    end

    args.input_dim = p
    args.hidden_dim = args.input_dim #Int(floor(2p/3)) #,args.input_dim 


    # x = x[shuffle(1:end), :]

    writedlm(string(args.current_run_path, "/", "original_data.csv"),  x, ',')

    return x, dataTypeArray, args
end


function load_exposure(data_string)

    exposure = Matrix(CSV.read("./data/$(data_string).csv" , DataFrame))[:, end-1]
end

function load_outcome(data_string)

    if contains(args.data_string,"ist")
        outcome = Matrix(CSV.read("./data/$(data_string).csv" , DataFrame))[:, end-2]
    else
        outcome = Matrix(CSV.read("./data/$(data_string).csv" , DataFrame))[:, end]

    end
    return outcome
end

function load_country(data_string)
    country = Matrix(CSV.read("./data/$(data_string).csv" , DataFrame))[:, end-1]

end



function load_region(data_string)

    region = Matrix(CSV.read("./data/$(data_string).csv" , DataFrame))[:, end]
end



# NOTE: this function is just loading the synthetic data but before reverse transformation

function load_from_run_for_propensity_score_plotting(run_number, vae_number)

    run_path = "runs/run_$(run_number)"

    # vae_dir = ""

    if vae_number ==0
        vae_dir = string(run_path, "/vae")
    else
        vae_dir = string(run_path, "/vae_$(vae_number)")
    end
    model = load_model(string(vae_dir, "/vae.bson"))
    preprocess_ps = load_struct(string(run_path, "/pre_transformation/preprocess_params.bson"))
    preprocessed_data = Matrix(CSV.read(string(run_path, "/pre_transformation/scaling/scaled_data.csv"), DataFrame, header = false))'
    # syn = Matrix(CSV.read(string(vae_dir, "/prior_sampling/vae_output.csv"), DataFrame, header = false))'
    syn = Matrix(CSV.read(string(vae_dir, "/prior_sampling/synthetic_data_prior.csv"), DataFrame, header = false))
    args = load_struct(string(vae_dir, "/args.bson"))
    z = get_latent(preprocessed_data, model, args, preprocess_ps)

    model, args, preprocess_ps, preprocessed_data, syn, z, vae_dir

end


#######################################Producing different distributions for testing############################################
function initialize_Normal(p, n)
    Random.seed!(11)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.MvNormal(truemu ,truesig),Int(n))'))
    x = x[randperm(n),:]
    return x
end

function initialize_skewed(p, n)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.Gumbel(0,0.15),n)'))
    x = vcat(x,(collect(rand(Distributions.Gumbel(0,0.15),n)')))
    x= x'
    x = x[randperm(n),:]
    return x
end

function initialize_bimodal(p, n)
    truesig = fill(0.0,p,p)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0,p)
    x = (collect(rand(Distributions.MvNormal(truemu .+0 ,truesig),Int(7n/10))'))
    x = vcat(x,collect(rand(Distributions.MvNormal(truemu .+ 4 ,truesig  ),Int(3n/10))'))
    x = x[randperm(n),:]
    return x
end


function generate_toy_example()
    Random.seed!(42)
    n = 5000
    p = 2
    x1 = initialize_skewed(p,n)
    x2 = initialize_bimodal(p,n)
    x = hcat(x1[:,1], x2[:,1])
    return x
end

# Function to split a matrix into n smaller matrices by rows
function split_matrix_randomly_by_rows(mat::AbstractMatrix, n::Int)
    # Determine the size of the original matrix
    rows, cols = size(mat)  
    
    # Calculate the number of rows in each smaller matrix
    subrows = Int(ceil(rows/n))
    
    # Ensure the matrix can be split evenly by rows
    # if rows % n != 0
    #     println("Number of rows in the matrix must be divisible by number of folds")
    # end
    
    # Shuffle the rows of the matrix
    shuffled_indices = shuffle(1:rows)
    shuffled_mat = mat[shuffled_indices, :]
    
    # Initialize an array to hold the smaller matrices
    submatrices = []
    
    # Loop through and create each submatrix
    for i in 0:n-1
        if i == n-1
            submat = shuffled_mat[(i*subrows+1):end, :]
        else
            submat = shuffled_mat[(i*subrows+1):(i+1)*subrows, :]
        end
        push!(submatrices, submat)
    end
    
    return submatrices
end



# Function to create training and validation sets for cross-validation
function create_cross_validation_sets(mat::AbstractMatrix, n::Int)
    submatrices = split_matrix_randomly_by_rows(mat, n)
    cross_val_sets = []
    
    for i in 1:n
        val_set = submatrices[i]
        train_set = vcat(submatrices[1:i-1]..., submatrices[i+1:end]...)
        
        push!(cross_val_sets, (train_set, val_set))
    end
    
    return cross_val_sets
end