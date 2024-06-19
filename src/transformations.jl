using Distributed
using Gadfly
using LinearAlgebra
using Random
using Distributions
using Flux
using Flux.Optimise: update!
using Statistics
using StatsBase: geomean
using Compose
using DelimitedFiles
using Zygote
using Colors
using DistributionsAD
using DataFrames
using TensorBoardLogger: TBLogger, tb_overwrite, set_step_increment!
using Logging: with_logger
using KernelDensity
using Peaks
using Revise
using HypothesisTests:ApproximateTwoSampleKSTest, pvalue
using StatsBase: mode
using BSON
using Plots
using LaTeXStrings

includet("quantile_transformation.jl")

# KernelDensity
# https://github.com/JuliaStats/KernelDensity.jl

# Peaks
# https://github.com/halleysfifthinc/Peaks.jl

########################################### Standardization ############################################
function standardize(μ::Array, σ::Array, x)
    return ((x .- μ') ./ 2σ')'
end

function reverse_standardize(μ::Array, σ::Array, x)
    return (x .* 2σ) .+ μ
end
##################################### BoxCox Transformation ############################################
# change abs(min) to -min 
function set_alpha!(λ2, x, dataTypeArray)
    
    min = vec(minimum(x, dims = 1))

    max = vec(maximum(x, dims = 1))

    #FIXME Here I am moving it to more than 1.01
    λ2 = @. ((min < 0) * (- min)) + 1 + 0.01  * (min <= 0) *  dataTypeArray

    return λ2
end



function set_lambda!(preprocess_ps, x, dataTypeArray, args)

    Random.seed!(args.seed)

    loss_array_λ1= []

    println("Estimation of Box-Cox transformation parameters:")
    
    opt = ADAM(preprocess_ps.box_cox_η)

    # training λ1s
    for feature_number = 1:args.input_dim
        println("\n λ1 estimation for feature $(feature_number):")
        stop = false
        if dataTypeArray[feature_number]
            ps = Flux.params(preprocess_ps.λ1)
            for epoch = 1: preprocess_ps.box_cox_epochs
                
                if epoch % args.verbose_freq ==0
                    print("Epoch $(epoch): loss = ") 
                    println(loss_array_λ1[epoch - 1]) 
                end

                if stop
                    println("λ1 estimation for feature $(feature_number) is converged at epoch $(epoch)!\n")
                    break
                else
                
                    local box_cox_loss = 0

                    gs = gradient(ps) do

                        # box_cox_loss = -log_likelihood_BoxCox(x, feature_number, preprocess_ps.λ2, preprocess_ps.λ1, dataTypeArray)
                        box_cox_loss = -log_likelihood_BoxCox_one_dimension(x[:, feature_number], preprocess_ps.λ2[feature_number], preprocess_ps.λ1[feature_number])

                        return box_cox_loss
                    end

                    update!(opt, ps, gs)

                    
                    if epoch > 1 
                        if loss_array_λ1[end] - box_cox_loss < 1e-4  
                            stop = true
                        end
                    end

                    
                    if args.tblogger_flag
                        !ispath(args.save_path) && mkpath(args.save_path)
                        
                        with_logger(args.tblogger_object) do
                            @info "train_box_cox" box_cox_loss
                        end
                    end
                
                    append!(loss_array_λ1, box_cox_loss)
                end
            end
        end
    end

    return preprocess_ps.λ1, loss_array_λ1
end


function BC_transform_one_dimension(x, λ2, λ1)

    # to prevent mutating arrays error
    output = Zygote.Buffer(x, length(x), 1)

    output =  λ1≈ 0 ? log.(x .+ λ2) : (((x .+ λ2) .^ λ1) .- 1.0) ./ λ1

    return copy(output)
end


function BC_transform_all_dimensions(x, λ2, λ1, dataTypeArray)

    # to prevent mutating arrays error
    output = Zygote.Buffer(x, size(x, 1), size(x, 2))

    output =  @. (log(x + λ2') * (λ1≈ 0)' +  ((((x + λ2') ^ λ1') - 1.0) / λ1') * !(λ1≈ 0)') * dataTypeArray' + x * !dataTypeArray'

    return copy(output)
end

function reverse_BC(preprocess_ps, x, dataTypeArray)
    
    output = copy(x)

    for i = 1:size(x, 2)

        if dataTypeArray[i]
            if preprocess_ps.λ1[i] > 0
                output[output[:, i] .< (((1.01)^preprocess_ps.λ1[i] -1)/preprocess_ps.λ1[i]), i] .= (-1/preprocess_ps.λ1[i]) + 0.0001
    
            elseif preprocess_ps.λ1[i] < 0
                output[output[:, i] .> (-1/preprocess_ps.λ1[i]), i] .= (-1/preprocess_ps.λ1[i]) - 0.0001
            end
            if (Rational(1/preprocess_ps.λ1[i])).den % 2 != 0 

                # NOTE: (-2) ^ (1/3) returns complex number! Therefore, (-2) & (1/3) ----> (sign(-2)* abs(-2))^(1/3)
                output[:, i] = @. preprocess_ps.λ1[i] ≈ 0 ? exp(output[:, i]) - preprocess_ps.λ2[i]  : sign((output[:, i]) * preprocess_ps.λ1[i] + 1.0) * (abs((output[:, i]) * preprocess_ps.λ1[i] + 1.0) ^ (1/preprocess_ps.λ1[i])) - preprocess_ps.λ2[i]
            else 

                # if the generated values are negative with even λ1we truncate the value to zero it is like we make λ1x + 1 -> 0 when it cannot be negative. We do it using relu.
                # since 0 ^ (-1/7) is Inf

                output[:, i] = @. preprocess_ps.λ1[i] ≈ 0 ? exp(output[:, i]) - preprocess_ps.λ2[i]  :  (output[:, i] * preprocess_ps.λ1[i] + 1.0) ^ (1/preprocess_ps.λ1[i])  - preprocess_ps.λ2[i]
                
            end
        end
    end

    return output
end

#######################################Power function############################################

function set_power_parameter_new!(x, original_data, preprocess_ps::preprocess_params, dataTypeArray, args::Args)  

    loss_array_power = [] 

    Random.seed!(args.seed)

    opt = ADAM(preprocess_ps.power_η)


    
    
    peaks_array, shift_array, power_tr_flag, anim_list = get_frequent_peaks_new(x, args.bimodality_score_threshold)

    

    # power_tr_flag = power_tr_flag .& (preprocess_ps.peak1 .!= -Inf) .& (preprocess_ps.peak2 .!= -Inf)
    preprocess_ps.shift = shift_array 
    preprocess_ps.peak1 =  preprocess_ps.peak1 .* peaks_array[:, 1] 
    preprocess_ps.peak2 = preprocess_ps.peak2 .* peaks_array[:, end] 


    for feature_number = 1:args.input_dim

        println("\n Power parameters estimation for feature $(feature_number):")
        
        if dataTypeArray[feature_number] & power_tr_flag[feature_number] 


            stop = false
            preprocess_ps.shift
            if args.data_string == "sim"
                ps = Flux.params(preprocess_ps.power, preprocess_ps.peak_rng)
            else
                ps = Flux.params(preprocess_ps.power, preprocess_ps.shift, preprocess_ps.peak_rng)
            end
            
            cnt = 0
            
            for epoch = 1: preprocess_ps.power_epochs

                if epoch % (args.verbose_freq * 10) == 0
                    print("Epoch $epoch: loss = ")
                end
                
                if !stop
                    local power_loss = 0
                    gs = gradient(ps) do

                    
                        power_loss = one_sigma_criteria(power_tr_all_dimensions(x[:,feature_number], preprocess_ps.shift[feature_number], preprocess_ps.peak_rng[feature_number], preprocess_ps.power[feature_number], dataTypeArray[feature_number], power_tr_flag[feature_number]), dataTypeArray[feature_number])
                        
                         
                        return power_loss
                    end

                    Flux.Optimise.update!(opt, ps, gs)        
                    
                else
                    println("Power parameters estimation for $(feature_number) is converged at epoch $(epoch)!")
                    break
                end
            
                power_loss = one_sigma_criteria(power_tr_all_dimensions(x[:,feature_number], preprocess_ps.shift[feature_number], preprocess_ps.peak_rng[feature_number], preprocess_ps.power[feature_number], dataTypeArray[feature_number], power_tr_flag[feature_number]), dataTypeArray[feature_number])                             
                
            
                if epoch > 10 
                    if abs(loss_array_power[end] - power_loss) < 1e-5  
                        cnt += 1
                        if cnt > 10
                            stop = true
                        end
                    elseif loss_array_power[end-1] - power_loss < 1e-5
                        cnt += 1
                        if cnt > 10
                            stop = true
                        end
                    end    
                end

                append!(loss_array_power, power_loss)
            
                if args.tblogger_flag
                    !ispath(args.save_path) && mkpath(args.save_path)

                    with_logger(args.tblogger_object) do
                        @info "train_power_loss" power_loss
                    end
                end
                
                if epoch % (args.verbose_freq * 10) == 0
                    println(loss_array_power[end])
                end
            end
        else
            println("The feature is not bimodal! \n")
        end
    end

    return preprocess_ps, loss_array_power, anim_list

end


function find_deepest_valley(density_vals, xaxis_vals, first_mode, second_mode)

    #for valleys we find the local maxima for the 
    valleys = findmaxima(- density_vals)[2]

    modes_x = xaxis_vals[map(x-> x in [first_mode, second_mode], density_vals)]

    valleys_x = xaxis_vals[map(x-> -x in valleys, density_vals)]

    valleys_x_between_modes = valleys_x[map(x->(x < modes_x[2]) & (x > modes_x[1]), valleys_x)]

    valleys_density_between_modes = density_vals[map(x-> x in valleys_x_between_modes, xaxis_vals)]

    deepest_valley_x = valleys_x_between_modes[valleys_density_between_modes .== maximum(valleys_density_between_modes)]

    return deepest_valley_x[1],  maximum(valleys_density_between_modes)

end

function harmonic_mean(a,b)
    return (1 / ( (1/a + 1/b) / 2 ))
end




function get_frequent_peaks(x)
    
    """
        This function returns the mroe frequent peaks in the distribution. 
        The ones with less frequency than 0.1 of mode are excluded.
    """

    input_dim = size(x, 2)
    
    peaks_array = fill(0.0, input_dim, 2)

    for feature_number in 1:input_dim

        t = x[:, feature_number]

        kernel_density = kde(t, npoints = length(t))
        density_vals = kernel_density.density
        xaxis_vals = kernel_density.x

        peaks = findmaxima(density_vals)[2]  #we need this because we are interested in the local maximas
        first_mode = maximum(peaks)

        more_frequent_peaks = []

        for i = 1:length(peaks)
            if peaks[i] > 0.1 * first_mode
                append!(more_frequent_peaks, peaks[i])
            end
        end

        membership(element) = element ∈ more_frequent_peaks

        peaks_x = xaxis_vals[membership.(density_vals)] # gets the value for the peaks

        if length(peaks_x) > 1
            # now we estimate what percentage of the data falls into the peaks range. Then in the loss function we only try to have that percentage as normal distribution behave
            
            peaks_array[feature_number, :] = [peaks_x[1], peaks_x[end]]
        else
            peaks_array[feature_number, :] = [peaks_x[1], peaks_x[1]]

        end
    end

    return peaks_array, peaks_array[:, 1] .!= peaks_array[:, 2]
end




function get_frequent_peaks_new(x, bimodality_score_threshold = 0.1)
    
    """
        This function returns the mroe frequent peaks in the distribution. 
        The ones with less frequency than 0.1 of mode are excluded.
    """

    # bandwidth_list = reverse(collect(range(0.01, step=0.1, length=40000)))
    bandwidth_list = collect(range(0.01, step= 0.1, length=40000))

    input_dim = size(x, 2)
    
    peaks_array = fill(0.0, input_dim, 2)
    shift_array = fill(0.0, input_dim)
    
    bimodality_flag = fill(false, input_dim)


    anim_list = []

    for feature_number in 1:input_dim

        density_pairs = []
        peaks_pairs = []

        if Set(x[:, feature_number]) != Set([0,1])

            for i in 1:length(bandwidth_list)
                
                t = x[:, feature_number]
                kernel_density = kde(t, bandwidth = bandwidth_list[i])
            
                density_vals = kernel_density.density
                # if maximum(density_vals)>1
                #     continue
                # end
                xaxis_vals = kernel_density.x

                push!(density_pairs, (xaxis_vals, density_vals))
    
                
                peaks = findmaxima(density_vals)[2]  #we need this because we are interested in the local maximas
                # peaks_x_for_plot = xaxis_vals[map(x->x in peaks, density_vals)]
                peaks_x_for_plot = xaxis_vals[findall(map(x->x in peaks, density_vals))]

                push!(peaks_pairs, (peaks_x_for_plot, peaks))


                first_mode = nothing
                second_mode = nothing
                second_mode_x = nothing

                if length(peaks) == 1
                    break

                elseif length(peaks)<= 5 # was 5 #FIXME
                    first_mode = maximum(peaks)
                    first_mode_x = xaxis_vals[map(x->x in [first_mode], density_vals)][1]


                    second_mode = partialsort(peaks, length(peaks)- 1)
                    second_mode_x = xaxis_vals[map(x->x in [second_mode], density_vals)][1]
                    
                    #we get the deepest valley between these two peaks
                    valley_x, valley_density = find_deepest_valley(density_vals, xaxis_vals, first_mode, second_mode) 
            
                    if !isnothing(second_mode)
                        peaks_x = xaxis_vals[map(x->x in [first_mode, second_mode], density_vals)]
                    else
                        peaks_x = xaxis_vals[map(x->x in [first_mode, first_mode], density_vals)]
                    end
                    
                    bimodality_score = abs(valley_density - second_mode)/ abs(valley_density - first_mode)
                    @show bimodality_score
                    @show feature_number
                    @show peaks_x
                    @show valley_x
                    @show bandwidth_list[i]
                    @show i

                    if bimodality_score > bimodality_score_threshold
                        if !isnothing(second_mode) 
                            bimodality_flag[feature_number] = 1
                            peaks_array[feature_number, :] = peaks_x
                            # shift_array[feature_number] = valley_x
                            # shift_array[feature_number] = (first_mode_x + valley_x)/2
                            if contains(args.data_string, "ist")
                                shift_array[feature_number] = first_mode_x 
                            else
                                shift_array[feature_number] = valley_x
                            end
                        end
                    end

                    break
                end
            
            end
        

        
            # Now, create the animation.
            step = Int(ceil(length(density_pairs)/50))

            indexes = unique(vcat(collect(1:step:length(density_pairs)), [length(density_pairs)]))
            
            @show indexes


            anim = @animate for i in indexes
                # if (i==1 )|| (i == length(density_pairs))|| (i%step == 0)
                    
                    Plots.plot(density_pairs[i][1], density_pairs[i][2], color = "#1D4A91", width = 2)
                    Plots.scatter!(peaks_pairs[i][1], peaks_pairs[i][2], color = "#AE232F", markersize = 5)


                    if i == indexes[end] && length(peaks_pairs[i][1]) == 2
                        Plots.annotate!(peaks_pairs[i][1], peaks_pairs[i][2].- 0.02, Plots.text.([L"β_1", L"β_2"], :bottom), color = "#AE232F", markersize = 10)
                    end
                
            end

            
            push!(anim_list, anim)
        end
    end


    
    @show anim_list

    return peaks_array, shift_array, bimodality_flag, anim_list
end

function power_tr_all_dimensions(x, shift, peak_rng, power, dataTypeArray, power_tr_flag)

    x_tr = Zygote.Buffer(x, size(x,1), size(x,2))
  
    x_tr = @.  (x - shift') * (dataTypeArray & power_tr_flag)' +  x * (!power_tr_flag | !dataTypeArray)'

    x_tr = @.  (relu'(x_tr) * x_tr/ (peak_rng^2)' + (1 - relu'(x_tr)) * x_tr/ (peak_rng^2)') * (dataTypeArray & power_tr_flag)' +  x * (!dataTypeArray | !power_tr_flag)' 

    x_tr = @. (relu'(x_tr) - (1 - relu'(x_tr))) * ((abs(x_tr)) ^ (1 + power ^2)')  *  (dataTypeArray & power_tr_flag)' + x * (!dataTypeArray | !power_tr_flag)'

end

function reverse_power(preprocess_ps, x, dataTypeArray)
    x_retr = copy(x)

    power_tr_flag = (preprocess_ps.peak1 .!= preprocess_ps.peak2)


    # if (dataTypeArray .& power_tr_flag) == 0 we have binary variable or we don't want the continuous to be transformed
    # if we don't have bimodal transformation this: -> x .* (.!power_tr_flag .| .!dataTypeArray)' gives x so that the following line is same as input
    
    x_retr =   root(x,  (1 .+ preprocess_ps.power .^ 2)) .* (dataTypeArray .& power_tr_flag)' .+  x .* (.!power_tr_flag .| .!dataTypeArray)'

    x_retr = @. relu'(x_retr) * x_retr * (preprocess_ps.peak_rng^2)' + (1 - relu'(x_retr)) * x_retr * (preprocess_ps.peak_rng^2)' * (dataTypeArray & power_tr_flag)' +  x * (!dataTypeArray | !power_tr_flag)' 

    x_retr = @. (x_retr + preprocess_ps.shift')  *  (dataTypeArray & power_tr_flag)' + x * (!dataTypeArray | !power_tr_flag)'

    return x_retr

end

# we define this function because the julia root function throws domain error when we have negative base
function root(x, r)
   output = @. (relu'(x) * x) ^ (1/r)' - (1 - relu'(x)) * (abs(x)^(1/r)')

    # For positive bases: (relu'(x) * x)^(1/r)
    # For negative bases: - (1 - relu'(x))*(abs(x))^(1/r)
    return output
end

#######################################Criteria for Normality############################################
function one_sigma_criteria(x, dataTypeArray)

    Med = [quantile(sort(x[:, i]), 0.5, sorted = true) for i = 1: size(x,2)]
    Std = [std(x[:, i]) for i = 1: size(x,2)] 

    quantile841 = [quantile(sort(x[:, i]), 0.8419,sorted = true) for i = 1: size(x,2)]
    quantile158 = [quantile(sort(x[:, i]), 0.1581, sorted = true) for i = 1: size(x,2)]

    value = @. (abs(quantile841 - Med - Std) + abs(Med - quantile158 - Std)) * dataTypeArray
    
    return sum(value)
end

function mean_median_distance(x, dataTypeArray)

    Med = [quantile(sort(x[:, i]), 0.5, sorted = true) for i = 1: size(x,2)]
    Mean = [mean(x[:, i]) for i = 1: size(x,2)] 
    Std = [std(x[:, i]) for i = 1: size(x,2)] 

    value = (abs.(Med .- Mean) ./Std) * dataTypeArray
    
    return sum(value)
end

function two_sigma_criteria(x, dataTypeArray)

    Med = [quantile(sort(x[:, i]), 0.5, sorted = true) for i = 1: size(x,2)]
    Std = [std(x[:, i]) for i = 1: size(x,2)] 

    quantile841 = [quantile(sort(x[:, i]), 0.8419,sorted = true) for i = 1: size(x,2)]
    quantile158 = [quantile(sort(x[:, i]), 0.1581, sorted = true) for i = 1: size(x,2)]

    quantile977 = [quantile(sort(x[:, i]), 0.9777, sorted = true) for i = 1: size(x,2)] 
    quantile02 = [quantile(sort(x[:, i]), 0.0223, sorted = true) for i = 1: size(x,2)] 

    value = @. (abs(quantile841 - Med - Std) + abs(Med - quantile158 - Std) + abs(quantile158 - quantile02 - Std) + abs(quantile977 - quantile158 - Std)) * dataTypeArray 
    
    return sum(value)
end


function three_sigma_criteria(x, dataTypeArray)

    Med = [quantile(sort(x[:, i]), 0.5, sorted = true) for i = 1: size(x,2)]
    Std = [std(x[:, i]) for i = 1: size(x,2)] 

    quantile841 = [quantile(sort(x[:, i]), 0.8419,sorted = true) for i = 1: size(x,2)]
    quantile158 = [quantile(sort(x[:, i]), 0.1581, sorted = true) for i = 1: size(x,2)]

    quantile977 = [quantile(sort(x[:, i]), 0.9777, sorted = true) for i = 1: size(x,2)] 
    quantile02 = [quantile(sort(x[:, i]), 0.0223, sorted = true) for i = 1: size(x,2)] 

    quantile997 = [quantile(sort(x[:, i]), 0.003, sorted = true) for i = 1: size(x,2)] 
    quantile003 = [quantile(sort(x[:, i]), 0.997, sorted = true) for i = 1: size(x,2)] 

    value = @. (abs(quantile841 - Med - Std) + abs(Med - quantile158 - Std) + abs(quantile158 - quantile02 - Std) + abs(quantile977 - quantile158 - Std) + abs(quantile997 - quantile977 - Std) + abs(quantile02 - quantile003 - Std)) * dataTypeArray
    
    return sum(value) 
end


function log_likelihood_BoxCox_one_dimension(x, λ2, λ1)

    N = size(x, 1)

    σ² = var(BC_transform_one_dimension(x, λ2, λ1), corrected = false)

    # the multiplication by dataTypeArray is to exclude the log part for the binary features
    -N / 2.0 * log(σ² + eps()) + (λ1- 1) * sum(log.(x .+ λ2 .+ eps())) 

end

function log_likelihood_BoxCox(x, λ2, λ1, dataTypeArray)

    N = size(x, 1)

    σ² = [dataTypeArray[i] .* var(BC_transform_all_dimensions(x, λ2, λ1, dataTypeArray)[:, i], corrected = false) for i = 1:size(x, 2)]

    # the multiplication by dataTypeArray is to exclude the log part for the binary features
    sum(-N / 2.0 .* log.(σ² .+ eps()) .+ (λ1.- 1) .* sum(log.(x .+ λ2' .+ eps()) .* dataTypeArray', dims =1)') 

end

#########################Change the type of data to the same type of original data########################

function round_discrete(synthetic, x)
    @show size(synthetic)
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
    @show output
    return output
end

########################## Preprocess function ###################################
function preprocess!(args::Args, preprocess_ps::preprocess_params, x, dataTypeArray)

    if args.pre_transformation && sum(dataTypeArray) > 0

        if preprocess_ps.pre_transformation_type == "power"
            Random.seed!(args.seed)

            ####################### ‌Box-Cox transformation ###########################

            preprocess_ps.λ2 = set_alpha!(preprocess_ps.λ2, x, dataTypeArray)
        
            preprocess_ps.λ1, loss_array_λ1= set_lambda!(preprocess_ps, x, dataTypeArray, args)

            x_tr_BC = BC_transform_all_dimensions(x, preprocess_ps.λ2, preprocess_ps.λ1, dataTypeArray)

            ####################### Power transformation #############################

            preprocess_ps, loss_array_power, anim_list = set_power_parameter_new!(x_tr_BC, x, preprocess_ps, dataTypeArray, args)

            x_tr_power = power_tr_all_dimensions(x_tr_BC, preprocess_ps.shift, preprocess_ps.peak_rng, preprocess_ps.power, dataTypeArray, preprocess_ps.peak2 .!= preprocess_ps.peak1 )

            data = x_tr_power
        elseif preprocess_ps.pre_transformation_type == "quantile"
            data = fill(0.0, size(x,1), size(x,2))
            
            for col = 1:size(x, 2)
                if dataTypeArray[col]
                    qt = fit_quantile_transformer(x[:, col]; preprocess_ps.n_quantiles)  # Fit the transformer
                    data[:, col] = quantile_transform(qt, x[:, col])  # Transform the data to follow a normal distribution
                    push!(preprocess_ps.qt_array, qt)
                else
                    data[:, col] = x[:, col]
                    empty_qt = QuantileTransformer([], 1000)
                    push!(preprocess_ps.qt_array, empty_qt)
                end
            end

        end

    else
        if sum(dataTypeArray) == 0
            println("All the variables are binary! No pre-transformation can be performed.")
        end

        data = x
    end
        


    if args.scaling
        
        if args.scaling_method == "standardization"

            preprocess_ps.μ = vec(mean(data, dims=1)) .* dataTypeArray
            preprocess_ps.σ = (vec(std(data, dims=1)) .* dataTypeArray) .+ (0.5 * .!dataTypeArray)

        else
            preprocess_ps.μ = vec(minimum(data, dims=1)) .* dataTypeArray 
            preprocess_ps.σ = vec(maximum(data, dims=1) .- minimum(data, dims=1)) .* dataTypeArray .* 0.5 .+ (1 .- dataTypeArray) .* 0.5

        end

        data = standardize(preprocess_ps.μ, preprocess_ps.σ, data)
        
    else
        data = data'
    end
    
    if args.pre_transformation && sum(dataTypeArray) > 0
        if preprocess_ps.pre_transformation_type == "power"
            save_preprocess_results(data', x_tr_BC, x_tr_power, loss_array_power, loss_array_λ1, args.scaling, preprocess_ps, dataTypeArray, anim_list)
        else
            save_preprocess_results(data', preprocess_ps, dataTypeArray)
        end
    elseif args.scaling
        save_preprocess_results(data', preprocess_ps, dataTypeArray)
    end

    return data, preprocess_ps
end
