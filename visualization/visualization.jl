using Plots
using StatsPlots
using LaTeXStrings
using CategoricalArrays
using ImageFiltering
using KernelDensity
using DataFrames
using Statistics:cor
# Set default DPI to 300
default(dpi=300)
#######################################Visualization############################################

################## scatteplot for every two variables ##################
function display_data_Gadfly(x, title)
    n = size(x)[1]
    p = size(x)[2]
    for i =1:p
        for j =i+1:p
            println("x$i, x$j")
            display(Gadfly.plot(x= x[:,i], y = x[:,j], Scale.x_continuous(minvalue=-5, maxvalue=5), Scale.y_continuous(minvalue=-5, maxvalue=5), Theme(point_size=4pt), Guide.title(title)))
        end
    end
end

################## 2-layer one-dimension density diagram from original and synthetic data ##################
function density_plot(x1, x2, title)
    Plots.density(x1, label = "original", linewidth= 3)
    return Plots.density!(x2, label = "synthetic", linewidth= 3, title = title)
end

################## 2-layer one-dimension histogram from original and synthetic data ##################
function histogram_plot(x1, x2, title, y_lim)

    df = DataFrame(data = vcat(x1, x2), type = vcat(fill("synthetic", length(x2)), fill("original", length(x1))))

    @df df groupedhist(:data, group = :type, bar_position = :dodge, ylim= y_lim)
    # Plots.histogram(x1, label = "original", fillalpha = 0.3, bins = 2, bar_width = 1, xlim = (-1,2))
    # Plots.histogram!(x2, label = "synthetic", title = title, fillalpha = 0.3, bins = 2, bar_width = 0.5, xlim = (-1,2))
end



################## x-layer one-dimension histogram from original and synthetic data of different methods ##################

function histogram_plot(title, data_arrays::Array, labels::Array, color_array::Array, y_lim::Tuple, yformatter_flag =true)
    # Check if the number of data arrays matches the number of labels
    if length(data_arrays) != length(labels)
        error("The number of data arrays must match the number of labels.")
    end

    # Create a mapping from labels to colors
    


    color_idx_list = collect(1:length(data_arrays))

    # Create a DataFrame to hold the data and group labels
    df = DataFrame(data = Float64[], type = String[], color = String[])

    # Iterate over each input array and add it to the DataFrame
    for (x, label, color) in zip(data_arrays, labels, color_array)
        append!(df, DataFrame(data = x, type = fill(label, length(x)), color = fill(color, length(x))))
    end
    
    desired_order = labels
    # df.type = CategoricalArrays.categorical(df.type, ordered=true, categories=desired_order)


    # Convert the 'Column' to a categorical variable
    df.type = categorical(df.type)

    # Set the categories and their order
    levels!(df.type, desired_order)

    # Plot the grouped histogram
    if yformatter_flag
        @df df groupedhist(:data, group = :type, color = color_array, bar_position = :dodge, title = title, bar_width = 0.5, xticks = [0,1], size = (200, 300), ylim = y_lim)
    else
        @df df groupedhist(:data, group = :type, color = color_array, bar_position = :dodge, title = title, bar_width = 0.5, xticks = [0,1], size = (200, 300), ylim = y_lim, yformatter=_->"")
    end
end
################## optimization loss plot ##################
function loss_plot(loss_array)
    plt = Gadfly.plot(x =[Float64(i) for i in 1:length(loss_array)], y = loss_array,  Geom.line, style(line_width=1mm))
    # display(plt)
    return plt
end


################## save plot function #####################
# this function only works for Gadfly plots. For Plots package generated plots simpy use savefig

function save_plot(plt, title)
    img = PNG(title , 50cm, 50cm)
    Gadfly.draw(img, plt)
end

################## scatterplot matrix ##################
function scatterplot_matrix(x, title)
    (nobs, nvars) = size(x)
    (fig, ax) = subplots(nvars, nvars, figsize=(8,8))
    subplots_adjust(hspace=0.05, wspace=0.05)
    # Plot data
    for i = 1:nvars
        for j = 1:nvars
            if j > i
                ax[i,j][:plot](x[:,j],x[:,i], markersize = 0.2,"ob",mfc="none")
            elseif i==j
                ax[i,j][:hist](x[:,i], bins=50)
            end
            ax[i,j][:xaxis][:set_visible](false)
            ax[i,j][:yaxis][:set_visible](false)
        end
    end

    # Set tick positions
    for i = 1:nvars
        ax[i,1][:yaxis][:set_ticks_position]("left")
        ax[i,end][:yaxis][:set_ticks_position]("right")
        ax[1,i][:xaxis][:set_ticks_position]("top")
        ax[end,i][:xaxis][:set_ticks_position]("bottom")
    end

    # Turn ticks on
    cc = repeat([nvars, 1],floor(Int, ceil(nvars/2)))
    for i = 1:nvars
        ax[i,cc[i]][:yaxis][:set_visible](true)
        ax[cc[i],i][:xaxis][:set_visible](true)
        # println(typeof(fig))
    end
    fig.suptitle(title, fontsize=16)
    # display(fig)
    return
end


################## histogram for all variables ##################
function histogram_matrix(x, x_retr, title)
    (nobs, nvars) = size(x)

    column = floor(Int, sqrt(nvars))
 
    (fig, ax) = subplots(ceil(Int64,(nvars/column)), column, figsize=(8,8))
    subplots_adjust(hspace=0.05, wspace=0.05)

    for i = 1:ceil(Int64,(nvars/column))
        for j = 1:column
            if (i-1)*column +j <= p

                if dataTypeArray[((i-1)*column +j)] == "Binary"
                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 4, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 4, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
                else
                    ax[i,j][:hist](x[:,((i-1)*column+j)],  bins = 50, color = "#0f87bf", alpha = 0.5, label = "Original Data")#, "ob",mfc="none",
                    ax[i,j][:hist](x_retr[:,((i-1)*column+j)], bins = 50, color = "#ed1010", alpha = 0.5, label = "Our Method") #,"ob",mfc="none",
                end

                ax[i,j][:xaxis][:set_visible](false)
                ax[i,j][:yaxis][:set_visible](false)
                
            else
                ax[i,j][:xaxis][:set_visible](false)
                ax[i,j][:yaxis][:set_visible](false)
            end
        end
    end

    # Set tick positions
    for i = 1:ceil(Int64,(nvars/column))
        ax[i,1][:yaxis][:set_ticks_position]("left")
        ax[i,end][:yaxis][:set_ticks_position]("right")
    end

    for i=1:column
        ax[1,i][:xaxis][:set_ticks_position]("top")
        ax[end,i][:xaxis][:set_ticks_position]("bottom")
    end

    # red_patch = mpatches.Patch(color="#0f87bf", label='Our Method')
    # blue_patch = mpatches.Patch(color="#0f87bf", label='The Original Data')
    # green_patch = mpatches.Patch(color="#0f87bf", label='Gaussian Copula')
    fig.legend(["The Original Data", "Our Method"], loc="lower right")
    
    fig.suptitle(title, fontsize=16)
    display(fig)
end


################## density diagram for all variables ##################

function histogram_all_dimensions(x, syn, title)
    set_default_plot_size(50cm, 50cm)
    rowSize = floor(Int64, p/3) + 1

    plotArray = fill(Gadfly.plot(), rowSize ,3)
    cnt = 1

    for i = 1: rowSize
        if cnt>p
            break
        end
        for j = 1: 3
            if cnt>p
                break
            end
            plt = Gadfly.plot(
            layer( x =x[:,cnt], color=[colorant"black"], Theme(line_width=0.5mm, line_style=[:dot]), Geom.density(bandwidth = 0.6)), #(bandwidth = 0.2)
            layer(x= syn[:,cnt],  color=[colorant"red"], Theme(line_width=0.5mm,line_style=[:dot]), Geom.density(bandwidth = 0.6)),
            Guide.manual_color_key("Legend",["Original Data", "Synthetic Data"], [colorant"black", colorant"red"]),  Guide.xlabel("Dimension $cnt "))
            plotArray[i, j] = plt
            cnt+=1
        end
    end   
    Gadfly.title(gridstack(plotArray), title)
end


using CairoMakie




function pairwise_correlation(x)
    """
    This function plots the correlation matrix
        NOTE: the first indexes are below left of the diagram. 
        1.0   0.2   0.3 
        0.2   1.0   0.4
        0.3   0.4   1.0

        in the plot is shown like:
        
        0.3   0.2   1.0  
        0.2   1.0   0.4
        1.0   0.4   0.3
    """

    return Plots.heatmap(cor(x), clim = (-1,1))
end



function pairwise_correlation_diff(x::Matrix, synthetic_data::Matrix)
    cs = cgrad([:white, palette(:Reds)[8],  palette(:Reds)[end]])
    return Plots.heatmap(abs.(cor(x) .- cor(synthetic_data)), clim = (0, 1), color = cs)
end

function pairwise_correlation_diff(x::DataFrame, synthetic_data::DataFrame)
    cs = cgrad([:white, palette(:Reds)[8],  palette(:Reds)[end]])
    return Plots.heatmap(abs.(cor(Matrix(x)) .- cor(Matrix(synthetic_data))), clim = (0, 1), color = cs)
end

function pairwise_correlation_diff(x, synthetic_data)
    cs = cgrad([:white, palette(:Reds)[8],  palette(:Reds)[end]])
    return Plots.heatmap(abs.(cor(Matrix(x)) .- cor(Matrix(synthetic_data))), clim = (0, 1), color = cs)
end


function contour_subplots(x, dataTypeArray)

    indices = findall(x->x!=0, dataTypeArray)

    plot_list = []

    for i = 1:length(indices)
        for j = 1:i-1

            Random.seed!(42)
            f = kde(hcat(x[:, indices[i]], x[:, indices[j]]), npoints = (10000, 10000))
            X = f.x
            Y = f.y
            Z = f.density # Note x-y "for" ordering

            plt = Plots.contourf(X, Y, Z, color=:viridis, title = "$(indices[i]) and $(indices[j])")

            push!(plot_list, plt)
        end
    end
    

    # return Plots.plot(plot_list[1:2]..., layout = (1, 2), size=(1500,800))
    return Plots.plot(plot_list[1], layout = 1, size=(1500,800))
end



function contour(x, syn, fed, dim1, dim2)

    Random.seed!(42)
    f = kde(hcat(x[:, dim1], x[:, dim2]), npoints = (10000, 10000))
    X = f.x
    Y = f.y
    Z = f.density # Note x-y "for" ordering

    plt_x = Plots.contourf(X, Y, Z, color=:viridis, title = "Original Data" )#, ylabel="dimension$(dim1)", xlabel="dimension$(dim2)", fontsize = 10)


    Random.seed!(42)
    f = kde(hcat(syn[:, dim1], syn[:, dim2]), npoints = (10000, 10000))
    X = f.x
    Y = f.y
    Z = f.density # Note x-y "for" ordering

    plt_syn = Plots.contourf(X, Y, Z, color=:viridis, title = "PTVAE")#, ylabel="dimension$(dim1)", xlabel="dimension$(dim2)", fontsize = 10)



    Random.seed!(42)
    f = kde(hcat(fed[:, dim1], fed[:, dim2]), npoints = (10000, 10000))
    X = f.x
    Y = f.y
    Z = f.density # Note x-y "for" ordering

    plt_fed = Plots.contourf(X, Y, Z, color=:viridis, title = "Norta-j")#, ylabel="dimension$(dim1)", xlabel="dimension$(dim2)", fontsize = 10)


    plot_list = [plt_x, plt_syn, plt_fed]

    return Plots.plot(plot_list..., layout = (1, 3) , size=(1800,400))
end


# red: #AE232F
# blue: #1D4A91

function scatter_latent(z, colorcode_str, colorcode, title)

    Plots.scatter(z[1,colorcode.==0],z[2,colorcode.==0], markershape = :circle, markersize = 3,markerstrokewidth=0.5,color = colorant"#1D4A91" , markerstrokecolor = :white, label = "$(colorcode_str) = 0", alpha = 0.6 )
    Plots.scatter!([0],[0], markershape = :circle, markersize = 1,markerstrokewidth=0,color = :white , label = " " )
    plt = Plots.scatter!(z[1,colorcode.==1],z[2,colorcode.==1], markershape = :dtriangle, markersize = 4, markerstrokewidth=0.5, color = colorant"#AE232F", markerstrokecolor = :white,label = "$(colorcode_str) = 1", title = title, dpi = 300  )
    
end


function scatter_latent_ist(z, colorcode)

    color_dict = Dict()

    unique_colors = unique(colorcode)

    for i in 1:length(unique_colors)
        color_dict[unique_colors[i]] = i
    end

    tab_val = length(unique_colors)<10 ? 10 : 20

    clr = Plots.palette(Meta.parse("tab$(tab_val)"))[1:length(unique_colors)]


    if size(z,1) >2
        Plots.scatter(z[1,:],z[2,:],z[3,:], group = colorcode , markercolor = clr[map(x->color_dict[x], colorcode)], alpha = 0.9, markerstrokewidth=0, legend = true, dpi = 300)
    else

        Plots.scatter(z[1,:],z[2,:], group = colorcode ,markercolor = clr[map(x->color_dict[x], colorcode)], alpha = 0.9, markerstrokewidth=0, legend = true, dpi = 300)
    end
end




function get_sample_for_plotting(sample_size, z, probabilities, E, y)
    Random.seed!(42)
    max_int = size(z, 2)
    rand_ints = sample(1:max_int, sample_size, replace=false)
    z_sample = z[:, rand_ints]
    E_sample = E[rand_ints]
    y_sample = y[rand_ints]
    probabilities_sample = probabilities[rand_ints]

    return z_sample, Int.(E_sample), Int.(y_sample), probabilities_sample
end

function get_sample_for_plotting_label(sample_size, z, probabilities, REGION)
    Random.seed!(42)
    max_int = size(z, 2)
    rand_ints = sample(1:max_int, sample_size, replace=false)
    z_sample = z[:, rand_ints]
    REGION_sample = REGION[rand_ints]
    probabilities_sample = probabilities[rand_ints]

    return z_sample, Int.(REGION_sample),  probabilities_sample
end


using Colors, ColorSchemes

function get_color(prob)
    mygrays = ColorScheme([RGB{Float64}(i, i, i) for i in 0:0.1:1.0])

    color = mygrays[Int(floor(prob * 20)+1)]
    return color
end


function get_color(prob, step)
    mygrays = ColorScheme([RGB{Float64}(i, i, i) for i in 0:step:1.0])

    color = mygrays[Int.(floor.(prob .* 1/step).+1)]
    return color
end



function scatter_latent_glm(z, E, y, probabilities, sample_size)

    z_sample, E_sample, y_sample, probabilities_sample = get_sample_for_plotting(sample_size, z, probabilities, E, y)

    Plots.scatter(z_sample[1,E_sample.==0],z_sample[2,E_sample.==0], markershape = :dtriangle, markersize = 4,markerstrokewidth= 0, c = get_color.(probabilities_sample[E_sample.==0]) , label = "E = 0" )
    plt = Plots.scatter!(z_sample[1,E_sample.==1],z_sample[2,E_sample.==1], markershape = :circle, markersize = 3, markerstrokewidth= 0, c = get_color.(probabilities_sample[E_sample.==1]), label = "E = 1" )
    display(plt)
end

function grid_for_heatmap(z, grid_point_size, probabilities)

    """
        This function creates the grid we want to show behind the points in the latent sapce that the colors are for representative for the propensity score.
        z: latent space
        grid_point_size: grid_point size for the grid
        probabilities: propensity score

    """

    max1 = maximum(z[1,:])
    max2 = maximum(z[2,:])
    min1 = minimum(z[1,:])
    min2 = minimum(z[2,:])

    max = maximum([max1, max2])
    min = minimum([min1, min2])

    grid_dict = Dict()

    grid_count = Int(ceil((max-min)/grid_point_size))

    #coloring trick
    colortrick_max = 0
    colortrick_min = 1

    for i = 1:grid_count
        for j = 1:grid_count
            grid_dict[(i, j)] = [-Inf]
        end
    end

    for i=1:size(z,2)
        # @show (Int(ceil((z[1,i] - min1)/grid_point_size)) , Int(ceil((z[2,i] - min2)/grid_point_size)))
        first_index = Int(ceil((z[1,i] - min)/grid_point_size))
        first_index = first_index==0 ? 1 : first_index
        second_index =  Int(ceil((z[2,i] - min)/grid_point_size))
        second_index = second_index==0 ? 1 : second_index


        # append!(grid_dict[(first_index[i], second_index[i])], probabilities[i])
        if probabilities[i] > colortrick_max
            colortrick_max = probabilities[i]
        end

        if probabilities[i] < colortrick_min
            colortrick_min = probabilities[i]
        end

        append!(grid_dict[(first_index, second_index)], probabilities[i])
    end

    # colormat = fill(0.0, grid_count1, grid_count2)
    colormat = fill(0.0, grid_count, grid_count)
    for key in keys(grid_dict)
        if length(grid_dict[key]) ==1 
            
            colormat[key[1], key[2]] = mean(grid_dict[key])
        else
            colormat[key[1], key[2]] = mean(grid_dict[key][2:end]) #removing -5
        end
        
    end


    X = collect(min:grid_point_size:max) .+ grid_point_size/2

    Y = collect(min:grid_point_size:max) .+ grid_point_size/2

 
    colortrick_max =  maximum(colormat)

    # colortrick_min = partialsort(vec(colormat), 2)

    # # to have the empty area white
    propensity_score_average = copy(colormat)
    colormat[colormat .== -Inf] .= 0.5#(colortrick_max + colortrick_min) /2
    return X, Y, colormat', propensity_score_average  
end



function latent_propensity_ovrlay(z, probabilities, sample_size, E, y, title, legend_flag, grid_point_size)

    z_sample, E_sample, y_sample, probabilities_sample = get_sample_for_plotting(sample_size, z, probabilities, E, y)

    colorcode_str = "E"

    cs = cgrad([colorant"#1D4A91", colorant"white",  colorant"#AE232F"])
    X, Y, colormat, propensity_score_average  = grid_for_heatmap(z, grid_point_size, probabilities)


    Plots.heatmap(X, Y, colormat, color = cs, alpha=0.3, clim = (0,1))

    Plots.scatter!(z_sample[1,(E_sample.==0)],z_sample[2,(E_sample.==0) ],  markerstrokewidth = 0,markershape = :circle, msc = :white, markersize = 3,color = colorant"#1D4A91"  , label = "E=0")
    Plots.scatter!([0],[0], label=" ", ms=0, mc=:white, msc=:white)
    plt2 = Plots.scatter!(z_sample[1,(E_sample.==1) ],z_sample[2,(E_sample.==1) ],  markerstrokewidth = 0, msc = :white, markershape = :dtriangle, markersize = 4,color = colorant"#AE232F", label = "E=1",  xlims = (minimum(z_sample[1,:]) - grid_point_size, maximum(z_sample[1,:])+ grid_point_size), ylims = (minimum(z_sample[2,:]) - grid_point_size, maximum(z_sample[2,:])+ grid_point_size),  fontsize = 8, legendfontsize=8, font= "Helvetica", title = title, legend = legend_flag)


    

    return plt2
end



function latent_propensity_ovrlay_no_outcome(z, probabilities, sample_size, E, y, title, legend_flag, grid_point_size)

    z_sample, E_sample, y_sample, probabilities_sample = get_sample_for_plotting(sample_size, z, probabilities, E, y)

    colorcode_str = "E"

    # cs = cgrad([colorant"#77a0d5", colorant"white",  colorant"#96295c"])
    cs = cgrad([colorant"#1D4A91", colorant"white",  colorant"#AE232F"])
    X, Y, colormat, propensity_score_average = grid_for_heatmap(z, grid_point_size, probabilities)

    Plots.heatmap(X, Y,  colormat, color = cs, alpha=0.3, clim = (0,1))
    Plots.scatter!(z_sample[1,(E_sample.==0)],z_sample[2,(E_sample.==0) ],  markerstrokewidth = 0,msc = :white, markershape = :circle, markersize = 3,color = colorant"#1D4A91"  , label = "E=0")
    Plots.scatter!([0],[0], label=" ", ms=0, mc=:white, msc=:white)
    plt2 = Plots.scatter!(z_sample[1,(E_sample.==1) ],z_sample[2,(E_sample.==1) ],  markerstrokewidth = 0, msc = :white, markershape = :dtriangle, markersize = 4,color = colorant"#AE232F", label = "E=1",  xlims = (minimum(z_sample[1,:]) - grid_point_size, maximum(z_sample[1,:])+ grid_point_size), ylims = (minimum(z_sample[2,:]) - grid_point_size, maximum(z_sample[2,:])+ grid_point_size),  fontsize = 8, legendfontsize=8, font= "Helvetica", title = title, legend = legend_flag)
    

    return plt2
end


function latent_propensity_ovrlay_region(z, probabilities, sample_size, REGION, title, grid_point_size)

    z_sample, REGION_sample, probabilities_sample = get_sample_for_plotting_label(sample_size, z, probabilities, REGION)

    colorcode_str = title

    #AE232F

    cs = cgrad([colorant"#1D4A91", colorant"white",  colorant"#AE232F"])

    X, Y, colormat, propensity_score_average = grid_for_heatmap(z, grid_point_size, probabilities)
    blur = imfilter(colormat, Kernel.gaussian(0.5))

    Plots.heatmap(X, Y, colormat, color = cs, alpha=0.3, clim = (0,1))
    Plots.scatter!(z_sample[1,(REGION_sample.==0)],z_sample[2,(REGION_sample.==0)],  markerstrokewidth = 0, msc = :white, markershape = :circle, markersize = 3,color = colorant"#1D4A91" , label = "REGION = EU-EAST")
    Plots.scatter!([0],[0], label=" ", ms=0, mc=:white, msc=:white)
    plt_propensity_score = Plots.scatter!(z_sample[1,(REGION_sample.==1)],z_sample[2,(REGION_sample.==1)],msc = :white, markerstrokewidth = 0, markershape = :circle, markersize = 3, color = colorant"#AE232F", label =  "REGION = EU-NORTH",  xlims = (minimum(z_sample[1,:]) - grid_point_size, maximum(z_sample[1,:])+ grid_point_size), ylims = (minimum(z_sample[2,:]) - grid_point_size, maximum(z_sample[2,:])+ grid_point_size),  fontsize = 8, legendfontsize=6, font= "Helvetica", title = string("A) Latent representation and propensity score"))
    
    
    colormat_IPW = normalize_IPW(compute_IPW.(propensity_score_average)) .* length(probabilities)

    
    
    colormat_IPW[colormat_IPW .== 0] .= (maximum(colormat_IPW) + partialsort(unique(vec(colormat_IPW)), 2)) / 2
    


    Plots.heatmap(X, Y, colormat_IPW', color = cs, alpha=0.3)
    Plots.scatter!(z_sample[1,(REGION_sample.==0)],z_sample[2,(REGION_sample.==0)],  markerstrokewidth = 0, msc = :white, markershape = :circle, markersize = 3,color = colorant"#1D4A91" , label = "REGION = EU-EAST")
    Plots.scatter!([0],[0], label=" ", ms=0, mc=:white, msc=:white)
    plt_weights = Plots.scatter!(z_sample[1,(REGION_sample.==1)],z_sample[2,(REGION_sample.==1)], markerstrokewidth = 0, msc = :white, markershape = :circle, markersize = 3, color = colorant"#AE232F", label =  "REGION = EU-NORTH",  xlims = (minimum(z_sample[1,:]) - grid_point_size, maximum(z_sample[1,:])+ grid_point_size), ylims = (minimum(z_sample[2,:]) - grid_point_size, maximum(z_sample[2,:])+ grid_point_size), fontsize = 8, legendfontsize=6, font= "Helvetica", title = string("B) Latent representation and sampling weights"))
    

    return plt_propensity_score, plt_weights
end




function latent_propensity_ovrlay_death(z, probabilities, sample_size, label, title, grid_point_size)

    z_sample, label_sample, probabilities_sample = get_sample_for_plotting_label(sample_size, z, probabilities, label)

    colorcode_str = title

    #AE232F

    cs = cgrad([colorant"#1D4A91", colorant"white",  colorant"#AE232F"])

    X, Y, colormat, propensity_score_average = grid_for_heatmap(z, grid_point_size, probabilities)

    blur = imfilter(colormat, Kernel.gaussian(0.5))
    
    Plots.heatmap(X, Y, colormat, color = cs, alpha=0.3)
    Plots.scatter!(z_sample[1,(label_sample.==0)],z_sample[2,(label_sample.==0)],  markerstrokewidth = 0,markershape = :circle, markersize = 2,color = colorant"#1D4A91" , label = "FDEAD = 0")
    Plots.scatter!([0],[0], label=" ", ms=0, mc=:white, msc=:white)
    plt1 = Plots.scatter!(z_sample[1,(label_sample.==1)],z_sample[2,(label_sample.==1)], markerstrokewidth = 0, markershape = :circle, markersize = 2, color = colorant"#AE232F", label =  "FDEAD = 1",  xlims = (minimum(X) - grid_point_size, maximum(X)+ grid_point_size), ylims = (minimum(Y) - grid_point_size, maximum(Y)+ grid_point_size), ylabel="Z2", xlabel="Z1", fontsize = 10, legendfontsize=8, font= "Helvetica", title =title)

    
end




################## PTVAE paper plots ##################


function contour(x, ptvae, fed, gan, qvae, vae, dim_array)


    plt_list = []
    for i = 1:length(dim_array)
        Plots.density(x[:, dim_array[i]], color=:black, label = "Original")
        Plots.density!(ptvae[:, dim_array[i]], color=:red, label = "PTVAE" )
        Plots.density!(gan[:, dim_array[i]], color=:blue, label = "GAN" )
        Plots.density!(qvae[:, dim_array[i]], color=:pink, label = "QVAE" )
        Plots.density!(fed[:, dim_array[i]], color=:green, label = "Norta-j" )
        plt = Plots.density!(vae[:, dim_array[i]], color=:purple, label = "VAE" )
        push!(plt, plot_list)
    end

  
    return Plots.plot(plt_list..., layout = (2, length(dim_array)/2) , size=(1800,400))
end



function region_contour(z, region_string)
    regions = load_region(args.data_string)

    region_indexes = findall(x->x==region_string, regions)
    
    
    Random.seed!(42)
    f = kde(hcat(z[1, region_indexes], z[2, region_indexes]))
    X = f.x
    Y = f.y
    Z = f.density
    
    
    Plots.contourf(X, Y, Z', color=:viridis, title = region_string, colorbar = false)

end


function region_scatter(z, region_string)
    regions = load_region(args.data_string)

    region_indexes = findall(x->x==region_string, regions)
    
    
    Plots.scatter(z[1, region_indexes], z[2, region_indexes], markerstrokewidth=0,  markersize = 2, color = :indigo, title = region_string, xlim = (2.2, 4.8),  ylim = (-2.8, -1))

end



function region_contour_scatter(args)

    regions = load_region(args.data_string)
    unique_regions = unique(regions)

    # mkdir("runs/run_222/vae/latent/region_contours")
    gr()
    plot_list = []
    for reg in unique_regions
        push!(plot_list, region_contour(z, reg))
        
        push!(plot_list, region_scatter(z, reg))
    end

    Plots.plot(plot_list..., layout=grid(11, 2, widths = [0.5,0.5]), size=(1200,3000), thickness_scaling = 0.6, left_margin = 50mm, bottom_margin = 10mm)
    # savefig("runs/run_222/vae/latent/subplots.pdf")
end




function create_weighted_sampling_plots(args, weighted_samples_Bernouli, normalized_IPW_grid, Z1, Z2)
    num_samples =  size(weighted_samples_Bernouli, 2)

    Plots.histogram(rand(Normal(0,1), num_samples),  markerstrokewidth = 5, color = "#1D4A91", bins =15, label = "Original Data")
    hist_second_dim = Plots.histogram!(weighted_samples_Bernouli[2,:], markerstrokewidth = 5, color = "#AE232F", bins =15, alpha = 0.8, label = "Synthetic Data")

    smoothed_propensity_score = Plots.heatmap(normalized_IPW_grid', color = :balance)

    Plots.scatter(weighted_samples_Bernouli[1,:], weighted_samples_Bernouli[2,:], color = :black)
    sampled_new_scatterplot = Plots.heatmap!(Z1, Z2, normalized_IPW_grid', alpha = 0.5, color = :balance)

    Random.seed!(args.seed)

    num_samples = size(weighted_samples_Bernouli, 2)
    truesig = fill(0.0, 2, 2)
    truesig[diagind(truesig)] .= 1.0
    truemu = fill(0.0, 2)
    rand(Distributions.MvNormal(truemu,truesig), )
    Plots.scatter(rand(Normal(0,1), num_samples), rand(Normal(0,1), num_samples), color = "#1D4A91", label = "Samples form N(0,1)")
    layered_latent_sampling_plot = Plots.scatter!(weighted_samples_Bernouli[1,:], weighted_samples_Bernouli[2,:], color = "#AE232F", label = "Weighted Samples form N(0,1)")

    return hist_second_dim, smoothed_propensity_score, sampled_new_scatterplot, layered_latent_sampling_plot

end


function rectangle_from_coords(xb, yb, xt, yt)
    [
        xb  yb
        xt  yb
        xt  yt
        xb  yt
        xb  yb
        NaN NaN
    ]
end