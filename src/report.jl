using Flux: DataLoader
using Revise
using Plots
using CSV
gr()


includet("structs.jl")
includet("../visualization/visualization.jl")


function get_log_dir(save_path, tblogger_flag)

    if !tblogger_flag
        return nothing
    else
        !isdir(save_path) && mkdir(save_path)

        parent_dir = string(pwd(), "/", save_path)
        dir_list = readdir(parent_dir)

       
        runs_list = []

        if length(dir_list) > 0
            for i = 1:length(dir_list)
                
                if isdir(string(parent_dir, "/",  dir_list[i]))                     # checking, if it's directory
                    append!(runs_list, parse(Int, string(dir_list[i])[5:end]))      # print the name of a directory
                end
            end
            run_number = (sort(runs_list))[end] + 1
        else
            run_number = 1
        end

        current_run_dir = string("$(save_path)/run_$(run_number)")
        
        return current_run_dir
    end
end

function save_preprocess_results(x_st::Matrix, x_tr_BC::Matrix, x_tr_power::Matrix, loss_array_power::Array, loss_array_λ1::Array, scaling::Bool, preprocess_ps::preprocess_params, dataTypeArray::Array, anim_list::Array)

    """
    This function saves the results for when we have pretransformations
    """

    pre_transformation_dir = string(args.current_run_path, "/pre_transformation")
    mkdir(pre_transformation_dir)
    
    ##### save box-cox transformation results #####
    mkdir(string(pre_transformation_dir, "/box_cox_transformation"))
    writedlm(string(pre_transformation_dir, "/", "box_cox_transformation/box_cox_transformed_data.csv"),  x_tr_BC, ',')
    df_BC_params = DataFrame(λ2 = preprocess_ps.λ2, λ1= preprocess_ps.λ1)
    CSV.write(string(pre_transformation_dir, "/", "box_cox_transformation/box_cox_transformation_params.csv"),  df_BC_params)
    box_cox_loss_plot = loss_plot(loss_array_λ1)
    save_plot(box_cox_loss_plot, string(pre_transformation_dir, "/", "box_cox_transformation/box_cox_transformation_loss.pdf"))
    # density
    marginal_density_plots(x_tr_BC, string(pre_transformation_dir, "/box_cox_transformation"), dataTypeArray)

    ###### save power transformation results ######
    mkdir(string(pre_transformation_dir, "/power_transformation"))
    writedlm(string(pre_transformation_dir, "/power_transformation/power_transformed_data.csv"),  x_tr_power, ',')
    df_power_params = DataFrame(shift = vec(preprocess_ps.shift), power = vec(preprocess_ps.power))
    CSV.write(string(pre_transformation_dir, "/power_transformation/power_transformation_params.csv"),  df_power_params)
    
    save_struct(preprocess_ps, string(pre_transformation_dir, "/preprocess_params.bson"))

    power_loss_plot = loss_plot(loss_array_power)
    save_plot(power_loss_plot, string(pre_transformation_dir, "/power_transformation/power_transformation_loss.pdf"))


    # save animations 
    mkdir(string(pre_transformation_dir, "/power_transformation/animations"))
    for i =  1:length(anim_list)
        gif(anim_list[i], string(pre_transformation_dir, "/power_transformation/animations/anim_$(i).gif"), fps = 2)
    end

    # density
    marginal_density_plots(x_tr_power, string(pre_transformation_dir, "/power_transformation"), dataTypeArray)

    if scaling
        ###### scaling results ######
        mkdir(string(pre_transformation_dir, "/scaling"))
        writedlm(string(pre_transformation_dir, "/scaling/scaled_data.csv"),  x_st, ',')
        df_st_params = DataFrame(μ = preprocess_ps.μ, σ = preprocess_ps.σ)
        CSV.write(string(pre_transformation_dir, "/scaling/scaling_params.csv"),  df_st_params)
        # density
        marginal_density_plots(x_st, string(pre_transformation_dir, "/scaling"), dataTypeArray)
    end

end

function save_preprocess_results(x_st::Matrix, preprocess_ps::preprocess_params, dataTypeArray::Array)

    """
    This function saves the results for when we only have scaling
    """
    pre_transformation_dir = string(args.current_run_path, "/pre_transformation")
    mkdir(pre_transformation_dir)

    ###### scaling results ######
    mkdir(string(pre_transformation_dir, "/scaling"))
    writedlm(string(pre_transformation_dir, "/scaling/scaled_data.csv"),  x_st, ',')
    df_st_params = DataFrame(μ = preprocess_ps.μ, σ = preprocess_ps.σ)
    CSV.write(string(pre_transformation_dir, "/scaling/scaling_params.csv"),  df_st_params)

    # density
    marginal_density_plots(x_st, string(pre_transformation_dir, "/scaling"), dataTypeArray)

end


function save_vae_results(training_data::DataLoader, preprocessed_data, original_data::Matrix, model, preprocess_ps::preprocess_params, args::Args, loss_array_vae::Array)

    vae_dir = logdir(model.tblogger_object)

    save_model(model, string(vae_dir, "/vae.bson"))

    vae_loss_plot = loss_plot(loss_array_vae)
    save_plot(vae_loss_plot, string(vae_dir, "/vae_loss.pdf"))

    ######################### posterior sampling ###################################
    # synthetic data 
    synthetic_data_posterior, _ = VAE_output(training_data, model, args, preprocess_ps, "posterior")
    
    writedlm(string(vae_dir, "/posterior_sampling/synthetic_data_posterior.csv"),  synthetic_data_posterior, ',')

    # correlation plots
    pairwise_correlation_plot_posterior = pairwise_correlation(synthetic_data_posterior)
    Plots.savefig(string(vae_dir, "/posterior_sampling/correlation_posterior.pdf"))

    pairwise_correlation_plot_original = pairwise_correlation(original_data)
    Plots.savefig(string(vae_dir, "/posterior_sampling/correlation_original.pdf"))

    pairwise_correlation_plot_posterior_diff = pairwise_correlation_diff(original_data , synthetic_data_posterior)
    Plots.savefig(string(vae_dir, "/posterior_sampling/correlation_diff_posterior.pdf"))

    marginal_density_plots(original_data, synthetic_data_posterior, string(vae_dir, "/posterior_sampling"), model.feature_type)

    #########################   prior sampling   ###################################
    # synthetic data
    synthetic_data_prior, glm_model_death_region = VAE_output(training_data, model, args, preprocess_ps, "prior")

    writedlm(string(vae_dir, "/prior_sampling/synthetic_data_prior.csv"),  synthetic_data_prior, ',')


    # correlation plots
    pairwise_correlation_plot_prior = pairwise_correlation(synthetic_data_prior)
    Plots.savefig(string(vae_dir, "/prior_sampling/correlation_prior.pdf"))

    pairwise_correlation_plot_original = pairwise_correlation(original_data)
    Plots.savefig(string(vae_dir, "/prior_sampling/correlation_original.pdf"))

    pairwise_correlation_plot_prior_diff = pairwise_correlation_diff(original_data, synthetic_data_prior)
    Plots.savefig(string(vae_dir, "/prior_sampling/correlation_diff_prior.pdf"))

    with_logger(model.tblogger_object) do
        @info "correlation_diff" pairwise_correlation_plot_prior_diff log_step_increment=0
    end

    marginal_density_plots(original_data, synthetic_data_prior, string(vae_dir, "/prior_sampling"), model.feature_type)
    #########################   latent   ###################################
    mkdir(string(vae_dir, "/latent"))

    z = get_latent(preprocessed_data, model, args, preprocess_ps)


    if (args.data_string == "data_scenario1")

        E = load_exposure(args.data_string)

        y = load_outcome(args.data_string)

        latent_plot = scatter_latent(z, "E", E, "A) Latent representation")
        Plots.savefig(string(vae_dir, "/latent/latent_color_code_exposure.pdf"))
        with_logger(model.tblogger_object) do
            @info "latent_exposure" latent_plot log_step_increment=0
        end


        plt = scatter_latent(z, "y", y, "A) Latent representation")
        Plots.savefig(string(vae_dir, "/latent/latent_color_code_outcome.pdf"))
        with_logger(model.tblogger_object) do
            @info "latent_outcome" plt log_step_increment=0
        end
    elseif (args.data_string == "ist_new") || (args.data_string == "ist2d") || (args.data_string == "ist2d_subset") ||(args.data_string == "ist_more_features")|| (args.data_string == "ist_more_features_2") || (args.data_string == "ist_more_features_no_west") || contains(args.data_string , "ist_randomization") 
        plt = scatter_latent_ist(z, load_outcome(args.data_string))
        Plots.savefig(string(vae_dir, "/latent/latent_outcome.pdf"))
        with_logger(model.tblogger_object) do
            @info "latent" plt log_step_increment=0
        end

        plt = scatter_latent_ist(z, load_country(args.data_string))
        Plots.savefig(string(vae_dir, "/latent/latent_countries.pdf"))
        with_logger(model.tblogger_object) do
            @info "latent_country" plt log_step_increment=0
        end

        plt = scatter_latent_ist(z, load_region(args.data_string))
        Plots.savefig(string(vae_dir, "/latent/latent_regions.pdf"))
        with_logger(model.tblogger_object) do
            @info "latent_region" plt log_step_increment=0
        end
    end


    #########################   IPW sampling   ###################################
    if args.IPW_sampling

        mkdir(string(vae_dir, "/prior_sampling/IPW_sampling"))

        save_model(glm_model_death_region, string(vae_dir, "/prior_sampling/IPW_sampling/glm_model_death_region.bson"))
       
        probabilities_death_region = predict_probability_region(glm_model_death_region, preprocessed_data', args)

        REGION = (load_region(args.data_string).=="EU-NORTH")


        size(700,500)
        plt_propensity_score, plt_weights_latent = latent_propensity_ovrlay_region(z, probabilities_death_region, 2000 , REGION, "Region", args.grid_point_size)

        plt_merged = Plots.plot(plt_propensity_score, plt_weights_latent, layout = (1,2), size = (1400, 500), fontsize = 8)
        Plots.savefig(plt_merged, string(vae_dir, "/prior_sampling/IPW_sampling/propensity_ist_fig.pdf"))
        Plots.savefig(plt_merged, string(vae_dir, "/prior_sampling/IPW_sampling/propensity_ist_fig.png"))

        Plots.savefig(plt_propensity_score, string(vae_dir, "/prior_sampling/IPW_sampling/propensity_score_latent.png"))

        with_logger(model.tblogger_object) do
            @info "propensity_score_latent" plt_propensity_score log_step_increment=0
        end

        Plots.savefig(plt_weights_latent, string(vae_dir, "/prior_sampling/IPW_sampling/plt_weights_latent.png"))
        with_logger(model.tblogger_object) do
            @info "plt_weights_latent" plt_weights_latent log_step_increment=0
        end
        
        vae_output_prior = Matrix(CSV.read(string(vae_dir, "/prior_sampling/vae_output.csv"), DataFrame, header = false))'

        probabilities_death_region_syn = predict_probability_region(glm_model_death_region, vae_output_prior, args)


        Plots.histogram(probabilities_death_region, alpha =1, color = "#1D4A91", bins =30, label = "Original Data")

        hist_propensity_score = Plots.histogram!(probabilities_death_region_syn, alpha = 0.8, color = "#AE232F", bins =30, label = "Synthetic Data")

        Plots.savefig(hist_propensity_score, string(vae_dir, "/prior_sampling/IPW_sampling/hist_propensity_score.png"))

        with_logger(model.tblogger_object) do
            @info "hist_propensity_score" hist_propensity_score log_step_increment=0
        end


        Plots.density(probabilities_death_region, color = "#1D4A91", bandwidth = 0.001)

        density_propensity_score = Plots.density!(probabilities_death_region_syn, color = "#AE232F", bandwidth = 0.001)

        Plots.savefig(density_propensity_score, string(vae_dir, "/prior_sampling/IPW_sampling/density_propensity_score.png"))

        with_logger(model.tblogger_object) do
            @info "density_propensity_score" density_propensity_score log_step_increment=0
        end

        z_syn = get_latent(vae_output_prior', model, args, preprocess_ps)
        z_syn_plot = Plots.scatter(z_syn[1,:], z_syn[2,:], marker_z = probabilities_death_region_syn,   color = :balance)
        Plots.savefig(z_syn_plot, string(vae_dir, "/prior_sampling/IPW_sampling/z_syn_plot.pdf"))

        with_logger(model.tblogger_object) do
            @info "z_syn_plot" z_syn_plot log_step_increment=0
        end

        z_plot = Plots.scatter(z[1,:], z[2,:], marker_z = probabilities_death_region,  color = :balance)
        Plots.savefig(z_plot, string(vae_dir, "/prior_sampling/IPW_sampling/z_plot.pdf"))
        with_logger(model.tblogger_object) do
            @info "z_plot" z_plot log_step_increment=0
        end

        weighted_samples_Bernouli, normalized_IPW_grid, propensity_score_average, Z1, Z2 = weighted_sampling(glm_model_death_region, preprocessed_data, z, args)

        hist_second_dim, smoothed_propensity_score, sampled_new_scatterplot, layered_latent_sampling_plot = create_weighted_sampling_plots(args, weighted_samples_Bernouli, normalized_IPW_grid, Z1, Z2)

        # Plots.savefig(hist_second_dim, string(vae_dir, "/prior_sampling/IPW_sampling/hist_second_dim.pdf"))
        # with_logger(model.tblogger_object) do
        #     @info "hist_second_dim" hist_second_dim log_step_increment=0
        # end

        Plots.savefig(smoothed_propensity_score, string(vae_dir, "/prior_sampling/IPW_sampling/smoothed_propensity_score.pdf"))
        with_logger(model.tblogger_object) do
            @info "smoothed_propensity_score" smoothed_propensity_score log_step_increment=0
        end

        Plots.savefig(sampled_new_scatterplot, string(vae_dir, "/prior_sampling/IPW_sampling/sampled_new_scatterplot.pdf"))
        with_logger(model.tblogger_object) do
            @info "sampled_new_scatterplot" sampled_new_scatterplot log_step_increment=0
        end

        Plots.savefig(layered_latent_sampling_plot, string(vae_dir, "/prior_sampling/IPW_sampling/layered_latent_sampling_plot.pdf"))
        with_logger(model.tblogger_object) do
            @info "layered_latent_sampling_plot" layered_latent_sampling_plot log_step_increment=0
        end

    end

end




function save_gan_results(original_data, model::GAN, preprocess_ps::preprocess_params, args::Args, loss_array_gan::Array, gen_error::Array)

    gan_dir = string(args.current_run_path, "/gan")
    mkdir(gan_dir)

    d_loss_plot = loss_plot(loss_array_gan[1])
    save_plot(d_loss_plot, string(gan_dir, "/discriminator_loss.pdf"))

    g_loss_plot = loss_plot(loss_array_gan[2])
    save_plot(g_loss_plot, string(gan_dir, "/generator_loss.pdf"))

    gen_error_plot = loss_plot(gen_error)
    save_plot(gen_error_plot, string(gan_dir, "/generator_error.pdf"))
    
    # synthetic data
    synthetic_data = GAN_output(original_data, model, args, preprocess_ps, loss_array_gan, gen_error)


    writedlm(string(gan_dir, "/synthetic_data.csv"),  synthetic_data, ',')

    # correlation plots
    pairwise_correlation_synthetic_plot = pairwise_correlation(synthetic_data)
    Plots.savefig(string(gan_dir, "/correlation_synthetic.pdf"))

    pairwise_correlation_original_plot = pairwise_correlation(original_data)
    Plots.savefig(string(gan_dir, "/correlation_original.pdf"))


    pairwise_correlation_plot_diff = pairwise_correlation_diff(original_data, synthetic_data)
    Plots.savefig(string(gan_dir, "/correlation_diff.pdf"))
    
    # mkdir(string(gan_dir, "/pdf_diagrams"))

    marginal_density_plots(original_data, synthetic_data, gan_dir, model.feature_type)
   
end


function marginal_density_plots(original_data, synthetic_data, path, dataTypeArray)


    mkdir(string(path, "/pdf_diagrams"))

     # density 
     for i = 1:args.input_dim
        if dataTypeArray[i]
            plt = density_plot(original_data[:, i], synthetic_data[:, i], "dimension $(i)")
            Plots.savefig(string(path, "/pdf_diagrams/dimension_$(i).pdf"))
        else
            plt = histogram_plot(original_data[:, i], synthetic_data[:, i], "dimension $(i)", (0, length(original_data[:, i])))
            Plots.savefig(string(path, "/pdf_diagrams/dimension_$(i).pdf"))
        end
    end
end


function marginal_density_plots(data, path, dataTypeArray)
    """
    This function only saves the continuous features.
    """

    mkdir(string(path, "/pdf_diagrams"))

    # density
    for i = 1:args.input_dim
       if dataTypeArray[i]
           plt = Plots.density(data[:, i], title =  "dimension $(i)")
           Plots.savefig(string(path, "/pdf_diagrams/dimension_$(i).pdf"))
       end
   end
end