using Distributions
using StatsBase

struct QuantileTransformer
    quantiles::Vector{Float64}
    n_quantiles::Int
end

function fit_quantile_transformer(data::Vector{Float64}; n_quantiles::Int = 1000)
    sorted_data = sort(data)
    quantiles = [sorted_data[ceil(Int, i * length(data) / n_quantiles)] for i in 1:n_quantiles]
    return QuantileTransformer(quantiles, n_quantiles)
end

function quantile_transform(transformer::QuantileTransformer, data::Vector{Float64})
    n = length(data)
    transformed_data = similar(data)

    lower_bound = minimum(transformer.quantiles)
    upper_bound = maximum(transformer.quantiles)

    for i in 1:n

        # Clip data within the bounds of the quantiles
        clipped_value = max(lower_bound, min(data[i], upper_bound))

        rank = searchsortedfirst(transformer.quantiles, clipped_value) - 1
        rank = max(1, min(rank, transformer.n_quantiles))  # Ensure rank is within bounds
        uniform_value = rank / transformer.n_quantiles
        transformed_data[i] = Distributions.quantile(Normal(), uniform_value)
    end
    return transformed_data
end

function inverse_quantile_transform(transformer::QuantileTransformer, data::Vector{Float64})
    n = length(data)
    inversed_data = similar(data)
    for i in 1:n
        uniform_value = cdf(Normal(), data[i])
        quantile_index = ceil(Int, uniform_value * transformer.n_quantiles)
        quantile_index = max(1, min(quantile_index, transformer.n_quantiles))  # Ensure index is within bounds
        inversed_data[i] = transformer.quantiles[quantile_index]
    end
    return inversed_data
end




function inverse_quantile_transform(transformer_array::Array{QuantileTransformer}, x, dataTypeArray::Array)

    inverse_data = fill(0.0, size(x,1), size(x,2))
    for col = 1:size(x, 2)
        if dataTypeArray[col]
            inverse_data[:, col] = inverse_quantile_transform(transformer_array[col], x[:, col])  # Inverse transform the data
        else
            inverse_data[:, col] = x[:, col]
        end
    end


    return inverse_data
end