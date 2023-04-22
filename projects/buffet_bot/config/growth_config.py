from ml_collections import ConfigDict

def get_config():
    config = ConfigDict()

    config.investor_type = "growth"
    config.initial_investment = 100_000
    config.context_window_date = "2018-01-01"
    config.investment_schedule = "monthly"
    config.num_simulated_months = 48
    config.num_simulations = 3
    config.llm_additional_context = "news"
    config.experiment_folder_path = "output/experiments/news_context_ss_200_filtered_growth"
    config.additional_context_dataset_path = "context_data/huff_news_with_impact_scores.json"
    config.additional_context_sample_size = 200  # Only used if llm_additional_context == "news"
    config.transaction_cost = 0.0001  # TODO implement
    return config
