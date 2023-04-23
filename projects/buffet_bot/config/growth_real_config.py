from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.real_trading = True
    config.investor_type = "value"
    config.initial_investment = 10000
    config.context_window_date = None
    config.investment_schedule = "monthly"
    config.num_simulated_months = 48
    config.num_simulations = 1
    config.llm_additional_context = None  # TODO implement
    config.experiment_folder_path = (
        "output/experiments/news_context_ss_200_filtered_growth_10_sim"
    )
    config.additional_context_dataset_path = (
        "context_data/huff_news_with_impact_scores.json"
    )
    config.additional_context_sample_size = (
        200  # Only used if llm_additional_context == "news"
    )
    config.transaction_cost = 0  # Alpaca doesn't charge transaction fees
    return config
