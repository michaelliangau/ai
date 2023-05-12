from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.real_trading = False
    config.investor_type = "growth"
    config.initial_investment = 100_000
    config.context_window_date = "2018-01-01"
    config.investment_schedule = "monthly"
    config.num_simulated_months = 48
    config.num_simulations = 10
    config.llm_additional_context = None
    config.experiment_folder_path = "output/experiments/growth_no_news_context"
    config.additional_context_dataset_path = None
    config.additional_context_sample_size = None
    config.transaction_cost = 0.0001
    return config
