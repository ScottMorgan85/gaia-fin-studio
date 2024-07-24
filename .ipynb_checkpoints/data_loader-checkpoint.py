from data.client_central_fact import get_fact_by_client_id
from data.client_returns import get_returns_by_asset

def load_client_data(client_id):
    return get_fact_by_client_id(client_id)

def load_asset_returns(asset_name):
    return get_returns_by_asset(asset_name)
