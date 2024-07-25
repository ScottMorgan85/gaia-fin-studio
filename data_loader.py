import pandas as pd
import data.client_central_fact as client_central_fact
import data.client_mapping as client_mapping

def load_strategy_returns(file_path='data/strategy_returns.xlsx'):
    df = pd.read_excel(file_path)
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    return df

def load_benchmark_returns(file_path='data/benchmark_returns.xlsx'):
    df = pd.read_excel(file_path)
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    return df


def load_client_data(client_id):
    data = client_central_fact.get_fact_by_client_id(client_id)
    client_info_dict = client_mapping.get_client_info()  # Call the function to get the dictionary
    matching_clients = [name for name, info in client_info_dict.items() if info['client_id'] == client_id]
    if matching_clients:
        client_name = matching_clients[0]
        data['client_name'] = client_name
    else:
        data['client_name'] = "Unknown Client"
    return data

def get_client_strategy_details(client_name):
    details = client_mapping.get_strategy_details(client_name)
    if details:
        print(f"Client Name: {details['client_name']}")
        print(f"Strategy Name: {details['strategy_name']}")
        print(f"Description: {details['description']}")
        print(f"Benchmark: {details['benchmark']}")
        print(f"Risk: {details['risk']}")
    else:
        print("Client not found or no details available.")
    return details