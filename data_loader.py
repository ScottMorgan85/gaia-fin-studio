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

def load_trailing_returns(client_name):
    client_info = client_mapping.get_client_info(client_name)
    if not client_info:
        return None

    client_id = client_info['client_id']
    trailing_columns = {
        'port_selected_quarter_return': 'Quarter',
        'bench_selected_quarter_return': 'Benchmark Quarter',
        'port_1_year_return': '1 Year',
        'bench_1_year_return': 'Benchmark 1 Year',
        'port_3_years_return': '3 Years',
        'bench_3_years_return': 'Benchmark 3 Years',
        'port_5_years_return': '5 Years',
        'bench_5_years_return': 'Benchmark 5 Years',
        'port_10_years_return': '10 Years',
        'bench_10_years_return': 'Benchmark 10 Years',
        'port_since_inception_return': 'Since Inception',
        'bench_since_inception_return': 'Benchmark Since Inception'
    }
    
    trailing_returns = [entry for entry in client_central_fact.fact_table if entry['client_id'] == client_id]
    if not trailing_returns:
        return None

    # Create DataFrame with portfolio returns and benchmark returns combined
    combined_data = []
    period_names = {
        'port_selected_quarter_return': 'Quarter',
        'port_1_year_return': '1 Year',
        'port_3_years_return': '3 Years',
        'port_5_years_return': '5 Years',
        'port_10_years_return': '10 Years',
        'port_since_inception_return': 'Since Inception'
    }
    
    for port_col, period in period_names.items():
        bench_col = port_col.replace('port', 'bench')
        port_value = float(trailing_returns[0][port_col])
        bench_value = float(trailing_returns[0][bench_col])
        active_value = port_value - bench_value
        combined_data.append([period, port_value, bench_value, active_value])

    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data, columns=['Period', 'Return', 'Benchmark', 'Active'])
    combined_df.set_index('Period', inplace=True)

    return combined_df