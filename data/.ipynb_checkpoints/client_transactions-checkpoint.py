
transactions = [
    {
        "date": 44950,
        "name": 'IBM 4.7 07/15/35 Corp, International Business Machines Corp.',
        "selectedStrategy": 'High Yield Bonds',
        "selectedClient": 'Hari Seldon',
        "direction": 'Long',
        "quantity": 24499,
        "pricePerShare": 103,
        "totalValue": 2523397,
        "transactionType": 'Buy',
        "commentary": 'Strong Q2 earnings with high consumer demand'
    },
    {
        "date": 45179,
        "name": 'AAPL 3.85 02/09/40 Corp, Apple Inc.',
        "selectedStrategy": 'High Yield Bonds',
        "selectedClient": 'Hari Seldon',
        "direction": 'Long',
        "quantity": 21393,
        "pricePerShare": 86,
        "totalValue": 1839798,
        "transactionType": 'Buy',
        "commentary": 'Increased cloud market share'
    },
    {
        "date": 44604,
        "name": 'MSFT 4.2 05/15/50 Corp, Microsoft Corp.',
        "selectedStrategy": 'High Yield Bonds',
        "selectedClient": 'Hari Seldon',
        "direction": 'Long',
        "quantity": 14933,
        "pricePerShare": 98,
        "totalValue": 1463434,
        "transactionType": 'Buy',
        "commentary": 'Expansion in new service sectors'
    },
    {
        "date": 45004,
        "name": 'JPM 3.9 11/20/30 Corp, JPMorgan Chase & Co.',
        "selectedStrategy": 'High Yield Bonds',
        "selectedClient": 'Hari Seldon',
        "direction": 'Long',
        "quantity": 11123,
        "pricePerShare": 88,
        "totalValue": 978824,
        "transactionType": 'Buy',
        "commentary": 'Successful market penetration in Europe'
    },
    {
        "date": 45105,
        "name": 'XOM 4.6 08/01/45 Corp, Exxon Mobil Corp.',
        "selectedStrategy": 'High Yield Bonds',
        "selectedClient": 'Hari Seldon',
        "direction": 'Long',
        "quantity": 43085,
        "pricePerShare": 102,
        "totalValue": 4394670,
        "transactionType": 'Buy',
        "commentary": 'Innovations in hybrid vehicles enhance market position'
    },
    # This pattern continues for all transactions...
]

def get_transaction_by_date(date):
    return [transaction for transaction in transactions if transaction['date'] == date]

# Example usage
if __name__ == '__main__':
    print(get_transaction_by_date(44950))
