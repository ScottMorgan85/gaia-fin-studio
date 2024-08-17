import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Define the path to your Excel file
excel_path = "./data/client_fact_table.xlsx"

def generate_and_save_embeddings(excel_path, model_name, index_file):
    # Check if GPU is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the Excel file
    data = pd.read_excel(excel_path, sheet_name='Sheet1')

    # Print the column names to verify them
    print("Column names in the dataset:", data.columns)

    # Preprocess the data to create text representations using all columns
    def create_text_representation(row):
        return ", ".join([f"{col}: {str(row[col])}" for col in data.columns])

    # Apply the function to create a list of text representations
    text_data = data.apply(create_text_representation, axis=1).tolist()

    # Initialize the embedding model and set it to the appropriate device
    embedding_model = SentenceTransformer(model_name, device=device)

    # Generate embeddings for the text data
    embeddings = embedding_model.encode(text_data, show_progress_bar=True)

    # Initialize a FAISS index and add the embeddings
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save the index and text data for later use
    faiss.write_index(index, index_file)
    with open(f"{index_file}_texts.txt", "w") as f:
        for item in text_data:
            f.write("%s\n" % item)

    print(f"Embeddings and index saved to {index_file} and {index_file}_texts.txt")

# Example usage with the defined excel_path
generate_and_save_embeddings(excel_path, 'all-MiniLM-L6-v2', 'client_embeddings.index')


# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import torch

# # Define the path to your Excel file
# excel_path = "./data/client_fact_table.xlsx"

# def generate_and_save_embeddings(excel_path, model_name, index_file):
#     # Check if GPU is available and set the device accordingly
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")

#     # Load the Excel file
#     data = pd.read_excel(excel_path, sheet_name='Sheet1')

#     # Print the column names to verify them
#     print("Column names in the dataset:", data.columns)

#     # Preprocess the data to create text representations
#     def create_text_representation(row):
#         # Replace 'portfolio_name' with the appropriate column, e.g., 'client_strategy'
#         return f"{row['client_name']}, {row['client_strategy']}, {row['aum']} USD, {row['benchmark']}"

#     # Apply the function to create a list of text representations
#     text_data = data.apply(create_text_representation, axis=1).tolist()

#     # Initialize the embedding model and set it to the appropriate device
#     embedding_model = SentenceTransformer(model_name, device=device)

#     # Generate embeddings for the text data
#     embeddings = embedding_model.encode(text_data, show_progress_bar=True)

#     # Initialize a FAISS index and add the embeddings
#     d = embeddings.shape[1]
#     index = faiss.IndexFlatL2(d)
#     index.add(embeddings)

#     # Save the index and text data for later use
#     faiss.write_index(index, index_file)
#     with open(f"{index_file}_texts.txt", "w") as f:
#         for item in text_data:
#             f.write("%s\n" % item)

#     print(f"Embeddings and index saved to {index_file} and {index_file}_texts.txt")

# # Example usage with the defined excel_path
# generate_and_save_embeddings(excel_path, 'all-MiniLM-L6-v2', 'client_embeddings.index')
