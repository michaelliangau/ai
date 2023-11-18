import pandas as pd
import os

def write_descriptions_to_txt(file_path):
    try:
        # Reading the CSV file
        df = pd.read_csv(file_path)

        # Ensure the 'company_data' directory exists
        os.makedirs("./company_data", exist_ok=True)

        # Iterating through each row in the DataFrame
        for index, row in df.iterrows():
            if pd.isna(row['Name']) or pd.isna(row['Description']) or pd.isna(row['VC Industry Classification']):
                continue
            
            # Using the 'Name' column value as the filename
            file_name = row['Name']
            file_name = file_name.replace(" / ", "_").replace(' ', '_').replace('.', '_').lower()
            file_name += '.txt'
            print(file_name)

            # Writing the 'Description' column content into a text file
            with open(f"./company_data/{file_name}", 'w') as file:
                file.write(row['Description'])

        return "Files have been successfully created."

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
print(write_descriptions_to_txt('/Users/michael/Desktop/test_set.csv'))
