import pandas as pd

def write_descriptions_to_txt(file_path):
    try:
        # Reading the Excel file
        df = pd.read_excel(file_path)

        # Iterating through each row in the DataFrame
        for index, row in df.iterrows():
            # Using the 'Category' column value as the filename
            file_name = row['Category'] + '.txt'
            file_name = file_name.replace(" / ", "_").replace(' ', '_').lower()

            # Writing the 'Description' column content into a text file
            with open(f"./category_data/{file_name}", 'w') as file:
                file.write(row['Definition'])

        return "Files have been successfully created."

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
print(write_descriptions_to_txt('/Users/michael/Desktop/categories.xlsx'))
