from typing import Dict, Any

def add_translation(row: Dict[str, Any], translations: Dict[str, Any], key: str) -> Dict:
    """
    Add translation to each item in the dataset
    
    Args:
        row (Dict[str, Any]): A dictionary containing the example data
        translations (Dict[str, Any]): A dictionary containing the translations
        key (str): The key to use for the translation
    
    Returns:
        row: A dictionary containing the data with the translation
            added
    """
    row[key] = translations.get(row['id'], None)
    return row