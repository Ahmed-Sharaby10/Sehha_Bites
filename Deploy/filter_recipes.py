# filter_recipes.py

import pandas as pd

def filter_recipes(df, health_conditions, min_calories, max_calories, allergies):
    """
    Filters recipes based on health conditions, calorie limit (range), and allergies.

    Parameters:
        df (pd.DataFrame): The dataset containing recipe details.
        health_conditions (list): List of health conditions to consider (e.g., ['HeartDisease', 'Diabetes']).
        min_calories (int): The minimum allowed calorie count for a recipe.
        max_calories (int): The maximum allowed calorie count for a recipe.
        allergies (list): List of ingredients to avoid (e.g., ['nuts', 'milk']).

    Returns:
        pd.DataFrame: Filtered DataFrame with suitable recipes.
    """
    # Filter based on health conditions
    for condition in health_conditions:
        df = df[df[condition] == 0]  # Only include recipes safe for the user's conditions

    # Filter based on calorie range (both min and max)
    df = df[(df['Calories'] >= min_calories) & (df['Calories'] <= max_calories)]

    # Filter based on allergies
    if allergies:
        df = df[~df['RecipeIngredientParts'].str.contains('|'.join(allergies), case=False, na=False)]

    return df