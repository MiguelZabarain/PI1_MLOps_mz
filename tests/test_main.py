import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test: PlayTimeGenre (endpoint #1)
# Purpose: This test checks if we can get the year with most played hours for a specific game genre
# Input: A genre name like 'Action'
# Expected output: A dictionary containing the release year with most hours played for that genre
# Example response: {"Año de lanzamiento con más horas jugadas para Género Action" : "2013"}
def test_PlayTimeGenre():
    response = client.get("/PlayTimeGenre/Action")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), dict)  # Validates response is a dictionary
    assert any("Año de lanzamiento" in key for key in response.json().keys())  # Validates expected key format is present

# Test: UserForGenre (endpoint #2)
# Purpose: This test validates if we can get the user with most played hours for a specific genre
# Input: A genre name like 'Action'
# Expected output: A dictionary with user ID and list of played hours per year
# Example response: {"Usuario con más horas jugadas para Género Action": "user123", "Horas jugadas":[{"Año": 2013, "Horas": 203}]}
def test_UserForGenre():
    response = client.get("/UserForGenre/Action")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), dict)  # Validates response is a dictionary
    assert "Usuario con más horas jugadas para Género" in list(response.json().keys())[0]  # Validates user key is present
    assert "Horas jugadas" in response.json()  # Validates hours played key is present
    assert all(["Año" in item and "Horas" in item for item in response.json()["Horas jugadas"]])  # Validates each item has year and hours
    assert all(isinstance(x["Horas"], int) for x in response.json()["Horas jugadas"])  # Validates hours are integers
    assert all(year.isdigit() and len(year) == 4 for year in [str(item["Año"]) for item in response.json()["Horas jugadas"]])  # Validates years are 4-digit numbers

# Test: UsersRecommend (endpoint #3)
# Purpose: This test checks if we can get top 3 recommended games for a specific year
# Input: A year (integer) like 2015
# Expected output: List of 3 dictionaries containing top recommended games
# Example response: [{"Puesto 1" : "Game1"}, {"Puesto 2" : "Game2"}, {"Puesto 3" : "Game3"}]
def test_UsersRecommend():
    response = client.get("/UsersRecommend/2015")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), list)  # Validates response is a list
    assert len(response.json()) == 3  # Validates list contains exactly 3 items
    assert all("Puesto" in list(item.keys())[0] for item in response.json())  # Validates each item has position key
    assert response.headers["content-type"] == "application/json"  # Validates proper response headers
    assert all(isinstance(item, dict) for item in response.json())  # Validates data type consistency in response items

# Test: UsersWorstDeveloper (endpoint #4)
# Purpose: This test validates if we can get top 3 worst-rated developers for a specific year
# Input: A year (integer) like 2015
# Expected output: List of 3 dictionaries containing worst-rated developers
# Example response: [{"Puesto 1" : "Dev1"}, {"Puesto 2" : "Dev2"}, {"Puesto 3" : "Dev3"}]
def test_UsersWorstDeveloper():
    response = client.get("/UsersWorstDeveloper/2011")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), list)  # Validates response is a list
    assert len(response.json()) == 3  # Validates list contains exactly 3 items
    assert all("Puesto" in list(item.keys())[0] for item in response.json())  # Validates each item has position key
    assert all(isinstance(item, dict) for item in response.json())  # Validates each item is a dictionary
    assert response.headers["content-type"] == "application/json"  # Validates proper response headers

# Test: sentiment_analysis (endpoint #5)
# Purpose: This test checks if we can get sentiment analysis results for a specific developer
# Input: A developer name like 'Valve'
# Expected output: Dictionary with developer name and counts of sentiment categories
# Example response: {'Valve' : ['Negative = 182', 'Neutral = 120', 'Positive = 278']}
def test_sentiment_analysis():
    response = client.get("/sentiment_analysis/Valve")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), dict)  # Validates response is a dictionary
    assert len(list(response.json().values())[0]) == 3  # Validates sentiment has 3 categories
    assert response.json()["Valve"][0].startswith("Negative = ")  # Validates negative sentiment format

# Test: recomendacion_juego (endpoint #6)
# Purpose: This test validates if we can get game recommendations based on a game ID
# Input: A game ID (string)
# Expected output: List of 5 dictionaries containing recommended games
# Example response: [{"game_id1": "game_name1"}, ..., {"game_id5": "game_name5"}]
def test_recomendacion_juego():
    response = client.get("/recomendacion_juego/222621")
    assert response.status_code == 200  # Validates successful API response
    assert isinstance(response.json(), list)  # Validates response is a list
    assert len(response.json()) == 5  # Validates list contains exactly 5 recommendations
    assert len(set(game.values() for game in response.json())) == len(response.json())  # Validates all recommendations are unique

# ======================================================================================
# Error Handling Tests
# These tests verify that the API properly handles invalid inputs

# Test: Invalid genre handling for PlayTimeGenre (endpoint #1)
def test_PlayTimeGenre_InvalidGenre():
    with pytest.raises(Exception):  # Validates exception is raised for invalid genre
        client.get("/PlayTimeGenre/InvalidGenre")

# Test: Invalid genre handling for UserForGenre (endpoint #2)
def test_UserForGenre_InvalidGenre():
    with pytest.raises(Exception):  # Validates exception is raised for invalid genre
        client.get("/UserForGenre/InvalidGenre")

# Test: Invalid year handling for UsersRecommend (endpoint #3)
def test_UsersRecommend_InvalidYear():
    with pytest.raises(Exception):  # Validates exception is raised for non-existent year
        client.get("/UsersRecommend/9999")

# Test: Invalid year handling for UsersWorstDeveloper (endpoint #4)
def test_UsersWorstDeveloper_InvalidYear():
    with pytest.raises(Exception):  # Validates exception is raised for non-existent year
        client.get("/UsersWorstDeveloper/9999")

# Test: Invalid developer handling for sentiment_analysis (endpoint #5)
def test_sentiment_analysis_InvalidDeveloper():
    with pytest.raises(Exception):  # Validates exception is raised for invalid developer
        client.get("/sentiment_analysis/InvalidDeveloper")

# Test: Invalid game ID handling for recomendacion_juego (endpoint #6)
def test_recomendacion_juego_InvalidGameID():
    with pytest.raises(Exception):  # Validates exception is raised for invalid game ID
        client.get("/recomendacion_juego/999999999")
