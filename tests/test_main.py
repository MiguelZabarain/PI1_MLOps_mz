import pytest
from fastapi.testclient import TestClient
from ..main import app

client = TestClient(app)

# Test: PlayTimeGenre endpoint #1
# Purpose: This test checks if we can get the year with most played hours for a specific game genre
# Input: A genre name like 'Action'
# Expected output: A dictionary containing the release year with most hours played for that genre
# Example response: {"Año de lanzamiento con más horas jugadas para Género Action" : "2013"}
def test_PlayTimeGenre():
    response = client.get("/PlayTimeGenre/Action")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert any("Año de lanzamiento" in key for key in response.json().keys())

# Test: UserForGenre endpoint #2
# Purpose: This test validates if we can get the user with most played hours for a specific genre
# Input: A genre name like 'Action'
# Expected output: A dictionary with user ID and list of played hours per year
# Example response: {"Usuario con más horas jugadas para Género Action": "user123", "Horas jugadas":[{"Año": 2013, "Horas": 203}]}
def test_UserForGenre():
    response = client.get("/UserForGenre/Action")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "Usuario con más horas jugadas para Género" in list(response.json().keys())[0]
    assert "Horas jugadas" in response.json()
    assert all(["Año" in item and "Horas" in item for item in response.json()["Horas jugadas"]])
    assert all(isinstance(x["Horas"], int) for x in response.json()["Horas jugadas"])
    assert all(year.isdigit() and len(year) == 4 for year in [str(item["Año"]) for item in response.json()["Horas jugadas"]])

# Test: UsersRecommend endpoint #3
# Purpose: This test checks if we can get top 3 recommended games for a specific year
# Input: A year (integer) like 2015
# Expected output: List of 3 dictionaries containing top recommended games
# Example response: [{"Puesto 1" : "Game1"}, {"Puesto 2" : "Game2"}, {"Puesto 3" : "Game3"}]
def test_UsersRecommend():
    response = client.get("/UsersRecommend/2015")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    assert all("Puesto" in list(item.keys())[0] for item in response.json())
    assert response.headers["content-type"] == "application/json" # Validates proper response headers
    assert all(isinstance(item, dict) for item in response.json()) # Validates data type consistency in response items

# Test: UsersWorstDeveloper endpoint #4
# Purpose: This test validates if we can get top 3 worst-rated developers for a specific year
# Input: A year (integer) like 2015
# Expected output: List of 3 dictionaries containing worst-rated developers
# Example response: [{"Puesto 1" : "Dev1"}, {"Puesto 2" : "Dev2"}, {"Puesto 3" : "Dev3"}]
def test_UsersWorstDeveloper():
    response = client.get("/UsersWorstDeveloper/2011")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 3
    assert all("Puesto" in list(item.keys())[0] for item in response.json())
    assert all(isinstance(item, dict) for item in response.json())
    assert response.headers["content-type"] == "application/json"

# Test: sentiment_analysis endpoint #5
# Purpose: This test checks if we can get sentiment analysis results for a specific developer
# Input: A developer name like 'Valve'
# Expected output: Dictionary with developer name and counts of sentiment categories
# Example response: {'Valve' : ['Negative = 182', 'Neutral = 120', 'Positive = 278']}
def test_sentiment_analysis():
    response = client.get("/sentiment_analysis/Valve")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert len(list(response.json().values())[0]) == 3
    assert response.json()["Valve"][0].startswith("Negative = ")

# Test: recomendacion_juego endpoint #6
# Purpose: This test validates if we can get game recommendations based on a game ID
# Input: A game ID (string)
# Expected output: List of 5 dictionaries containing recommended games
# Example response: [{"game_id1": "game_name1"}, ..., {"game_id5": "game_name5"}]
def test_recomendacion_juego():
    response = client.get("/recomendacion_juego/222621")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 5
    assert len(set(game.values() for game in response.json())) == len(response.json())

# Error Handling Tests
# These tests verify that the API properly handles invalid inputs

# Test: Invalid genre handling
def test_invalid_genre():
    with pytest.raises(Exception): 
        client.get("/PlayTimeGenre/InvalidGenre")

# Test: Invalid year handling
def test_invalid_year():
    with pytest.raises(Exception): 
        client.get("/UsersRecommend/9999")

# Test: Invalid developer handling
def test_invalid_developer():
    response = client.get("/sentiment_analysis/InvalidDeveloper")
    assert response.json() == {"InvalidDeveloper": ["Negative = 0", "Neutral = 0", "Positive = 0"]}

# Test: Invalid game ID handling
def test_invalid_game_id():
    with pytest.raises(Exception):
        client.get("/recomendacion_juego/999999999")
