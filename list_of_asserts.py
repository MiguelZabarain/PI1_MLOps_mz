################################
# API FUNCTION TESTS
################################
# These tests validate endpoint responses, formats and basic API functionality

#1. Status Code Validation
# When making an HTTP request to any endpoint (like /PlayTimeGenre/Action)
# We expect a successful response with status code 200
# This validates the API is responding correctly and the endpoint exists
assert response.status_code == 200

#2. Response Type Validation
# For endpoints that return complex data (like /UserForGenre/Action)
# The response should be a dictionary containing user and hours data
# This ensures the response structure matches our API contract
# Example: {"Usuario con más horas jugadas para Género Action": "user123", "Horas jugadas": [...]}
assert isinstance(response.json(), dict)

#3. List Length Validation
# When requesting top 3 rankings (like in /UsersRecommend/2015)
# The response must contain exactly 3 items (no more, no less)
# This validates the ranking logic is working as designed
# Example: [{"Puesto 1": "Game1"}, {"Puesto 2": "Game2"}, {"Puesto 3": "Game3"}]
assert len(response.json()) == 3

#4. Key Existence in Dictionary
# In responses containing playtime data (like /UserForGenre endpoint)
# The "Horas jugadas" key must exist to show gaming statistics
# This ensures the response contains the required playtime information
# Example: {"Horas jugadas": [{"Año": 2013, "Horas": 203}]}
assert "Horas jugadas" in response.json()

#5. String Content Validation
# For genre-based queries (like /PlayTimeGenre/Action)
# The response key must contain "Año de lanzamiento" text
# This validates the correct response format for year-based queries
# Example: {"Año de lanzamiento con más horas jugadas para Género Action": "2013"}
assert "Año de lanzamiento" in key

#6. List Content Structure
# In ranking responses (like /UsersRecommend or /UsersWorstDeveloper)
# Each item must have "Puesto" in its key to indicate ranking position
# This ensures proper ranking structure in the response
# Example: [{"Puesto 1": "Value"}, {"Puesto 2": "Value"}]
assert all("Puesto" in list(item.keys())[0] for item in response.json())

#7. Response Content Format
# For ranking endpoints returning top 3 positions
# Response must have exactly three ranked positions
# This validates complete ranking information
# Example: {"Puesto 1": "Value", "Puesto 2": "Value", "Puesto 3": "Value"}
assert response.json() == {"Puesto 1": str, "Puesto 2": str, "Puesto 3": str}

#8. Non-Empty Response
# For all endpoints returning data
# Response cannot be an empty dictionary
# This ensures meaningful data is always returned
# Example: response.json() contains at least one key-value pair
assert response.json() != {}

#9. Response Headers
# For all API responses
# Content type must be application/json
# This ensures proper API response format
# Example: response.headers["content-type"] = "application/json"
assert response.headers["content-type"] == "application/json"

################################
# ML MODEL TESTS
################################
# These tests validate the recommendation system and sentiment analysis

#10. List Item Count for Recommendations
# For game recommendations endpoint (/recomendacion_juego/id)
# Must return exactly 5 similar games based on cosine similarity
# This validates the recommendation system returns complete results
# Example: [{"game_id1": "name1"}, ..., {"game_id5": "name5"}]
assert len(response.json()) == 5

#11. Specific Value in Response
# For sentiment analysis responses
# First category must start with "Negative = "
# This validates correct sentiment category format
# Example: ["Negative = 182", "Neutral = 120", "Positive = 278"]
assert response.json()["Valve"][0].startswith("Negative = ")

#12. Optional Values
# For developer information in game data
# Developer field exists only when has_developer is true
# This validates conditional data presence
# Example: "developer" present when has_developer = True
assert "developer" in response.json() if response.json()["has_developer"] else True

#13. Value Uniqueness
# For game recommendations
# All recommended games must be unique
# This prevents duplicate recommendations
# Example: 5 different games in recommendations list
assert len(set(game.values() for game in response.json())) == len(response.json())

#14. Numerical Range Constraints
# For sentiment analysis values
# Sentiment scores must be 0, 1, or 2
# This validates correct sentiment classification
# Example: All sentiment values in [0,1,2]
assert all(0 <= int(x.split("=")[1]) <= 2 for x in response.json()["sentiment_analysis"])

################################
# INTEGRATION TESTS
################################
# These tests verify different components work together correctly

#15. Dictionary Key Position
# In genre-based user statistics (/UserForGenre)
# The first key must contain user identification text
# This ensures proper response structure for user data
# Example: {"Usuario con más horas jugadas para Género X": "user_id", ...}
assert "Usuario con más horas jugadas para Género" in list(response.json().keys())[0]

#16. Nested Dictionary Structure
# For responses with year and hours data
# Each item must contain both "Año" and "Horas" keys
# This validates complete time tracking information
# Example: [{"Año": 2013, "Horas": 203}, {"Año": 2014, "Horas": 150}]
assert all(["Año" in item and "Horas" in item for item in response.json()["Horas jugadas"]])

#17. Multiple Key Existence
# For game detail responses
# Must contain all required game information fields
# This validates complete game data
# Example: All keys ["id", "app_name", "developer", "genres"] exist
assert all(key in response.json() for key in ["id", "app_name", "developer", "genres"])

#18. Data Consistency
# For sentiment analysis totals
# Sum of sentiment counts must match total reviews
# This validates complete sentiment data
# Example: sum of ["Negative", "Neutral", "Positive"] = total_reviews
assert sum(int(x.split("=")[1]) for x in response.json()["Valve"]) == response.json()["total_reviews"]

#19. Cross-field Validation
# For game release data
# Release year must match posted date year
# This ensures data consistency
# Example: release_year "2015" in posted "2015-03-21"
assert str(response.json()["release_year"]) in response.json()["posted"]

#20. Response Completeness
# For all response data fields
# No field can contain null values
# This ensures complete data responses
# Example: All values in response are non-null
assert all(value is not None for value in response.json().values())

################################
# LOAD TESTS
################################
# These tests monitor performance and system behavior

#21. Response Time
# For all API endpoints
# Response time must be under 1 second
# This ensures acceptable API performance
# Example: response.elapsed.total_seconds() = 0.5
assert response.elapsed.total_seconds() < 1.0

#22. Empty List Handling
# For recommendation responses
# Return empty list for 404 status, otherwise non-empty
# This validates proper error handling
# Example: [] for 404, [recommendations] otherwise
assert response.json() == [] if response.status_code == 404 else len(response.json()) > 0

################################
# DATA TESTS
################################
# These tests validate data integrity, formats and transformations

#23. Dictionary Value Length
# For sentiment analysis responses (/sentiment_analysis/developer)
# Must contain exactly 3 categories: Negative, Neutral, Positive
# This validates complete sentiment analysis results are returned
# Example: {"Valve": ["Negative = 182", "Neutral = 120", "Positive = 278"]}
assert len(list(response.json().values())[0]) == 3

#24. Key Pattern in List Items
# In year-based query responses (like /PlayTimeGenre)
# At least one key must contain "Año de lanzamiento" text
# This ensures the response includes year information
# Example: {"Año de lanzamiento con más horas jugadas para Género X": "2013"}
assert any("Año de lanzamiento" in key for key in response.json().keys())

#25. Data Type of List Elements
# For endpoints returning multiple items
# Each item in the response must be a dictionary
# This ensures consistent data structure across all items
# Example: [{"key1": "value1"}, {"key2": "value2"}]
assert all(isinstance(item, dict) for item in response.json())

#26. Numeric Value Range
# For year-based queries in game data
# Year must be within valid range (1990-2023)
# This prevents invalid historical data
# Example: response["year"] = "2015"
assert 1990 <= int(response.json()["year"]) <= 2023

#27. Value Type in Nested Structures
# For playtime statistics in nested responses
# Hours played must be integer values
# This ensures consistent time measurement
# Example: {"Horas jugadas": [{"Año": 2013, "Horas": 203}]}
assert isinstance(response.json()["Horas jugadas"][0]["Horas"], int)

#28. Value Relationships
# For playtime statistics
# Total playtime must be non-negative
# This ensures valid time tracking data
# Example: response["playtime_forever"] >= 0
assert int(response.json()["playtime_forever"]) >= 0

#29. List Order
# For ranked playtime responses
# Hours must be in descending order
# This validates correct ranking by playtime
# Example: [{"Horas": 100}, {"Horas": 50}, {"Horas": 25}]
assert all(response.json()[i]["Horas"] >= response.json()[i+1]["Horas"] for i in range(len(response.json())-1))

#30. String Format
# For game ID validation
# ID must be a string of digits
# This ensures valid game identification
# Example: response["id"] = "123456"
assert response.json()["id"].isdigit()

#31. Float Value Precision
# For playtime hours conversion
# Hours must be rounded to whole numbers
# This ensures consistent time display
# Example: response["playtime_forever_Hrs"] = 10.0
assert float(response.json()["playtime_forever_Hrs"]).is_integer()

#32. Case Sensitivity
# For genre names in responses
# Genre must be properly capitalized
# This ensures consistent genre formatting
# Example: response["genres"][0] = "Action"
assert response.json()["genres"][0].istitle()

#33. String Length
# For game IDs in response
# Each ID must be non-empty string
# This ensures valid identification
# Example: All game_ids have length > 0
assert all(len(str(game_id)) > 0 for game_id in response.json().keys())

#34. Date Format
# For year values in playtime data
# Years must be 4-digit numbers
# This ensures valid date formatting
# Example: All years like "2015"
assert all(year.isdigit() and len(year) == 4 for year in [str(item["Año"]) for item in response.json()["Horas jugadas"]])

#35. Value Type Consistency
# For playtime values
# All playtime entries must be integers
# This ensures consistent time measurement
# Example: All "Horas" values are integers
assert all(isinstance(x["Horas"], int) for x in response.json()["Horas jugadas"])
