import os
import logging
from datetime import datetime, timezone

from pyowm.owm import OWM
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
# Instructions

You are an assistant that will help me determine if I should stay at home or
go outside based on the weather conditions.

The user MUST provide you a location and potentially a time. If a location is
not provided, you should fail with a message indicating that the location is
required. Now, when a time is not supplied you should use the machine's time.

A time may be given as an ISO 8601 timestamp or a relative time expression
(e.g., "tomorrow at 3pm"). When the user provides you with a relative time
expression, you should convert it to an absolute time using the current
date and time.

When the user supplies a location it might be only the city name (e.g.,
"London" or "New York". In this case you will try to do your best effort
to retrieve the weather information for that city. For example, if the user
only passes you "London", you should retrieve the weather information for
that city using the code "Longon,GB".

You can ONLY answer questions about the present and up to 5 days in the future.
When a user requests in the past, you should respond with a message indicating
that you cannot provide that information.

Also, if you can't retrieve weather information for a specific location or
time, you should let the user know. You MUST NOT try to guess a location,
time, or weather information.

Also, you MUST retrieve the current datetime when its not provided and you must
use the get_current_datetime tool. DO NOT ATTEMPT TO GUESS the datetime.

You should answer with a warm and informative tone, providing the user with
the necessary weather information and guidance. For instance, if the weather
is severe, you should convey a sense of urgency in your response.

Now, assume that you were able to retrieve weather information for a specific
location and time. Your response should begin with your assessment of whether
it's a good idea to go outside or stay indoors, followed by the detailed
weather information. For example:

### Example Weather Assessment Response (It will be sunny)

You should definitively go outside and enjoy the beautiful weather!
The weather report is as follows:
- Location: {location}
- Time: {time}
- Temperature: {temperature}°C
- Rain: {rainchance}% chance
- Winds: {windspeed} km/h
- Humidity: {humidity}%
- Conditions: {conditions}

Have a happy day outside!

### Example Weather Assessment Response (It will be rainy)

You should definitely stay indoors, as heavy rain is expected.
The weather report is as follows:
- Location: {location}
- Time: {time}
- Temperature: {temperature}°C
- Rain: {rainchance}% chance
- Winds: {windspeed} km/h
- Humidity: {humidity}%
- Conditions: {conditions}

Have a cozy day inside!

### Example Weather Assessment Response (Failed to retrieve forecasts)

I'm sorry, but I couldn't retrieve the weather forecasts at this time.
Please try again later :(.
"""

FORECAST_TEMPLATE = """
The weather forecast for {location} at {time} is as follows:
- Temperature: {temperature:.2f}°C.
- Min Temperature: {min_temperature:.2f}°C.
- Max Temperature: {max_temperature:.2f}°C.
- Feels Like Temperature: {feels_like:.2f}°C.
- Rain: {rain_chance}%
- Winds: {wind_speed} m/s {wind_direction}°
- Humidity: {humidity}%
- Status: {status}
"""

WEATHER_API_KEY = os.environ['OPENWEATHERMAP_API_KEY']

if not WEATHER_API_KEY:
    raise ValueError("No Weather API key provided")

owm = OWM(WEATHER_API_KEY)
mgr = owm.weather_manager()


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15


def get_num_forecasts(end_time: str) -> int:
    end_time_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    delta = end_time_dt - now
    delta_in_hours = delta.total_seconds() / 3600
    return max(0, int((delta_in_hours // 3) + 1))


def get_weather_forecast(location: str, end_time: str) -> str:
    """
    Retrieves weather forecasts for a specific location in 3-hour intervals.
    For instance, to get the next 5 forecasts for London, you would call:
    get_weather_forecast("London,GB", "2023-10-01T12:00:00Z").

    Attributes:
        location (str): The location for which to retrieve the weather
            forecast. Must be on ISO 3166-1 format (e.g., "London,GB",
            "New York,US", etc.).
        end_time (str): The end time for the forecast in ISO 8601 format.

    Returns:
        str: A summary of the weather forecast with the following format:

        The weather forecast for {location} at {time} is as follows:
            - Temperature: {temperature}°C.
            - Min Temperature: {min_temperature}°C.
            - Max Temperature: {max_temperature}°C.
            - Feels Like Temperature: {feels_like}°C.
            - Rain chance: {rain_chance}%.
            - Wind speed: {wind_speed} m/s {wind_direction}.
            - Humidity: {humidity}%.
            - Conditions: {conditions}.
    """
    try:
        limit = get_num_forecasts(end_time)
        forecasts = mgr.forecast_at_place(location, "3h", limit).forecast
        forecasts = list(forecasts)
        forecast = forecasts[-1]

        temperature = forecast.temperature()
        wind = forecast.wind()
        rain = forecast.rain.get("3h", 0)
        rain_chance = rain * 100

        ans = FORECAST_TEMPLATE.format(
            location=location,
            time=forecast.reference_time('iso'),
            temperature=kelvin_to_celsius(temperature["temp"]),
            min_temperature=kelvin_to_celsius(temperature["temp_min"]),
            max_temperature=kelvin_to_celsius(temperature["temp_max"]),
            feels_like=kelvin_to_celsius(temperature["feels_like"]),
            rain_chance=rain_chance,
            wind_speed=wind["speed"],
            wind_direction=wind["deg"],
            humidity=forecast.humidity,
            status=forecast.detailed_status,
        )

        return ans
    except Exception as e:
        logger.error(f"Error occurred while retrieving weather forecast: {e}")
        return ("I'm sorry, but I couldn't retrieve the weather forecasts at "
                "this time. Please try again later :(.")


def get_current_datetime() -> str:
    """
    Returns the machine's current date and time in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


agent = create_react_agent(
    model=ChatOllama(model="qwen2.5:7b-instruct"),
    tools=[get_weather_forecast, get_current_datetime],
    prompt=SYSTEM_PROMPT,
)
