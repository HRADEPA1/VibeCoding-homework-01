"""
main.py – LLM Function Calling Assistant: "What Next" Activity Recommender
============================================================================
Uses OpenAI's Tool Use (Function Calling) to recommend activities based on
the user's location, age group, current time (auto-detected), and season
(auto-detected from system clock).

Features
--------
• trafilatura web search – fetches real local activity listings from the web.
  The search query is constructed from the user's location and interests.
  Searches start near the user (walking distance) and expand radius
  automatically when nearby results are sparse.
• Rich event details – web queries request venue name, address, start time,
  and website URL so recommendations are actionable.
• Weather-aware – get_weather_forecast() fetches the current forecast from
  wttr.in (no API key). Outdoor activities are suppressed when the weather
  is unsuitable (rain, storm, heavy wind, etc.).
• Public transport – search_public_transport() finds transit options between
  the user's location and a chosen event address.
• Age inference – if the user does not state their age, the assistant asks
  about their last activity and estimates an age group instead.
• Three-horizon planning:
    short  (day)   – what to do today / in the next few hours
    mid    (week)  – recurring or best-fit activities for the coming week
    long   (month) – seasonal or monthly events to plan ahead

Supported backends (configured via .env):
  1. Remote OpenAI API  – set OPENAI_API_KEY
  2. Local AI server    – set OPENAI_BASE_URL (Ollama / LM Studio)

Run:
    python main.py
"""

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

import trafilatura
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment / client setup
# ---------------------------------------------------------------------------

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY", "")
_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
_model = os.getenv("MODEL_NAME", "").strip() or "gpt-4o-mini"

# For local backends (Ollama, LM Studio) a real API key is not required.
# We fall back to the placeholder "local" so the SDK does not complain.
if not _api_key:
    if _base_url:
        _api_key = "local"
    else:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set and no OPENAI_BASE_URL is configured.\n"
            "  • For the remote OpenAI API: copy .env.example to .env and add your key.\n"
            "  • For a local AI (Ollama / LM Studio): set OPENAI_BASE_URL in .env."
        )

client = OpenAI(api_key=_api_key, base_url=_base_url)

# ---------------------------------------------------------------------------
# Helper: current season (Northern Hemisphere)
# ---------------------------------------------------------------------------

def get_current_season() -> str:
    """Return the current meteorological season based on today's date."""
    month = datetime.now().month
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"


# ---------------------------------------------------------------------------
# Age-group helpers
# ---------------------------------------------------------------------------

# Maps age-group label → (min_age, max_age) inclusive.
# "senior" has no practical upper bound.
# Reserved for future age-range validation and display logic.
_AGE_GROUP_RANGES: dict[str, tuple[int, int]] = {
    "child":       (0,  12),
    "teen":        (13, 17),
    "young_adult": (18, 30),
    "adult":       (31, 60),
    "senior":      (61, 120),
}

# Representative midpoint age used when only a group label is known.
_AGE_GROUP_MIDPOINT: dict[str, int] = {
    "child":       8,
    "teen":        15,
    "young_adult": 25,
    "adult":       40,
    "senior":      68,
}


def resolve_age(age_or_group: str | int) -> int:
    """Resolve a numeric age string or age-group label to a representative integer age.

    Args:
        age_or_group: An integer age, a numeric string, or an age-group name
                      ('child', 'teen', 'young_adult', 'adult', 'senior').

    Returns:
        A representative integer age.
    """
    if isinstance(age_or_group, int):
        return age_or_group
    normalised = str(age_or_group).strip().lower().replace("-", "_").replace(" ", "_")
    if normalised in _AGE_GROUP_MIDPOINT:
        return _AGE_GROUP_MIDPOINT[normalised]
    try:
        return int(normalised)
    except ValueError:
        return _AGE_GROUP_MIDPOINT["adult"]  # safe fallback


# ---------------------------------------------------------------------------
# Built-in activity catalogue
# ---------------------------------------------------------------------------

_ACTIVITY_CATALOGUE: dict[str, list[dict[str, Any]]] = {
    "spring": [
        {"name": "Cycling tour",       "type": "outdoor", "min_age": 8,  "recurring": True},
        {"name": "Yoga in the park",   "type": "outdoor", "min_age": 16, "recurring": True},
        {"name": "Photography walk",   "type": "outdoor", "min_age": 12, "recurring": False},
        {"name": "Cinema matinée",     "type": "indoor",  "min_age": 0,  "recurring": False},
        {"name": "Board-game café",    "type": "indoor",  "min_age": 10, "recurring": True},
        {"name": "Farmers' market",    "type": "outdoor", "min_age": 0,  "recurring": True},
        {"name": "Art workshop",       "type": "indoor",  "min_age": 8,  "recurring": True},
    ],
    "summer": [
        {"name": "Open-air swimming pool", "type": "outdoor", "min_age": 5,  "recurring": True},
        {"name": "Beach volleyball",       "type": "outdoor", "min_age": 12, "recurring": True},
        {"name": "Outdoor cinema",         "type": "outdoor", "min_age": 0,  "recurring": False},
        {"name": "Gym workout",            "type": "indoor",  "min_age": 16, "recurring": True},
        {"name": "Art gallery visit",      "type": "indoor",  "min_age": 0,  "recurring": False},
        {"name": "Sunrise hike",           "type": "outdoor", "min_age": 12, "recurring": False},
        {"name": "Language exchange meetup","type": "indoor", "min_age": 14, "recurring": True},
    ],
    "autumn": [
        {"name": "Museum tour",                   "type": "indoor",  "min_age": 0,  "recurring": False},
        {"name": "Escape room",                   "type": "indoor",  "min_age": 12, "recurring": False},
        {"name": "Study session at a library",    "type": "indoor",  "min_age": 8,  "recurring": True},
        {"name": "Hiking in the forest",          "type": "outdoor", "min_age": 10, "recurring": True},
        {"name": "Cooking class",                 "type": "indoor",  "min_age": 14, "recurring": True},
        {"name": "Photography walk (autumn foliage)", "type": "outdoor", "min_age": 12, "recurring": False},
        {"name": "Theatre performance",           "type": "indoor",  "min_age": 10, "recurring": False},
    ],
    "winter": [
        {"name": "Ice skating",              "type": "outdoor", "min_age": 5,  "recurring": True},
        {"name": "Cinema",                   "type": "indoor",  "min_age": 0,  "recurring": False},
        {"name": "Gym / fitness centre",     "type": "indoor",  "min_age": 16, "recurring": True},
        {"name": "Yoga studio",              "type": "indoor",  "min_age": 16, "recurring": True},
        {"name": "Online course / self-study","type": "indoor", "min_age": 10, "recurring": True},
        {"name": "Hot-chocolate café walk",  "type": "outdoor", "min_age": 0,  "recurring": False},
        {"name": "Volunteer at a food bank", "type": "indoor",  "min_age": 14, "recurring": True},
    ],
}

_DAYTIME_OUTDOOR_CUTOFF = 19  # hours; outdoor activities not offered after 19:00


# ---------------------------------------------------------------------------
# Tool 1: get_recommendations
# ---------------------------------------------------------------------------

def get_recommendations(
    location: str,
    age_group: str,
    horizon: str = "day",
) -> dict[str, Any]:
    """Return catalogue-based activity recommendations.

    Time and season are sourced automatically from the system clock so
    the caller never needs to supply them.

    Args:
        location:  City and street/neighbourhood, e.g. "Prague, Wenceslas Square".
        age_group: Numeric age (e.g. "28") **or** group label:
                   'child' (0-12), 'teen' (13-17), 'young_adult' (18-30),
                   'adult' (31-60), 'senior' (61+).
        horizon:   Planning window – 'day', 'week', or 'month'.

    Returns:
        A dict with location, horizon, current_time, season, and a list of
        recommendations each carrying suitability and availability flags.
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    season = get_current_season()
    hour = now.hour

    season_key = season.lower().strip()
    if season_key not in _ACTIVITY_CATALOGUE:
        season_key = "spring"

    age_int = resolve_age(age_group)
    horizon_key = horizon.lower().strip()

    activities = _ACTIVITY_CATALOGUE[season_key]
    results: list[dict[str, Any]] = []

    for activity in activities:
        suitable_age = age_int >= activity["min_age"]

        # Outdoor activities not shown after dark
        available_now = True
        if activity["type"] == "outdoor" and hour >= _DAYTIME_OUTDOOR_CUTOFF:
            available_now = False

        # Horizon filtering: week/month horizons emphasise recurring activities;
        # day horizon focuses on immediately available options.
        horizon_relevant = True
        if horizon_key == "day" and not available_now:
            horizon_relevant = False
        elif horizon_key == "week":
            # For a weekly plan, recurring activities are highlighted
            horizon_relevant = True
        elif horizon_key == "month":
            # For a monthly plan, all activities are relevant
            horizon_relevant = True

        results.append(
            {
                "name": activity["name"],
                "type": activity["type"],
                "suitable_age": suitable_age,
                "available_now": available_now,
                "recurring": activity["recurring"],
                "horizon_relevant": horizon_relevant,
            }
        )

    return {
        "location": location,
        "age_group": age_group,
        "current_time": current_time,
        "season": season,
        "horizon": horizon_key,
        "recommendations": results,
    }


# ---------------------------------------------------------------------------
# Tool 2: search_activities_web
# ---------------------------------------------------------------------------

_WEB_SEARCH_MAX_CHARS = 4000  # maximum characters of extracted text to return
_CONSOLE_PREVIEW_LENGTH = 300  # characters of tool result shown in the console

# Weather codes from WorldWeatherOnline / wttr.in that indicate unsuitable
# conditions for outdoor activities (rain, snow, sleet, fog, thunderstorms).
_BAD_WEATHER_CODES: frozenset[int] = frozenset({
    176, 179, 182, 185, 200, 227, 230, 248, 260,
    263, 266, 281, 284, 293, 296, 299, 302, 305, 308,
    311, 314, 317, 320, 323, 326, 329, 332, 335, 338,
    350, 353, 356, 359, 362, 365, 368, 371, 374, 377,
    386, 389, 392, 395,
})
_WIND_UNSUITABLE_KMPH = 50   # wind speed above which outdoor is unsuitable
_FEELS_LIKE_MIN_C = -5       # feels-like temperature below which outdoor is unsuitable


def search_activities_web(
    location: str,
    interests: str,
    radius_km: int = 2,
) -> dict[str, Any]:
    """Search the web for real local activities using trafilatura.

    Constructs a rich natural-language query from the location, interests,
    and search radius. Includes terms that help surface event details such
    as venue name, address, start time, and website.

    Args:
        location:  City/neighbourhood, e.g. "Brno, Czech Republic".
        interests: Comma-separated interests or activity keywords gathered
                   from the conversation, e.g. "outdoor, hiking, family".
        radius_km: Search radius in kilometres. Use 1-2 for walking distance,
                   5 for neighbourhood, 15 for whole city. The model should
                   start small and call again with a larger radius when the
                   first call returns insufficient results.

    Returns:
        A dict with the search query used, the radius, and extracted text.
    """
    now = datetime.now()
    season = get_current_season()
    day_str = now.strftime("%A")  # e.g. "Wednesday"
    date_str = now.strftime("%d %B %Y")

    # Build a radius-aware location hint
    city = location.split(",")[0].strip()
    if radius_km <= 2:
        location_hint = f"near {location} walking distance"
    elif radius_km <= 7:
        location_hint = f"within {radius_km} km of {location}"
    elif radius_km <= 20:
        location_hint = f"in {city}"
    else:
        location_hint = location

    query = (
        f"{interests} activities {location_hint} {season} {day_str} {date_str} "
        f"venue address schedule opening hours website tickets"
    )
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    try:
        downloaded = trafilatura.fetch_url(search_url)
    except Exception as exc:
        return {"error": f"Fetch failed: {exc}", "results": "", "radius_km": radius_km}

    if not downloaded:
        return {"error": "No content returned from search engine", "results": "", "radius_km": radius_km}

    text = trafilatura.extract(
        downloaded,
        include_links=True,
        no_fallback=False,
        favor_recall=True,
    )

    if not text:
        return {"query": query, "radius_km": radius_km, "results": "No readable content could be extracted."}

    return {
        "query": query,
        "radius_km": radius_km,
        "results": text[:_WEB_SEARCH_MAX_CHARS],
    }


# ---------------------------------------------------------------------------
# Tool 3: get_weather_forecast
# ---------------------------------------------------------------------------

def get_weather_forecast(location: str) -> dict[str, Any]:
    """Fetch the current weather and 3-day forecast from wttr.in (no API key needed).

    Uses the public wttr.in JSON API.  Returns structured weather data
    including an ``outdoor_suitable`` flag so callers can decide whether to
    recommend outdoor activities.

    Args:
        location: City or address, e.g. "Brno, Czech Republic".

    Returns:
        A dict with current conditions, a 3-day forecast array, and
        ``outdoor_suitable`` (bool).  On error the function returns
        ``outdoor_suitable=True`` so the conversation is never blocked.
    """
    url = f"https://wttr.in/{urllib.parse.quote_plus(location)}?format=j1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WhatNextApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return {
            "error": str(exc),
            "outdoor_suitable": True,
            "note": "Weather check failed – assuming outdoor is OK",
        }

    current = data.get("current_condition", [{}])[0]
    weather_code = int(current.get("weatherCode", "113"))
    temp_c = int(current.get("temp_C", "20"))
    feels_like_c = int(current.get("FeelsLikeC", str(temp_c)))
    wind_kmph = int(current.get("windspeedKmph", "0"))
    humidity = int(current.get("humidity", "50"))
    desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")

    outdoor_suitable = (
        weather_code not in _BAD_WEATHER_CODES
        and wind_kmph < _WIND_UNSUITABLE_KMPH
        and feels_like_c > _FEELS_LIKE_MIN_C
    )

    if not outdoor_suitable:
        if weather_code in _BAD_WEATHER_CODES:
            reason = f"precipitation/storm (code {weather_code})"
        elif wind_kmph >= _WIND_UNSUITABLE_KMPH:
            reason = f"strong wind ({wind_kmph} km/h)"
        else:
            reason = f"very cold (feels like {feels_like_c}°C)"
    else:
        reason = None

    forecast = []
    for day in data.get("weather", [])[:3]:
        hourly = day.get("hourly", [])
        midday = hourly[4] if len(hourly) > 4 else (hourly[0] if hourly else {})
        day_code = int(midday.get("weatherCode", "113"))
        day_desc = midday.get("weatherDesc", [{}])[0].get("value", "")
        forecast.append(
            {
                "date": day.get("date", ""),
                "max_temp_c": day.get("maxtempC", ""),
                "min_temp_c": day.get("mintempC", ""),
                "description": day_desc,
                "outdoor_suitable": day_code not in _BAD_WEATHER_CODES,
            }
        )

    return {
        "location": location,
        "current": {
            "description": desc,
            "temp_c": temp_c,
            "feels_like_c": feels_like_c,
            "wind_kmph": wind_kmph,
            "humidity_pct": humidity,
        },
        "outdoor_suitable": outdoor_suitable,
        "outdoor_unsuitable_reason": reason,
        "forecast": forecast,
    }


# ---------------------------------------------------------------------------
# Tool 4: search_public_transport
# ---------------------------------------------------------------------------

def search_public_transport(from_location: str, to_event_address: str) -> dict[str, Any]:
    """Search for public transport options between the user's location and an event.

    Uses trafilatura to fetch DuckDuckGo search results for transit directions.
    Looks for bus, tram, metro, and train options.

    Args:
        from_location:    User's current location, e.g. "Brno, Masarykova 10".
        to_event_address: Destination address or venue name,
                          e.g. "Janáček Theatre, Brno".

    Returns:
        A dict with the departure, destination, and extracted transit text.
    """
    query = (
        f"public transport bus tram metro train directions "
        f"from {from_location} to {to_event_address}"
    )
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    try:
        downloaded = trafilatura.fetch_url(search_url)
    except Exception as exc:
        return {"error": f"Fetch failed: {exc}", "results": ""}

    if not downloaded:
        return {"error": "No content returned from search engine", "results": ""}

    text = trafilatura.extract(
        downloaded,
        include_links=True,
        no_fallback=False,
        favor_recall=True,
    )

    if not text:
        return {
            "from": from_location,
            "to": to_event_address,
            "results": "No transit information found.",
        }

    return {
        "from": from_location,
        "to": to_event_address,
        "results": text[:_WEB_SEARCH_MAX_CHARS],
    }



# ---------------------------------------------------------------------------
# OpenAI Tool / Function schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Return catalogue-based local activity recommendations. "
                "The current time and season are sourced automatically from the "
                "system clock – do NOT pass them. "
                "Call this once you know the user's location and age (or age group). "
                "Choose horizon='day' for today, 'week' for a weekly plan, "
                "'month' for a monthly overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and street or neighbourhood, "
                            "e.g. 'Prague, Wenceslas Square'."
                        ),
                    },
                    "age_group": {
                        "type": "string",
                        "description": (
                            "Numeric age as a string (e.g. '28') "
                            "OR one of: 'child' (0-12), 'teen' (13-17), "
                            "'young_adult' (18-30), 'adult' (31-60), 'senior' (61+). "
                            "Infer from context if the user has not stated their age."
                        ),
                    },
                    "horizon": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "description": (
                            "Planning window: 'day' = today, "
                            "'week' = this week, 'month' = this month."
                        ),
                    },
                },
                "required": ["location", "age_group", "horizon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Fetch the current weather and 3-day forecast for the user's location "
                "from wttr.in (no API key required). "
                "Always call this BEFORE recommending activities so you can skip "
                "outdoor activities when the weather is unsuitable. "
                "Returns outdoor_suitable=true/false, current conditions, and a forecast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or address, e.g. 'Brno, Czech Republic'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_activities_web",
            "description": (
                "Search the web for real local activities at the user's location. "
                "Returns venue names, addresses, start times, and website URLs when available. "
                "Start with radius_km=2 (walking distance). "
                "If that returns too few results, call again with radius_km=5, then radius_km=15. "
                "The season and date are injected automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Brno, Czech Republic'.",
                    },
                    "interests": {
                        "type": "string",
                        "description": (
                            "Comma-separated interests or activity keywords from "
                            "the conversation, e.g. 'outdoor, hiking, family-friendly'."
                        ),
                    },
                    "radius_km": {
                        "type": "integer",
                        "description": (
                            "Search radius in kilometres. "
                            "Use 2 for walking distance, 5 for neighbourhood, "
                            "15 for the whole city. Default is 2."
                        ),
                    },
                },
                "required": ["location", "interests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_public_transport",
            "description": (
                "Search for public transport options (bus, tram, metro, train) "
                "between the user's current location and the event address. "
                "Call this after the user has chosen a specific activity or venue "
                "to help them get there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_location": {
                        "type": "string",
                        "description": "User's current location, e.g. 'Brno, Masarykova 10'.",
                    },
                    "to_event_address": {
                        "type": "string",
                        "description": (
                            "Destination address or venue name, "
                            "e.g. 'Janáček Theatre, Brno, Rooseveltova 1'."
                        ),
                    },
                },
                "required": ["from_location", "to_event_address"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool_call(name: str, arguments: str) -> str:
    """Execute the requested tool and return its result as a JSON string."""
    args = json.loads(arguments)
    if name == "get_recommendations":
        result = get_recommendations(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "get_weather_forecast":
        result = get_weather_forecast(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "search_activities_web":
        result = search_activities_web(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "search_public_transport":
        result = search_public_transport(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Build the system prompt, injecting current date/time and season."""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    season = get_current_season()
    date_str = now.strftime("%A, %d %B %Y")

    return (
        "You are a friendly local activity assistant. "
        "Your goal is to recommend the best activities for the user across three "
        "planning horizons: today (short), this week (mid), and this month (long).\n\n"
        f"Current context (auto-detected – do NOT ask the user for these):\n"
        f"  • Date and time : {date_str}, {current_time}\n"
        f"  • Season        : {season}\n\n"
        "Conversation guidelines:\n"
        "1. Ask for the user's **location** (city + street or neighbourhood).\n"
        "2. For **age**:\n"
        "   a. Ask once: 'How old are you?' (optional, not mandatory).\n"
        "   b. If they skip or decline, ask: 'What was the last activity you did?'"
        "      Use their answer to estimate an age group"
        "      (child / teen / young_adult / adult / senior) and proceed.\n"
        "   c. Never ask the user for the current time or the season –"
        "      those are already known.\n"
        "3. Once you have the location, call `get_weather_forecast` IMMEDIATELY.\n"
        "   • If outdoor_suitable is false, do NOT suggest outdoor activities.\n"
        "     Briefly mention the weather reason and offer indoor alternatives only.\n"
        "   • If outdoor_suitable is true, include outdoor options as appropriate.\n"
        "4. Call `get_recommendations` with the appropriate horizon for a catalogue view.\n"
        "   • Offer all three horizons (day / week / month) unless the user picks one.\n"
        "5. Also call `search_activities_web` for live, real-world events.\n"
        "   • Start with radius_km=2 (walking distance).\n"
        "   • If results are sparse (fewer than 3 activities), call again with radius_km=5,"
        "     then radius_km=15 if still sparse.\n"
        "   • Extract and present for each activity:\n"
        "       – venue / building / club name\n"
        "       – full street address\n"
        "       – start time and date\n"
        "       – website or booking link (if found)\n"
        "6. After listing activities, ask the user if they want directions.\n"
        "   If yes, call `search_public_transport` with their location and the venue address.\n"
        "7. Present results grouped by planning horizon (🕐 Today / 📅 Week / 🗓️ Month).\n"
        "8. Always reply in the same language the user uses.\n"
    )


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

def run_conversation() -> None:
    """Run the interactive activity-recommender conversation loop."""
    print("=" * 60)
    backend_label = _base_url if _base_url else "OpenAI API"
    print("  What Next? – Activity Recommender")
    print(f"  Backend : {backend_label}")
    print(f"  Model   : {_model}")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
    ]

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Keep looping until the model produces a final (non-tool-call) response
        while True:
            try:
                response = client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                print(f"\n[ERROR] API call failed: {exc}")
                break

            choice = response.choices[0]
            message = choice.message

            # --- Tool call branch ------------------------------------------
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments

                    print(
                        f"\n[TOOL CALL] '{fn_name}' "
                        f"← {fn_args}"
                    )

                    tool_result = dispatch_tool_call(fn_name, fn_args)

                    print(f"[TOOL RESULT] {tool_result[:_CONSOLE_PREVIEW_LENGTH]}{'…' if len(tool_result) > _CONSOLE_PREVIEW_LENGTH else ''}")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                continue  # let model process tool results

            # --- Final response branch -------------------------------------
            final_text = message.content or ""
            print("\n[ASSISTANT RESPONSE] Generating final answer …")
            print(f"\nAssistant: {final_text}")

            messages.append({"role": "assistant", "content": final_text})
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_conversation()
