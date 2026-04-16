"""
main.py – LLM Function Calling Assistant: "What Next" Activity Recommender
============================================================================
Uses OpenAI's Tool Use (Function Calling) to recommend activities based on
the user's location, age group, current time (auto-detected), and season
(auto-detected from system clock).

NEW in this version
-------------------
• trafilatura web search – fetches real local activity listings from the web.
  The search query is constructed from the user's location and interests.
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

_WEB_SEARCH_RESULT_LIMIT = 3000  # characters of extracted text to return


def search_activities_web(location: str, interests: str) -> dict[str, Any]:
    """Search the web for real local activities using trafilatura.

    Constructs a natural-language query from the location and the user's
    interests, fetches DuckDuckGo HTML results, and extracts the main text.

    Args:
        location:  City/neighbourhood, e.g. "Brno, Czech Republic".
        interests: Comma-separated interests or activity keywords gathered
                   from the conversation, e.g. "outdoor, hiking, family".

    Returns:
        A dict with the search query used and extracted text snippets.
    """
    now = datetime.now()
    season = get_current_season()
    day_str = now.strftime("%A")  # e.g. "Wednesday"

    query = f"{interests} activities {location} {season} {day_str}"
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
        include_links=False,
        no_fallback=False,
        favor_recall=True,
    )

    if not text:
        return {"query": query, "results": "No readable content could be extracted."}

    return {
        "query": query,
        "results": text[:_WEB_SEARCH_RESULT_LIMIT],
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
            "name": "search_activities_web",
            "description": (
                "Search the web for real local activities at the user's location. "
                "Use this to supplement catalogue results with current/live events. "
                "The season and weekday are injected automatically. "
                "Call this when the user wants specific or up-to-date suggestions."
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
                },
                "required": ["location", "interests"],
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
    if name == "search_activities_web":
        result = search_activities_web(**args)
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
        "   b. If they skip or decline, ask: 'What was the last activity you did?' "
        "      Use their answer to estimate an age group "
        "      (child / teen / young_adult / adult / senior) and proceed.\n"
        "   c. Never ask the user for the current time or the season – "
        "      those are already known.\n"
        "3. Once you have location and age (or estimated age group), call "
        "`get_recommendations` with the appropriate horizon.\n"
        "   • Offer all three horizons (day / week / month) in your reply "
        "     unless the user specifies one.\n"
        "   • People naturally plan on three cycles: "
        "     daily impulse, weekly routine, monthly calendar event.\n"
        "4. For live/local events, also call `search_activities_web` using the "
        "   user's location and inferred interests.\n"
        "5. Present results in a friendly, concise way, grouping them by horizon.\n"
        "6. Always reply in the same language the user uses.\n"
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
            print("Goodbye! / Nashledanou!")
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

                    print(f"[TOOL RESULT] {tool_result[:300]}{'…' if len(tool_result) > 300 else ''}")

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
