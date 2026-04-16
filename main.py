"""
main.py – LLM Function Calling Assistant: "What Next" Activity Recommender
============================================================================
Uses OpenAI's Tool Use (Function Calling) to recommend activities based on
the user's location, age, current time and season.

Run:
    python main.py
"""

import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment / client setup
# ---------------------------------------------------------------------------

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Copy .env.example to .env and fill in your key."
    )

client = OpenAI(api_key=_api_key)

# ---------------------------------------------------------------------------
# Helper: determine the current season (Northern Hemisphere)
# ---------------------------------------------------------------------------

def get_current_season() -> str:
    """Return the current meteorological season based on today's date.

    Returns:
        One of 'spring', 'summer', 'autumn', 'winter'.
    """
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
# Tool implementation
# ---------------------------------------------------------------------------

# A simple in-memory activity catalogue keyed by season → list of activities.
# In a real application this would call an external API or a database.
_ACTIVITY_CATALOGUE: dict[str, list[dict[str, Any]]] = {
    "spring": [
        {"name": "Cycling tour", "type": "outdoor", "min_age": 8},
        {"name": "Yoga in the park", "type": "outdoor", "min_age": 16},
        {"name": "Photography walk", "type": "outdoor", "min_age": 12},
        {"name": "Cinema matinée", "type": "indoor", "min_age": 0},
        {"name": "Board-game café", "type": "indoor", "min_age": 10},
    ],
    "summer": [
        {"name": "Open-air swimming pool", "type": "outdoor", "min_age": 5},
        {"name": "Beach volleyball", "type": "outdoor", "min_age": 12},
        {"name": "Outdoor cinema", "type": "outdoor", "min_age": 0},
        {"name": "Gym workout", "type": "indoor", "min_age": 16},
        {"name": "Art gallery visit", "type": "indoor", "min_age": 0},
    ],
    "autumn": [
        {"name": "Museum tour", "type": "indoor", "min_age": 0},
        {"name": "Escape room", "type": "indoor", "min_age": 12},
        {"name": "Study session at a library", "type": "indoor", "min_age": 8},
        {"name": "Hiking in the forest", "type": "outdoor", "min_age": 10},
        {"name": "Cooking class", "type": "indoor", "min_age": 14},
    ],
    "winter": [
        {"name": "Ice skating", "type": "outdoor", "min_age": 5},
        {"name": "Cinema", "type": "indoor", "min_age": 0},
        {"name": "Gym / fitness centre", "type": "indoor", "min_age": 16},
        {"name": "Yoga studio", "type": "indoor", "min_age": 16},
        {"name": "Online course / self-study", "type": "indoor", "min_age": 10},
    ],
}

# Activity adjustments based on time-of-day
_DAYTIME_OUTDOOR_CUTOFF = 19  # hours; outdoor activities only before 19:00


def get_recommendations(
    location: str,
    age: int,
    current_time: str,
    season: str,
) -> dict[str, Any]:
    """Return a list of activity recommendations for the given context.

    This function simulates looking up local activities (cinema, gym, yoga,
    study, outdoor sports, etc.) based on the provided parameters.

    Args:
        location: City and street (or neighbourhood) of the user,
                  e.g. "Prague, Wenceslas Square".
        age:      Age of the user in years.
        current_time: Current local time in HH:MM format, e.g. "14:30".
        season:   Current season – one of 'spring', 'summer', 'autumn', 'winter'.

    Returns:
        A dictionary with:
          - "location": echoed back location string
          - "recommendations": list of activity dicts, each with keys
            "name", "type", "suitable_age", "available_now"
    """
    # Normalise season to lower-case
    season_key = season.lower().strip()
    if season_key not in _ACTIVITY_CATALOGUE:
        season_key = "spring"

    # Parse hour from current_time
    try:
        hour = int(current_time.split(":")[0])
    except (ValueError, IndexError):
        hour = 12  # default to noon if parsing fails

    activities = _ACTIVITY_CATALOGUE[season_key]

    results: list[dict[str, Any]] = []
    for activity in activities:
        suitable_age = age >= activity["min_age"]
        # Outdoor activities are only available during daylight hours
        available_now = True
        if activity["type"] == "outdoor" and hour >= _DAYTIME_OUTDOOR_CUTOFF:
            available_now = False

        results.append(
            {
                "name": activity["name"],
                "type": activity["type"],
                "suitable_age": suitable_age,
                "available_now": available_now,
            }
        )

    return {
        "location": location,
        "recommendations": results,
    }


# ---------------------------------------------------------------------------
# OpenAI Tool / Function schema
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": (
                "Fetch a list of local activity recommendations (cinema, gym, "
                "yoga, study, outdoor sports, etc.) based on the user's "
                "location, age, current time, and season. Call this tool once "
                "you know all four parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "City and street or neighbourhood of the user, "
                            "e.g. 'Prague, Wenceslas Square'."
                        ),
                    },
                    "age": {
                        "type": "integer",
                        "description": "Age of the user in years.",
                    },
                    "current_time": {
                        "type": "string",
                        "description": (
                            "Current local time in HH:MM 24-hour format, "
                            "e.g. '14:30'."
                        ),
                    },
                    "season": {
                        "type": "string",
                        "enum": ["spring", "summer", "autumn", "winter"],
                        "description": "Current season.",
                    },
                },
                "required": ["location", "age", "current_time", "season"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool_call(name: str, arguments: str) -> str:
    """Execute the requested tool and return its result as a JSON string.

    Args:
        name:      Name of the tool to call.
        arguments: JSON-encoded string of arguments from the model.

    Returns:
        JSON string with the tool result.
    """
    args = json.loads(arguments)
    if name == "get_recommendations":
        result = get_recommendations(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Build the system prompt that includes the current time and season.

    Returns:
        A string containing the full system prompt.
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    season = get_current_season()
    date_str = now.strftime("%A, %d %B %Y")

    return (
        "You are a friendly local activity assistant. "
        "Your goal is to recommend the best 'what to do next' activity "
        "for the user based on their location, age, the current time, and season.\n\n"
        f"Current context (auto-detected):\n"
        f"  • Date and time: {date_str}, {current_time}\n"
        f"  • Season: {season}\n\n"
        "Instructions:\n"
        "1. Greet the user and ask for their age and location "
        "(city + street or neighbourhood) if you do not have it yet.\n"
        "2. Once you have location and age, call the `get_recommendations` tool "
        "passing location, age, the current time shown above, and the current season.\n"
        "3. Present the recommended activities in a friendly, concise way. "
        "Filter out activities that are not suitable for the user's age or "
        "not available at the current time.\n"
        "4. Always reply in the same language the user uses "
        "(Czech or English, or any other language they prefer).\n"
    )


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

def run_conversation() -> None:
    """Run the interactive activity-recommender conversation loop.

    The loop continues until the user types 'exit' or 'quit'.
    """
    print("=" * 60)
    print("  What Next? – Activity Recommender (powered by OpenAI)")
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
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                print(f"\n[ERROR] OpenAI API call failed: {exc}")
                break

            choice = response.choices[0]
            message = choice.message

            # --- Tool call branch ------------------------------------------
            if message.tool_calls:
                # Add the assistant message (with tool calls) to history
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments

                    print(
                        f"\n[TOOL CALL] Model is calling tool '{fn_name}' "
                        f"with arguments: {fn_args}"
                    )

                    tool_result = dispatch_tool_call(fn_name, fn_args)

                    print(f"[TOOL RESULT] {tool_result}")

                    # Append the tool result as a tool message
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                # Let the model process the tool result and continue
                continue

            # --- Final response branch -------------------------------------
            final_text = message.content or ""
            print(f"\n[ASSISTANT RESPONSE] Generating final answer …")
            print(f"\nAssistant: {final_text}")

            # Add assistant response to history for multi-turn context
            messages.append({"role": "assistant", "content": final_text})
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_conversation()
