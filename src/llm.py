"""
llm.py - OpenAI API wrapper with tool calling support.

Handles communication with OpenAI's API, including tool/function calling
for statistical analysis of football data.
"""

import os
import json
from typing import List, Dict, Any, Optional, Generator
from functools import lru_cache
import hashlib

# OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.tools import TOOL_DEFINITIONS, execute_tool


# Model configuration - easy to change
MODEL_NAME = "gpt-4o-mini"  # Cost-effective model with good capabilities
# Alternative models: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"

# System prompt that defines the AI's behavior
SYSTEM_PROMPT = """Du är en expert på fotbollsstatistik som specialiserar dig på Premier League-analys. Du kommunicerar alltid på svenska.

DITT FOKUSOMRÅDE:
Du är särskilt kunnig om hur outliers (extremvärden) kan göra medelvärden missvisande i fotbollsstatistik. Du förklarar alltid pedagogiskt:
- Varför medelvärde kan lura en (en extrem match drar upp/ner snittet)
- Varför median och trimmat medelvärde ofta ger en bättre bild
- Hur man identifierar outliers med IQR och MAD-metoden

VIKTIGA REGLER:
1. ANVÄND ALLTID verktygen för att hämta faktiska siffror - gissa ALDRIG statistik
2. Basera ALLA svar på verktygsresultat, inte på egna antaganden
3. Förklara pedagogiskt vad siffrorna betyder och varför de ser ut som de gör
4. Om du hittar outliers, berätta vilka matcher det gäller och hur de påverkar snittet
5. Ge konkreta rekommendationer om vilka mått man bör titta på

NÄR DU SVARAR:
- Börja med en sammanfattning av det viktigaste
- Visa relevanta siffror (medel, median, trimmat medelvärde)
- Peka ut eventuella outliers och förklara deras påverkan
- Avsluta med en praktisk slutsats

TILLGÄNGLIGA METRICS:
- throw_ins: Antal inkast
- fouls: Antal frisparkar (begångna)
- shots: Antal skott

EXEMPEL PÅ BRA SVAR:
"Arsenal har i snitt 24.3 inkast per match de senaste 10 matcherna. MEN - medianen är bara 21, och det trimmat medelvärdet är 21.5. Varför skillnaden? Jo, matchen mot Burnley (2024-02-15) stack ut med hela 42 inkast - en extrem outlier som drar upp snittet med nästan 3 inkast. Utan den matchen ligger snittet på 21.8, vilket ger en mer rättvisande bild av Arsenals normala spel."
"""


# Response cache
_response_cache: Dict[str, str] = {}


def get_cache_key(messages: List[Dict]) -> str:
    """Generate a cache key from messages."""
    content = json.dumps(messages, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def check_api_key() -> tuple[bool, str]:
    """
    Check if OpenAI API key is available.
    
    Returns:
        Tuple of (is_available, message)
    """
    if not OPENAI_AVAILABLE:
        return False, "OpenAI-biblioteket är inte installerat. Kör: pip install openai"
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return False, """
⚠️ **OPENAI_API_KEY saknas!**

För att använda denna app behöver du sätta din OpenAI API-nyckel som miljövariabel.

**Windows (PowerShell):**
```
$env:OPENAI_API_KEY="din-api-nyckel-här"
```

**Windows (CMD):**
```
set OPENAI_API_KEY=din-api-nyckel-här
```

**Linux/Mac:**
```
export OPENAI_API_KEY="din-api-nyckel-här"
```

Starta sedan om Streamlit-appen.

Du kan skaffa en API-nyckel på: https://platform.openai.com/api-keys
"""
    
    return True, "API-nyckel hittad ✓"


def get_client() -> Optional[OpenAI]:
    """Get OpenAI client if API key is available."""
    is_available, _ = check_api_key()
    if not is_available:
        return None
    return OpenAI()


def chat_with_tools(
    messages: List[Dict[str, str]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Send a chat request with tool calling support.
    
    This function handles the complete flow:
    1. Send user message to the model
    2. If model wants to call tools, execute them
    3. Send tool results back to model
    4. Return final response
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        use_cache: Whether to cache responses
    
    Returns:
        Dictionary with 'response' (text), 'tool_calls' (list of calls made),
        and 'tool_results' (list of results)
    """
    client = get_client()
    if client is None:
        is_available, message = check_api_key()
        return {
            "response": message,
            "tool_calls": [],
            "tool_results": [],
            "error": True
        }
    
    # Check cache
    cache_key = get_cache_key(messages)
    if use_cache and cache_key in _response_cache:
        return _response_cache[cache_key]
    
    # Build full message list with system prompt
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + messages
    
    tool_calls_made = []
    tool_results = []
    
    try:
        # Initial API call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=full_messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Check if model wants to call tools
        while assistant_message.tool_calls:
            # Process each tool call
            tool_call_messages = []
            
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the tool
                result = execute_tool(function_name, function_args)
                
                tool_calls_made.append({
                    "name": function_name,
                    "arguments": function_args
                })
                tool_results.append({
                    "name": function_name,
                    "result": json.loads(result)
                })
                
                # Add tool result to messages
                tool_call_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Add assistant message with tool calls and tool results
            full_messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            full_messages.extend(tool_call_messages)
            
            # Make another API call with tool results
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=full_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
        
        # Final response
        final_response = assistant_message.content or ""
        
        result = {
            "response": final_response,
            "tool_calls": tool_calls_made,
            "tool_results": tool_results,
            "error": False
        }
        
        # Cache the result
        if use_cache:
            _response_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        return {
            "response": f"❌ Ett fel uppstod vid kommunikation med OpenAI: {str(e)}",
            "tool_calls": tool_calls_made,
            "tool_results": tool_results,
            "error": True
        }


def clear_cache():
    """Clear the response cache."""
    global _response_cache
    _response_cache = {}


def get_example_questions() -> List[str]:
    """Get list of example questions in Swedish."""
    return [
        "Hur ser Arsenals inkaststatistik ut de senaste 10 matcherna?",
        "Jämför Liverpool och Manchester City's skott de senaste 15 matcherna",
        "Vilka outlier-matcher har Chelsea haft för frisparkar?",
        "Analysera Tottenhams statistik och förklara varför medelvärdet kan vara missvisande",
        "Hur påverkar extremmatcher Newcastles genomsnittliga inkast?",
        "Jämför West Ham och Everton - vilket lag har mest stabila värden?",
        "Visa mig alla metrics för Brighton de senaste 8 matcherna",
        "Finns det några tydliga outliers i Manchester Uniteds skottstatistik?",
        "Vilka lag finns tillgängliga i databasen?",
        "Förklara skillnaden mellan medelvärde och median för Burnleys inkast"
    ]


def format_tool_results_for_display(tool_results: List[Dict]) -> List[Dict]:
    """
    Format tool results for display in Streamlit expander.
    
    Returns simplified, display-friendly version of results.
    """
    formatted = []
    
    for result in tool_results:
        name = result["name"]
        data = result["result"]
        
        if "error" in data:
            formatted.append({
                "tool": name,
                "type": "error",
                "content": data["error"]
            })
            continue
        
        if name == "get_team_summary":
            formatted.append({
                "tool": name,
                "type": "summary",
                "team": data.get("team"),
                "num_matches": data.get("num_matches"),
                "date_range": data.get("date_range"),
                "metrics": data.get("metrics", {}),
                "matches": data.get("matches", [])
            })
        
        elif name == "compare_teams":
            formatted.append({
                "tool": name,
                "type": "comparison",
                "team_a": data.get("team_a"),
                "team_b": data.get("team_b"),
                "comparisons": data.get("comparisons", {}),
                "insights": data.get("insights", [])
            })
        
        elif name == "get_outlier_matches":
            formatted.append({
                "tool": name,
                "type": "outliers",
                "team": data.get("team"),
                "metric": data.get("metric"),
                "stats": data.get("stats", {}),
                "outlier_matches": data.get("outlier_matches", []),
                "mean_impact": data.get("mean_impact", 0)
            })
        
        elif name == "get_available_teams_list":
            formatted.append({
                "tool": name,
                "type": "teams",
                "teams": data.get("teams", [])
            })
        
        else:
            formatted.append({
                "tool": name,
                "type": "raw",
                "content": data
            })
    
    return formatted
