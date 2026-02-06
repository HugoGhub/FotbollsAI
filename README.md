# ⚽ Premier League Statistik Chatbot

En Streamlit-baserad AI-chattbot som analyserar Premier League-statistik och fokuserar på **outliers** (extremvärden) som kan göra medelvärden missvisande. Använder **OpenAI API** med **tool calling** för intelligent analys.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)

## 🎯 Funktioner

- **AI-driven chatt** - Ställ frågor på svenska om fotbollsstatistik
- **OpenAI Tool Calling** - AI:n använder verktyg för att hämta faktisk data
- **Outlier-detektion** med två metoder:
  - IQR-metoden (Interquartile Range)
  - Robust z-score baserat på median och MAD
- **Robusta statistiska mått**:
  - Median
  - Trimmed mean (10%)
  - Jämförelse med/utan outliers
- **Interaktiva visualiseringar** med Plotly
- **Jämförelse mellan lag** med insikter
- **Session state** - Chatthistorik bevaras under sessionen

## 📊 Tillgänglig statistik

För alla 20 Premier League-lag (38 matcher per lag):
- **Inkast** (throw_ins): 12-35 normalt, outliers 38+
- **Frisparkar** (fouls): 5-18 normalt, outliers 19+
- **Skott** (shots): 5-22 normalt, outliers 28+

## 🚀 Installation

### 1. Klona/ladda ner projektet

```bash
cd "Kunskapskontroll - AI2"
```

### 2. Skapa virtuell miljö (rekommenderat)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Installera dependencies

```bash
pip install -r requirements.txt
```

### 4. Sätt OpenAI API-nyckel

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="din-api-nyckel-här"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=din-api-nyckel-här
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="din-api-nyckel-här"
```

Du kan skaffa en API-nyckel på: https://platform.openai.com/api-keys

## 🏃 Kör appen

```bash
streamlit run app.py
```

Appen öppnas automatiskt i din webbläsare på `http://localhost:8501`.

## 💬 Exempelfrågor

Här är några frågor du kan ställa:

1. **Analys av ett lag:**
   - "Hur ser Arsenals inkaststatistik ut de senaste 10 matcherna?"
   - "Analysera Liverpools statistik och förklara varför medelvärdet kan vara missvisande"
   - "Vilka outlier-matcher har Chelsea haft för frisparkar?"

2. **Jämförelse mellan lag:**
   - "Jämför Liverpool och Manchester City's skott de senaste 15 matcherna"
   - "Jämför West Ham och Everton - vilket lag har mest stabila värden?"

3. **Outlier-fokus:**
   - "Hur påverkar extremmatcher Newcastles genomsnittliga inkast?"
   - "Finns det några tydliga outliers i Manchester Uniteds skottstatistik?"

4. **Pedagogiska frågor:**
   - "Förklara skillnaden mellan medelvärde och median för Burnleys inkast"
   - "Vilka lag finns tillgängliga i databasen?"

## 📁 Projektstruktur

```
Kunskapskontroll - AI2/
├── app.py                  # Huvudfil - Streamlit app
├── src/
│   ├── __init__.py
│   ├── data.py             # Mockdata-generering och inläsning
│   ├── stats.py            # Outlier-logik och robust statistik
│   ├── tools.py            # Tool-funktioner för OpenAI
│   └── llm.py              # OpenAI API wrapper med tool calling
├── data/
│   └── mock_pl_stats.csv   # Genererad mockdata (skapas automatiskt)
├── requirements.txt
└── README.md
```

## 🤖 AI-arkitektur

### System Prompt
AI:n är konfigurerad som en fotbollsstatistik-analytiker som:
1. Alltid använder verktyg för att hämta faktiska siffror
2. Aldrig gissar statistik
3. Fokuserar på outliers och robust statistik
4. Förklarar pedagogiskt på svenska

### Tool Functions
AI:n har tillgång till följande verktyg:

| Verktyg | Beskrivning |
|---------|-------------|
| `get_team_summary` | Hämtar statistiksammanfattning för ett lags senaste N matcher |
| `compare_teams` | Jämför statistik mellan två lag |
| `get_outlier_matches` | Hämtar detaljerad info om outlier-matcher |
| `get_available_teams_list` | Listar alla tillgängliga lag |

### Modell
Använder `gpt-4o-mini` som standard (kan enkelt ändras i `src/llm.py`).

## 🔬 Outlier-metoder

### IQR-metoden
Värden utanför `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` klassas som outliers.

### Robust Z-score (MAD)
Baserat på median och MAD (Median Absolute Deviation):
```
z = 0.6745 × (x - median) / MAD
```
Värden med |z| > 3.5 klassas som outliers.

## 📈 Varför är detta viktigt?

Medelvärden kan vara missvisande när data innehåller extremvärden. Till exempel:

> "Arsenal har i snitt 24.3 inkast per match de senaste 10 matcherna. MEN - medianen är bara 21, och det trimmade medelvärdet är 21.5. Varför skillnaden? Jo, matchen mot Burnley stack ut med hela 42 inkast - en extrem outlier som drar upp snittet med nästan 3 inkast. Utan den matchen ligger snittet på 21.8, vilket ger en mer rättvisande bild av Arsenals normala spel."

## 🛠️ Teknisk stack

- **Streamlit** - Web-ramverk
- **OpenAI** - LLM med tool calling
- **Pandas** - Datahantering
- **NumPy** - Numeriska beräkningar
- **SciPy** - Statistiska funktioner (trimmed mean)
- **Plotly** - Interaktiva grafer

## 💰 Kostnadsestimering

Med `gpt-4o-mini`:
- Ca $0.15 per 1M input tokens
- Ca $0.60 per 1M output tokens
- Typisk fråga kostar ca $0.001-0.005

## 📝 Licens

Detta projekt är skapat för utbildningssyfte.

---

*Skapat som en del av Kunskapskontroll AI2*
