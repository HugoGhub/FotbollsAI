"""
app.py - Huvudfil för Streamlit-appen.

En AI-chattbot för fotbollsstatistik som använder OpenAI API
med tool calling för att analysera outliers och robust statistik.

Kör med: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Importera våra moduler
from src.data import load_data, get_available_teams, get_available_metrics, get_team_matches
from src.llm import (
    chat_with_tools, 
    check_api_key, 
    get_example_questions, 
    format_tool_results_for_display,
    clear_cache
)
from src.stats import detect_outliers_combined


# Sidkonfiguration
st.set_page_config(
    page_title="⚽ PL Statistik Chatbot",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .outlier-highlight {
        background-color: #ffcccb;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initierar session state variabler."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "tool_results_history" not in st.session_state:
        st.session_state.tool_results_history = []
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None


def load_app_data():
    """Laddar data om den inte redan är laddad."""
    if not st.session_state.data_loaded:
        with st.spinner("Laddar matchdata..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True


def create_metric_chart(df: pd.DataFrame, metric: str, team: str) -> go.Figure:
    """Skapar ett diagram som visar metrikvärden med outliers markerade."""
    data = df[metric].values
    outliers = detect_outliers_combined(data)
    
    # Skapa figur
    fig = go.Figure()
    
    # Lägg till normala punkter
    normal_mask = ~outliers
    fig.add_trace(go.Scatter(
        x=df[normal_mask]["date"],
        y=df[normal_mask][metric],
        mode="markers+lines",
        name="Normal",
        marker=dict(color="#1f77b4", size=10),
        line=dict(color="#1f77b4", width=1)
    ))
    
    # Lägg till outlier-punkter
    if outliers.any():
        fig.add_trace(go.Scatter(
            x=df[outliers]["date"],
            y=df[outliers][metric],
            mode="markers",
            name="Outlier",
            marker=dict(color="#d62728", size=14, symbol="diamond")
        ))
    
    # Lägg till medelvärdes-linje
    mean_val = data.mean()
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Medel: {mean_val:.1f}")
    
    # Lägg till median-linje
    median_val = pd.Series(data).median()
    fig.add_hline(y=median_val, line_dash="dot", line_color="green",
                  annotation_text=f"Median: {median_val:.1f}")
    
    metric_names = {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}
    
    fig.update_layout(
        title=f"{metric_names.get(metric, metric)} - {team}",
        xaxis_title="Datum",
        yaxis_title=metric_names.get(metric, metric),
        hovermode="x unified",
        height=350
    )
    
    return fig


def create_comparison_chart(data_a: dict, data_b: dict, metric: str) -> go.Figure:
    """Skapar jämförelsediagram mellan två lag."""
    teams = [data_a["name"], data_b["name"]]
    
    fig = go.Figure()
    
    # Lägg till staplar för medel, median, trimmat medel
    stats_labels = ["Medelvärde", "Median", "Trimmat medel"]
    
    fig.add_trace(go.Bar(
        name=teams[0],
        x=stats_labels,
        y=[data_a["stats"]["mean"], data_a["stats"]["median"], data_a["stats"]["trimmed_mean"]],
        marker_color="#1f77b4"
    ))
    
    fig.add_trace(go.Bar(
        name=teams[1],
        x=stats_labels,
        y=[data_b["stats"]["mean"], data_b["stats"]["median"], data_b["stats"]["trimmed_mean"]],
        marker_color="#ff7f0e"
    ))
    
    metric_names = {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}
    
    fig.update_layout(
        title=f"Jämförelse: {metric_names.get(metric, metric)}",
        barmode="group",
        yaxis_title="Värde",
        height=350
    )
    
    return fig


def render_tool_results(tool_results: list):
    """Renderar verktygsresultat i en expander med tabeller och grafer."""
    if not tool_results:
        return
    
    formatted = format_tool_results_for_display(tool_results)
    
    for result in formatted:
        if result["type"] == "summary":
            team = result["team"]
            
            # Visa statistiktabell
            metrics_data = []
            for metric_name, metric_data in result["metrics"].items():
                stats = metric_data["stats"]
                outliers = metric_data["outlier_analysis"]
                metrics_data.append({
                    "Metric": {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}.get(metric_name, metric_name),
                    "Medelvärde": f"{stats['mean']:.1f}",
                    "Median": f"{stats['median']:.1f}",
                    "Trimmat medel": f"{stats['trimmed_mean']:.1f}",
                    "Outliers": outliers["num_combined_outliers"],
                    "Medel utan outliers": f"{outliers['mean_without_outliers']:.1f}"
                })
            
            st.markdown(f"**📊 Statistik för {team}** ({result['num_matches']} matcher)")
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            
            # Visa grafer
            if result.get("matches") and len(result["matches"]) > 0:
                df_matches = pd.DataFrame(result["matches"])
                df_matches["date"] = pd.to_datetime(df_matches["date"])
                
                # Skapa grafer för varje metric
                cols = st.columns(len(result["metrics"]))
                for idx, metric_name in enumerate(result["metrics"].keys()):
                    with cols[idx]:
                        fig = create_metric_chart(df_matches, metric_name, team)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Visa outlier-matcher
            for metric_name, metric_data in result["metrics"].items():
                outlier_matches = metric_data["outlier_analysis"].get("outlier_matches", [])
                if outlier_matches:
                    metric_label = {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}.get(metric_name, metric_name)
                    st.markdown(f"**🔴 Outlier-matcher ({metric_label}):**")
                    outlier_df = pd.DataFrame(outlier_matches)
                    if not outlier_df.empty:
                        outlier_df = outlier_df.rename(columns={
                            "date": "Datum",
                            "opponent": "Motståndare",
                            "home_away": "Hemma/Borta",
                            "value": "Värde",
                            "robust_zscore": "Z-score"
                        })
                        st.dataframe(outlier_df[["Datum", "Motståndare", "Hemma/Borta", "Värde", "Z-score"]], 
                                    use_container_width=True, hide_index=True)
        
        elif result["type"] == "comparison":
            st.markdown(f"**⚔️ Jämförelse: {result['team_a']} vs {result['team_b']}**")
            
            comparisons = result.get("comparisons", {})
            for metric_name, comp in comparisons.items():
                metric_label = {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}.get(metric_name, metric_name)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{result['team_a']}**")
                    team_a_stats = comp["team_a"]["stats"]
                    st.metric("Medelvärde", f"{team_a_stats['mean']:.1f}")
                    st.metric("Median", f"{team_a_stats['median']:.1f}")
                    st.metric("Outliers", comp["team_a"]["outliers"]["num_combined_outliers"])
                
                with col2:
                    st.markdown(f"**{result['team_b']}**")
                    team_b_stats = comp["team_b"]["stats"]
                    st.metric("Medelvärde", f"{team_b_stats['mean']:.1f}")
                    st.metric("Median", f"{team_b_stats['median']:.1f}")
                    st.metric("Outliers", comp["team_b"]["outliers"]["num_combined_outliers"])
                
                # Jämförelsediagram
                fig = create_comparison_chart(comp["team_a"], comp["team_b"], metric_name)
                st.plotly_chart(fig, use_container_width=True)
            
            # Visa insikter
            insights = result.get("insights", [])
            if insights:
                st.markdown("**💡 Insikter:**")
                for insight in insights:
                    st.info(insight)
        
        elif result["type"] == "outliers":
            st.markdown(f"**🔍 Outlier-analys för {result['team']}**")
            
            stats = result.get("stats", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Medelvärde", f"{stats.get('mean', 0):.1f}")
            with col2:
                st.metric("Median", f"{stats.get('median', 0):.1f}")
            with col3:
                st.metric("Påverkan av outliers", f"{result.get('mean_impact', 0):+.1f}")
            
            outlier_matches = result.get("outlier_matches", [])
            if outlier_matches:
                st.markdown("**Outlier-matcher:**")
                outlier_df = pd.DataFrame(outlier_matches)
                st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        elif result["type"] == "teams":
            st.markdown("**📋 Tillgängliga lag:**")
            teams = result.get("teams", [])
            cols = st.columns(4)
            for idx, team in enumerate(teams):
                with cols[idx % 4]:
                    st.write(f"• {team}")


def main():
    """Huvudfunktion för appen."""
    initialize_session_state()
    load_app_data()
    
    # Kontrollera API-nyckel
    api_available, api_message = check_api_key()
    
    # Sidebar
    with st.sidebar:
        st.title("⚽ PL Statistik")
        st.markdown("---")
        
        # API Status
        if api_available:
            st.success("🟢 OpenAI API ansluten")
        else:
            st.error("🔴 API ej konfigurerad")
        
        st.markdown("---")
        st.subheader("🎯 Snabbval")
        
        # Lagval
        teams = get_available_teams()
        team_a = st.selectbox("Lag A", teams, index=0)
        team_b = st.selectbox("Lag B (valfritt)", ["-- Ingen --"] + teams, index=0)
        
        # Antal matcher
        n_matches = st.slider("Antal matcher", min_value=5, max_value=20, value=10)
        
        # Metrics-val
        st.markdown("**Metrics att analysera:**")
        metrics = get_available_metrics()
        metric_names = {"throw_ins": "Inkast", "fouls": "Frisparkar", "shots": "Skott"}
        selected_metrics = []
        for metric in metrics:
            if st.checkbox(metric_names[metric], value=True, key=f"metric_{metric}"):
                selected_metrics.append(metric)
        
        st.markdown("---")
        
        # Snabbknappar
        if st.button("📊 Analysera Lag A", use_container_width=True):
            metrics_str = ", ".join([metric_names[m] for m in selected_metrics])
            question = f"Analysera {team_a}s statistik ({metrics_str}) för de senaste {n_matches} matcherna. Fokusera på outliers."
            st.session_state.pending_question = question
        
        if team_b != "-- Ingen --":
            if st.button("⚔️ Jämför lagen", use_container_width=True):
                question = f"Jämför {team_a} och {team_b} för de senaste {n_matches} matcherna."
                st.session_state.pending_question = question
        
        st.markdown("---")
        
        # Slumpa exempel-fråga
        if st.button("🎲 Slumpa exempel-fråga", use_container_width=True):
            examples = get_example_questions()
            st.session_state.pending_question = random.choice(examples)
        
        # Rensa chatt
        if st.button("🗑️ Rensa chatt", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tool_results_history = []
            clear_cache()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **ℹ️ Om appen**
        
        Denna chatbot analyserar Premier League-statistik med fokus på:
        - 📈 Robusta mått (median, trimmat medelvärde)
        - 🔍 Outlier-detektion (IQR & MAD)
        - 📊 Visualisering av extremvärden
        
        *AI-driven av OpenAI*
        """)
    
    # Huvudområde
    st.title("⚽ Premier League Statistik Chatbot")
    st.markdown("*Fråga mig om fotbollsstatistik - jag fokuserar på outliers och varför medelvärden kan lura dig!*")
    
    # Visa API-nyckel varning om ej tillgänglig
    if not api_available:
        st.warning(api_message)
        st.markdown("---")
    
    # Visa chatthistorik
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Visa verktygsresultat i expander för assistant-meddelanden
            if message["role"] == "assistant" and idx < len(st.session_state.tool_results_history):
                tool_results = st.session_state.tool_results_history[idx]
                if tool_results:
                    with st.expander("📊 Visa data och grafer", expanded=False):
                        render_tool_results(tool_results)
    
    # Hantera pending question från sidebar
    if "pending_question" in st.session_state:
        pending = st.session_state.pending_question
        del st.session_state.pending_question
        
        # Lägg till i meddelanden och processa
        st.session_state.messages.append({"role": "user", "content": pending})
        
        with st.chat_message("user"):
            st.markdown(pending)
        
        # Hämta AI-svar
        with st.chat_message("assistant"):
            with st.spinner("Analyserar..."):
                # Bygg konversationshistorik
                conversation = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                
                result = chat_with_tools(conversation)
                response = result["response"]
                tool_results = result.get("tool_results", [])
                
                st.markdown(response)
                
                # Visa verktygsresultat
                if tool_results:
                    with st.expander("📊 Visa data och grafer", expanded=True):
                        render_tool_results(tool_results)
        
        # Spara till historik
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.tool_results_history.append(tool_results)
        
        # Fyll på tool_results_history för att matcha messages
        while len(st.session_state.tool_results_history) < len(st.session_state.messages):
            st.session_state.tool_results_history.insert(0, [])
        
        st.rerun()
    
    # Chattinput
    if prompt := st.chat_input("Ställ en fråga om Premier League-statistik..."):
        # Lägg till user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Hämta AI-svar
        with st.chat_message("assistant"):
            with st.spinner("Analyserar..."):
                # Bygg konversationshistorik
                conversation = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                
                result = chat_with_tools(conversation)
                response = result["response"]
                tool_results = result.get("tool_results", [])
                
                st.markdown(response)
                
                # Visa verktygsresultat
                if tool_results:
                    with st.expander("📊 Visa data och grafer", expanded=True):
                        render_tool_results(tool_results)
        
        # Spara till historik
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.tool_results_history.append(tool_results)
        
        # Fyll på tool_results_history för att matcha messages
        while len(st.session_state.tool_results_history) < len(st.session_state.messages):
            st.session_state.tool_results_history.insert(0, [])
        
        st.rerun()


if __name__ == "__main__":
    main()
