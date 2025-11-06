"""
NBA AI Analyzer - Streamlit App
Mostra estatísticas reais de jogadores da NBA, puxa (ou simula) odds,
calcula probabilidades estimadas e Valor Esperado (EV) para markets:
 - pontos (PTS)
 - assistências (AST)
 - rebotes (REB)

Instalação:
 pip install streamlit nba_api pandas numpy requests plotly

Execução:
 streamlit run app.py
"""
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# NBA API imports
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

# -------------------------
# Config / Constants
# -------------------------
SPORT_KEY = "basketball_nba"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"

st.set_page_config(page_title="NBA AI Analyzer", layout="wide")

# -------------------------
# Helper: NBA stats via nba_api
# -------------------------
@lru_cache(maxsize=256)
def get_player_id_by_name(name: str):
    all_players = players.get_players()
    for p in all_players:
        if p["full_name"].lower() == name.lower():
            return p["id"]
    return None

def fetch_player_season_averages(player_id: int):
    """
    Busca estatísticas de carreira / última temporada e retorna médias (PTS/AST/REB).
    Atenção: nba_api tem limites; adicionamos pequenos delays quando necessário.
    """
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = career.get_data_frames()[0]
        # Pegamos a última linha (temporada mais recente no dataset)
        last = df.iloc[-1]
        return {"PTS": float(last["PTS"]), "AST": float(last["AST"]), "REB": float(last["REB"])}
    except Exception as e:
        st.write(f"Erro ao buscar stats para player_id={player_id}: {e}")
        return None

def build_stats_dict(player_names):
    stats = {}
    for name in player_names:
        pid = get_player_id_by_name(name)
        if pid is None:
            st.warning(f"Jogador não encontrado (nome exato requerido): {name}")
            continue
        with st.spinner(f"Buscando stats de {name}..."):
            s = fetch_player_season_averages(pid)
            if s:
                stats[name] = s
            time.sleep(0.6)  # gentle delay to avoid rate limits
    return stats

# -------------------------
# Helper: Odds (real via TheOddsAPI or simulated)
# -------------------------
def fetch_odds_the_odds_api(sport_key=SPORT_KEY, regions="us", markets="player_points,player_assists,player_rebounds", odds_format="decimal"):
    api_key = os.getenv("THE_ODDS_API_KEY", "")
    if not api_key:
        raise RuntimeError("THE_ODDS_API_KEY not set in environment.")
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format
    }
    url = THE_ODDS_API_URL.format(sport=sport_key)
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Erro TheOddsAPI: {r.status_code} — {r.text}")
    return r.json()

def generate_fake_odds_for_stats(stats_dict):
    """
    Gera odds simuladas a partir das médias (para POC / offline testing).
    Retorna DataFrame com colunas:
      Player, PTS_line, PTS_odds, AST_line, AST_odds, REB_line, REB_odds
    """
    rows = []
    rng = np.random.default_rng(seed=42)
    for player, s in stats_dict.items():
        pts_line = round(s["PTS"] + rng.uniform(-2.0, 2.0), 1)
        ast_line = round(s["AST"] + rng.uniform(-1.0, 1.0), 1)
        reb_line = round(s["REB"] + rng.uniform(-1.5, 1.5), 1)
        rows.append({
            "Player": player,
            "PTS_line": max(0.5, pts_line),
            "PTS_odds": round(rng.uniform(1.65, 2.5), 2),
            "AST_line": max(0.5, ast_line),
            "AST_odds": round(rng.uniform(1.65, 2.5), 2),
            "REB_line": max(0.5, reb_line),
            "REB_odds": round(rng.uniform(1.65, 2.5), 2)
        })
    return pd.DataFrame(rows)

# -------------------------
# Core: Probability & EV model
# -------------------------
def stat_to_win_prob(avg: float, line: float):
    """
    Converte diferença entre média e linha para probabilidade de 'over'.
    Usamos uma S-shaped function (logistic) para mapear diferenças a [0,1].
    Fórmula simples e transparente — substituível por modelo ML.
    """
    diff = avg - line
    # scale factor controls sensitivity; tune as needed
    scale = 1.0
    prob = 1.0 / (1.0 + np.exp(-diff / scale))
    return prob

def calculate_ev(row, stats_dict):
    """Recebe uma linha do odds_df e calcula EV para PTS/AST/REB."""
    player = row["Player"]
    s = stats_dict[player]
    p_pts = stat_to_win_prob(s["PTS"], row["PTS_line"])
    p_ast = stat_to_win_prob(s["AST"], row["AST_line"])
    p_reb = stat_to_win_prob(s["REB"], row["REB_line"])

    # Probabilidades implícitas das odds (para over)
    implied_pts = 1.0 / row["PTS_odds"]
    implied_ast = 1.0 / row["AST_odds"]
    implied_reb = 1.0 / row["REB_odds"]

    ev_pts = p_pts * row["PTS_odds"]
    ev_ast = p_ast * row["AST_odds"]
    ev_reb = p_reb * row["REB_odds"]

    return {
        "Player": player,
        "PTS_line": row["PTS_line"], "PTS_avg": s["PTS"], "PTS_odds": row["PTS_odds"],
        "P_over_prob": round(p_pts, 3), "P_impl": round(implied_pts, 3), "EV_PTS": round(ev_pts, 3),
        "AST_line": row["AST_line"], "AST_avg": s["AST"], "AST_odds": row["AST_odds"],
        "AST_over_prob": round(p_ast, 3), "AST_impl": round(implied_ast, 3), "EV_AST": round(ev_ast, 3),
        "REB_line": row["REB_line"], "REB_avg": s["REB"], "REB_odds": row["REB_odds"],
        "REB_over_prob": round(p_reb, 3), "REB_impl": round(implied_reb, 3), "EV_REB": round(ev_reb, 3)
    }

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏀 NBA AI Analyzer — Insights de Valor (PTS / AST / REB)")
st.markdown(
    "Ferramenta educativa: mostra estatísticas reais da NBA, compara com odds (reais ou simuladas) "
    "e calcula probabilidades estimadas e Valor Esperado (EV)."
)

st.sidebar.header("Configuração")
use_real_odds = st.sidebar.checkbox("Usar The Odds API (requer THE_ODDS_API_KEY)", value=False)
player_input = st.sidebar.text_area(
    "Cole nomes de jogadores (um por linha) — ex.: LeBron James",
    value="LeBron James\nStephen Curry\nGiannis Antetokounmpo\nLuka Doncic\nJayson Tatum",
    height=160
)
players_to_analyze = [p.strip() for p in player_input.splitlines() if p.strip()]

threshold_ev = st.sidebar.slider("Limite EV para destacar", 1.05, 1.0, 1.2, step=0.01)

if st.sidebar.button("Rodar análise"):
    if len(players_to_analyze) == 0:
        st.error("Informe ao menos um jogador.")
    else:
        with st.spinner("Coletando estatísticas reais (nba_api)..."):
            stats = build_stats_dict(players_to_analyze)

        if not stats:
            st.error("Nenhuma estatística carregada. Verifique os nomes (devem coincidir com o nome completo oficial).")
        else:
            st.success(f"Estatísticas coletadas para {len(stats)} jogadores.")

            # Odds: real ou simulada
            if use_real_odds:
                try:
                    with st.spinner("Buscando odds reais na The Odds API..."):
                        # NOTE: here we attempt to fetch sports-level player markets, but many providers do not expose
                        # per-player markets in a consistent way. For robust integration, map provider output to our schema.
                        raw = fetch_odds_the_odds_api()
                        st.info("Odds reais carregadas (raw). Mapeamento custom pode ser necessário.")
                        st.write(raw[:1])  # debug spot-check
                        st.warning("Integração completa com TheOddsAPI requer adaptar nomes/linhas do provedor.")
                        # For this app we fallback to simulated mapping until provider mapping is implemented.
                        odds_df = generate_fake_odds_for_stats(stats)
                except Exception as e:
                    st.error(f"Erro ao buscar odds reais: {e}\nUsando odds simuladas.")
                    odds_df = generate_fake_odds_for_stats(stats)
            else:
                odds_df = generate_fake_odds_for_stats(stats)

            st.subheader("📋 Odds (linhas e preços) — tabela")
            st.dataframe(odds_df, width=1100)

            # Compute EV table
            st.subheader("🧮 Cálculo: Probabilidades estimadas & Valor Esperado (EV)")
            results = []
            for _, r in odds_df.iterrows():
                res = calculate_ev(r, stats)
                results.append(res)
            ev_df = pd.DataFrame(results)

            # Highlight EV > threshold
            ev_df["Best_EV"] = ev_df[["EV_PTS", "EV_AST", "EV_REB"]].max(axis=1)
            ev_df_sorted = ev_df.sort_values(by="Best_EV", ascending=False)
            st.dataframe(ev_df_sorted.style.format("{:.3f}"), width=1200)

            # Show top opportunities
            top = ev_df_sorted[ev_df_sorted["Best_EV"] >= threshold_ev]
            if top.empty:
                st.info(f"Nenhuma oportunidade com EV >= {threshold_ev} encontrada.")
            else:
                st.subheader(f"🔥 Oportunidades com EV >= {threshold_ev}")
                st.dataframe(top, width=1200)

            # Charts: EV per player per market
            st.subheader("📈 Visualização — EV por jogador e mercado")
            plot_df = ev_df_sorted.melt(id_vars=["Player"], value_vars=["EV_PTS", "EV_AST", "EV_REB"],
                                        var_name="Market", value_name="EV")
            fig = px.bar(plot_df, x="Player", y="EV", color="Market", barmode="group",
                         title="EV comparado por jogador e mercado")
            st.plotly_chart(fig, use_container_width=True)

            # Small explanation box
            st.markdown("""
            **Sobre o modelo**:
            - `P_over_prob` é uma estimativa simples (função logística) baseada em diferença entre média do jogador e a linha oferecida.
            - `EV = P_over_prob * odd`. Valores de EV > 1 indicam que, em média, a aposta teria retorno positivo **segundo este modelo simplificado**.
            - Este é um *modelo educacional*: para produção, substitua a função de probabilidade por um modelo estatístico/ML treinado com histórico.
            """)
            st.success("Análise concluída ✅")