import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    compute_player_stats,
    compute_win_probability,
    create_win_probability_radar,
    get_all_teams,
    get_all_venues,
    get_team_players,
    load_deliveries_data,
    load_matches_data,
    player_of_match_counts,
    season_winners_trend,
)


BG_COLOR = "#0D1117"
SIDEBAR_GRADIENT = "linear-gradient(180deg, #161B22 0%, #111827 100%)"
CARD_COLOR = "#1C2333"
BORDER_COLOR = "#30363D"
PRIMARY = "#F97316"
PRIMARY_GRADIENT = "linear-gradient(135deg, #FB923C 0%, #F97316 45%, #EA580C 100%)"
SECONDARY = "#3B82F6"
SUCCESS = "#22C55E"
TEXT_PRIMARY = "#F0F6FC"
TEXT_MUTED = "#8B949E"
CHART_BG = "#161B22"
PLOTLY_SEQUENCE = [PRIMARY, SECONDARY, SUCCESS, "#FACC15", "#EC4899", "#14B8A6"]


st.set_page_config(page_title="IPL Analytics Dashboard", layout="wide", page_icon="🏏")


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Rajdhani:wght@600;700&display=swap');

:root {{
    --bg: {BG_COLOR};
    --card: {CARD_COLOR};
    --border: {BORDER_COLOR};
    --primary: {PRIMARY};
    --secondary: {SECONDARY};
    --success: {SUCCESS};
    --text: {TEXT_PRIMARY};
    --muted: {TEXT_MUTED};
    --chart: {CHART_BG};
}}

html, body, [class*=\"css\"] {{
    font-family: 'DM Sans', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 30%),
        radial-gradient(circle at top left, rgba(249,115,22,0.10), transparent 28%),
        var(--bg);
    color: var(--text);
}}

[data-testid=\"stAppViewContainer\"] {{
    background: transparent;
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}

section[data-testid=\"stSidebar\"] {{
    background: {SIDEBAR_GRADIENT};
    border-right: 1px solid rgba(249,115,22,0.15);
}}

section[data-testid=\"stSidebar\"] > div {{
    background: transparent;
}}

[data-testid=\"stSidebar\"] .sidebar-brand {{
    padding: 0.25rem 0 1rem 0;
}}

[data-testid=\"stSidebar\"] .sidebar-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: 0.03em;
}}

[data-testid=\"stSidebar\"] .sidebar-subtitle {{
    color: var(--muted);
    font-size: 0.95rem;
}}

[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] {{
    gap: 0.45rem;
}}

[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] label {{
    background: linear-gradient(180deg, rgba(28,35,51,0.95), rgba(18,24,37,0.95));
    border: 1px solid rgba(48,54,61,0.95);
    border-left: 3px solid transparent;
    border-radius: 999px;
    padding: 0.6rem 0.85rem;
    transition: all 0.2s ease;
}}

[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] label:hover {{
    border-color: rgba(249,115,22,0.45);
    box-shadow: 0 0 0 1px rgba(249,115,22,0.2), 0 0 18px rgba(249,115,22,0.15);
    transform: translateX(2px);
}}

[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] label[data-checked=\"true\"] {{
    border-color: rgba(249,115,22,0.5);
    border-left-color: var(--primary);
    background: linear-gradient(90deg, rgba(249,115,22,0.14), rgba(28,35,51,1) 28%);
    box-shadow: 0 0 0 1px rgba(249,115,22,0.18), 0 0 22px rgba(249,115,22,0.18);
}}

[data-testid=\"stSidebar\"] div[role=\"radiogroup\"] p {{
    color: var(--text);
    font-weight: 700;
}}

h1, h2, h3, .page-title, [data-testid=\"stMetricLabel\"] {{
    font-family: 'Rajdhani', sans-serif;
}}

.page-hero {{
    margin-bottom: 1.35rem;
}}

.page-title {{
    font-size: clamp(2.3rem, 4vw, 3.6rem);
    font-weight: 700;
    line-height: 1;
    margin: 0;
    background: linear-gradient(90deg, #FB923C 0%, #F97316 40%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.page-subtitle {{
    color: var(--muted);
    font-size: 1rem;
    margin: 0.4rem 0 0.9rem 0;
}}

.title-divider {{
    height: 2px;
    width: 140px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(249,115,22,0.95), rgba(249,115,22,0.15));
}}

.metric-card {{
    background: linear-gradient(180deg, rgba(28,35,51,0.96), rgba(19,26,38,0.96));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1.15rem;
    min-height: 132px;
    position: relative;
    overflow: hidden;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    border-color: rgba(249,115,22,0.45);
    box-shadow: 0 16px 32px rgba(0,0,0,0.3), 0 0 24px rgba(249,115,22,0.16);
}}

.metric-icon {{
    font-size: 1.5rem;
    margin-bottom: 0.2rem;
}}

.metric-label {{
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 600;
}}

.metric-value {{
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    margin-top: 0.45rem;
}}

.overview-card, .factor-card, .stat-card, div[data-testid=\"stMetric\"] {{
    background: linear-gradient(180deg, rgba(28,35,51,0.96), rgba(17,24,39,0.96));
    border: 1px solid var(--border);
    border-radius: 18px;
}}

.overview-card {{
    background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(59,130,246,0.10) 55%, rgba(28,35,51,0.95));
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}}

.overview-card ul {{
    margin: 0.75rem 0 0 1.25rem;
    color: var(--muted);
}}

.probability-card {{
    border-radius: 24px;
    padding: 1.3rem;
    color: white;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 18px 36px rgba(0,0,0,0.28);
}}

.probability-team {{
    font-size: 1.05rem;
    font-weight: 700;
}}

.probability-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.45rem 0 0.3rem 0;
}}

.probability-caption, .factor-weight, .team-filter-label, .stat-label {{
    color: var(--muted);
}}

.factor-card {{
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
}}

.factor-head, .factor-team-label {{
    display: flex;
    justify-content: space-between;
}}

.factor-head {{
    gap: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.7rem;
}}

.factor-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
}}

.factor-team-label {{
    font-size: 0.92rem;
    margin-bottom: 0.38rem;
}}

.factor-track {{
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: rgba(139,148,158,0.12);
    overflow: hidden;
}}

.factor-fill {{
    height: 100%;
    border-radius: 999px;
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}}

@keyframes shimmer {{
    0% {{ background-position: 0% 0; }}
    100% {{ background-position: 200% 0; }}
}}
.stat-card {{
    padding: 0.95rem 1rem;
    min-height: 108px;
}}

.stat-value {{
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 0.45rem;
}}

.stat-accent {{
    width: 38px;
    height: 4px;
    border-radius: 999px;
    margin-top: 0.65rem;
    background: linear-gradient(135deg, #FB923C 0%, #F97316 45%, #EA580C 100%);
}}

.section-label {{
    font-family: 'Rajdhani', sans-serif;
    color: var(--text);
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0.4rem 0 0.8rem 0;
}}

.stTabs [data-baseweb=\"tab-list\"] {{ gap: 0.6rem; }}
.stTabs [data-baseweb=\"tab\"] {{
    background: rgba(28,35,51,0.95);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text);
}}
.stTabs [aria-selected=\"true\"] {{
    background: linear-gradient(90deg, rgba(249,115,22,0.22), rgba(59,130,246,0.18));
    border-color: rgba(249,115,22,0.45);
}}

.stButton > button,
.stForm button[kind=\"primary\"] {{
    background: {PRIMARY_GRADIENT};
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.65rem 1rem;
    font-weight: 700;
}}

.stButton > button:hover,
.stForm button[kind=\"primary\"]:hover {{
    filter: brightness(0.92);
    transform: translateY(-1px);
}}

.stButton > button[kind=\"secondary\"] {{
    background: linear-gradient(180deg, rgba(28,35,51,0.96), rgba(18,24,37,0.96));
    color: var(--text);
    border: 1px solid var(--border);
}}

.stButton > button[kind=\"secondary\"]:hover {{
    border-color: rgba(249,115,22,0.45);
}}

div[data-testid=\"stMetric\"] {{
    padding: 1rem;
}}

div[data-testid=\"stMetricValue\"] {{
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
}}

label, .stSelectbox label, .stMultiSelect label, .stSlider label {{
    color: var(--text) !important;
    font-weight: 700 !important;
}}

.stSelectbox [data-baseweb=\"select\"] > div,
.stMultiSelect [data-baseweb=\"select\"] > div {{
    background: rgba(22,27,34,0.95);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 14px;
}}

.stAlert {{
    background: rgba(28,35,51,0.96);
    border: 1px solid var(--border);
    color: var(--text);
}}

[data-testid=\"stDataFrame\"] {{
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    background: rgba(22,27,34,0.98);
}}

[data-testid=\"stDataFrame\"] thead tr th {{
    background: linear-gradient(90deg, #EA580C, #F97316) !important;
    color: white !important;
    font-weight: 700 !important;
}}

[data-testid=\"stDataFrame\"] tbody tr:nth-child(odd) td {{
    background: rgba(28,35,51,0.92) !important;
    color: var(--text) !important;
}}

[data-testid=\"stDataFrame\"] tbody tr:nth-child(even) td {{
    background: rgba(22,27,34,0.96) !important;
    color: var(--text) !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_team_players(deliveries_df):
    return get_team_players(deliveries_df)


def apply_dark_figure_style(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_MUTED, family="DM Sans"),
        colorway=PLOTLY_SEQUENCE,
        title_font=dict(color=TEXT_PRIMARY, family="Rajdhani", size=22),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_PRIMARY))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(139,148,158,0.12)", zeroline=False, linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_PRIMARY))
    if title is not None:
        fig.update_layout(title=title)
    return fig


def render_header():
    st.markdown(
        """
        <div class="page-hero">
            <h1 class="page-title">IPL Analytics Dashboard</h1>
            <p class="page-subtitle">A modern, data-rich view of team performance, player impact, venue patterns, and match-winning signals from IPL 2008 to 2019.</p>
            <div class="title-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon: str, label: str, value):
    st.markdown(f"""<div class=\"metric-card\"><div class=\"metric-icon\">{icon}</div><div class=\"metric-label\">{label}</div><div class=\"metric-value\">{value}</div></div>""", unsafe_allow_html=True)


def probability_card(team: str, probability: float, color: str):
    gradient = "linear-gradient(135deg, rgba(249,115,22,0.30) 0%, rgba(249,115,22,0.95) 100%)" if color == PRIMARY else "linear-gradient(135deg, rgba(59,130,246,0.30) 0%, rgba(59,130,246,0.95) 100%)"
    st.markdown(f"""<div class=\"probability-card\" style=\"background:{gradient};\"><div class=\"probability-team\">{team}</div><div class=\"probability-value\">{probability:.1f}%</div><div class=\"probability-caption\">Estimated win probability</div></div>""", unsafe_allow_html=True)


def render_factor_breakdown(breakdown_df: pd.DataFrame, team1: str, team2: str):
    for row in breakdown_df.itertuples(index=False):
        team1_width = max(0.0, min(100.0, float(row.team1_value) * 100))
        team2_width = max(0.0, min(100.0, float(row.team2_value) * 100))
        st.markdown(f"""
            <div class=\"factor-card\">
                <div class=\"factor-head\"><span>{row.factor}</span><span class=\"factor-weight\">{int(row.weight * 100)}% weight</span></div>
                <div class=\"factor-grid\">
                    <div><div class=\"factor-team-label\"><span>{team1}</span><span>{row.team1_value:.2f}</span></div><div class=\"factor-track\"><div class=\"factor-fill\" style=\"width:{team1_width}%; background-image:linear-gradient(90deg, rgba(249,115,22,0.95), rgba(251,146,60,0.75), rgba(249,115,22,0.95));\"></div></div></div>
                    <div><div class=\"factor-team-label\"><span>{team2}</span><span>{row.team2_value:.2f}</span></div><div class=\"factor-track\"><div class=\"factor-fill\" style=\"width:{team2_width}%; background-image:linear-gradient(90deg, rgba(59,130,246,0.95), rgba(96,165,250,0.78), rgba(59,130,246,0.95));\"></div></div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_stat_card(label: str, value, accent: str = PRIMARY):
    st.markdown(f"""<div class=\"stat-card\"><div class=\"stat-label\">{label}</div><div class=\"stat-value\">{value}</div><div class=\"stat-accent\" style=\"background:{accent};\"></div></div>""", unsafe_allow_html=True)


def render_team_filter(label: str, teams_list: list[str], state_key: str) -> str:
    default_team = st.session_state.get(state_key)
    if default_team not in teams_list:
        default_team = teams_list[0] if teams_list else None
    if default_team is not None:
        st.session_state[state_key] = default_team
    st.markdown(f"<div class='team-filter-label'>{label}</div>", unsafe_allow_html=True)
    if not teams_list:
        return ""
    per_row = 4
    total_rows = math.ceil(len(teams_list) / per_row)
    for row_index in range(total_rows):
        row_teams = teams_list[row_index * per_row:(row_index + 1) * per_row]
        columns = st.columns(len(row_teams))
        for column, team_name in zip(columns, row_teams):
            button_type = "primary" if st.session_state.get(state_key) == team_name else "secondary"
            with column:
                if st.button(team_name, key=f"{state_key}_{team_name}", type=button_type, use_container_width=True):
                    st.session_state[state_key] = team_name
    return st.session_state[state_key]


matches_df = load_matches_data()
deliveries_df = load_deliveries_data()
player_stats = compute_player_stats(deliveries_df, matches_df)
team_players = load_team_players(deliveries_df) if deliveries_df is not None else {}


render_header()

total_matches = len(matches_df)
teams = get_all_teams(matches_df)
seasons = matches_df["match_year"].nunique()
venues = matches_df["venue"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("🏏", "Matches", total_matches)
with c2:
    metric_card("🏆", "Teams", len(teams))
with c3:
    metric_card("📅", "Seasons", seasons)
with c4:
    metric_card("🏟️", "Venues", venues)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("""<div class=\"sidebar-brand\"><div class=\"sidebar-title\">🏆 IPL Analytics</div><div class=\"sidebar-subtitle\">Insightful, fast, match-ready dashboards</div></div>""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["Home", "Team Performance", "Toss Analysis", "Venue Analysis", "Player Awards", "Season Trends", "Player Stats", "Win Probability"])


if page == "Home":
    st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""<div class=\"overview-card\"><strong>Explore the IPL through performance, momentum, and match context.</strong><ul><li>Team winning patterns across seasons</li><li>Toss impact on match outcomes</li><li>Venue-driven advantages and trends</li><li>Player performance snapshots and win probability signals</li></ul></div>""", unsafe_allow_html=True)
    matches_by_team = pd.concat([matches_df["team1"], matches_df["team2"]]).value_counts().reset_index()
    matches_by_team.columns = ["team", "matches"]
    fig = px.bar(matches_by_team.head(10), x="team", y="matches", color="matches", color_continuous_scale=[SECONDARY, PRIMARY], title="Top Teams by Matches Played")
    apply_dark_figure_style(fig)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Matches")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Team Performance":
    st.markdown("<div class='section-label'>Team Performance</div>", unsafe_allow_html=True)
    team = st.selectbox("Select Team", teams)
    team_matches = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)]
    matches_played = len(team_matches)
    wins = (team_matches["winner"] == team).sum()
    win_pct = (wins / matches_played) * 100 if matches_played else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches Played", matches_played)
    c2.metric("Wins", wins)
    c3.metric("Win %", f"{win_pct:.1f}%")
    wins_by_season = matches_df[matches_df["winner"] == team].groupby("match_year").size().reset_index(name="wins")
    fig = px.bar(wins_by_season, x="match_year", y="wins", title=f"{team} Wins Per Season", color_discrete_sequence=[SUCCESS])
    apply_dark_figure_style(fig)
    fig.update_xaxes(title="Season")
    fig.update_yaxes(title="Wins")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Toss Analysis":
    st.markdown("<div class='section-label'>Toss Impact</div>", unsafe_allow_html=True)
    toss_df = matches_df.copy()
    toss_df["toss_win_match"] = np.where(toss_df["toss_winner"] == toss_df["winner"], "Won Match", "Lost Match")
    toss_data = toss_df["toss_win_match"].value_counts().reset_index()
    toss_data.columns = ["Result", "Count"]
    fig = px.pie(toss_data, names="Result", values="Count", title="Does Toss Winner Win the Match?", color="Result", color_discrete_map={"Won Match": SUCCESS, "Lost Match": PRIMARY})
    apply_dark_figure_style(fig)
    fig.update_traces(textfont_color=TEXT_PRIMARY)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Venue Analysis":
    st.markdown("<div class='section-label'>Venue Advantage</div>", unsafe_allow_html=True)
    venue = st.selectbox("Select Venue", sorted(matches_df["venue"].dropna().unique()))
    venue_df = matches_df[matches_df["venue"] == venue]
    wins = venue_df["winner"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    fig = px.bar(wins.head(10), x="Wins", y="Team", orientation="h", title=f"Wins at {venue}", color="Wins", color_continuous_scale=[SECONDARY, PRIMARY])
    apply_dark_figure_style(fig)
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Player Awards":
    st.markdown("<div class='section-label'>Top Players</div>", unsafe_allow_html=True)
    pom = player_of_match_counts(matches_df)
    pom.columns = ["Player", "Awards"]
    fig = px.bar(pom.head(10), x="Awards", y="Player", orientation="h", title="Top 10 Player of the Match Winners", color="Awards", color_continuous_scale=[SECONDARY, PRIMARY])
    apply_dark_figure_style(fig)
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pom.head(20), use_container_width=True, hide_index=True)

elif page == "Season Trends":
    st.markdown("<div class='section-label'>Season Trends</div>", unsafe_allow_html=True)
    matches_per_season = matches_df.groupby("match_year").size().reset_index(name="matches")
    fig = px.line(matches_per_season, x="match_year", y="matches", markers=True, title="Matches Per Season", color_discrete_sequence=[SECONDARY])
    apply_dark_figure_style(fig)
    fig.update_traces(line_width=3, marker_size=9)
    fig.update_xaxes(title="Season")
    fig.update_yaxes(title="Matches")
    st.plotly_chart(fig, use_container_width=True)
    try:
        fig2, winners_df = season_winners_trend(matches_df)
        apply_dark_figure_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(winners_df, use_container_width=True, hide_index=True)
    except Exception:
        st.warning("Season winners data not available.")

elif page == "Player Stats":
    st.markdown("<div class='section-label'>Player Stats</div>", unsafe_allow_html=True)
    if player_stats is None:
        st.warning("`data/deliveries.csv` was not found. Add it to enable batting, bowling, and player profile analytics.")
    else:
        available_teams = sorted(team_players.keys()) if team_players else teams
        batting_tab, bowling_tab, profile_tab = st.tabs(["Batting", "Bowling", "Player Profile"])

        with batting_tab:
            selected_batting_team = render_team_filter("Batting Team Filter", available_teams, "batting_team_filter")
            batting_df = player_stats["batting"].copy()
            batting_team_players = set(team_players.get(selected_batting_team, []))
            if batting_team_players:
                batting_df = batting_df[batting_df["player"].isin(batting_team_players)].reset_index(drop=True)
            min_runs = int(batting_df["total_runs"].max()) if not batting_df.empty else 0
            min_runs_filter = st.slider("Minimum Runs", 0, min_runs, min(200, min_runs) if min_runs else 0)
            filtered_batting = batting_df[batting_df["total_runs"] >= min_runs_filter].copy()
            top_batters = batting_df.head(15).sort_values("total_runs", ascending=True)
            fig = px.bar(top_batters, x="total_runs", y="player", orientation="h", title="Top 15 Run Scorers", color="total_runs", color_continuous_scale=[SECONDARY, PRIMARY])
            apply_dark_figure_style(fig)
            fig.update_yaxes(title="")
            fig.update_xaxes(title="Runs")
            st.plotly_chart(fig, use_container_width=True)
            scatter_df = batting_df[batting_df["total_runs"] > 0].copy()
            fig = px.scatter(scatter_df, x="strike_rate", y="batting_avg", size="total_runs", hover_name="player", title="Strike Rate vs Batting Average", color="total_runs", color_continuous_scale=[SECONDARY, PRIMARY])
            apply_dark_figure_style(fig)
            fig.update_xaxes(title="Strike Rate")
            fig.update_yaxes(title="Batting Average")
            st.plotly_chart(fig, use_container_width=True)
            display_columns = ["player", "total_runs", "balls_faced", "innings", "dismissals", "batting_avg", "strike_rate", "fours", "sixes"]
            st.dataframe(filtered_batting[display_columns].round({"batting_avg": 2, "strike_rate": 2}), use_container_width=True, hide_index=True)

        with bowling_tab:
            selected_bowling_team = render_team_filter("Bowling Team Filter", available_teams, "bowling_team_filter")
            bowling_df = player_stats["bowling"].copy()
            bowling_team_players = set(team_players.get(selected_bowling_team, []))
            if bowling_team_players:
                bowling_df = bowling_df[bowling_df["player"].isin(bowling_team_players)].reset_index(drop=True)
            max_wickets = int(bowling_df["wickets"].max()) if not bowling_df.empty else 0
            min_wickets_filter = st.slider("Minimum Wickets", 0, max_wickets, min(5, max_wickets) if max_wickets else 0)
            filtered_bowling = bowling_df[bowling_df["wickets"] >= min_wickets_filter].copy()
            top_bowlers = bowling_df.head(15).sort_values("wickets", ascending=True)
            fig = px.bar(top_bowlers, x="wickets", y="player", orientation="h", title="Top 15 Wicket Takers", color="wickets", color_continuous_scale=[SECONDARY, PRIMARY])
            apply_dark_figure_style(fig)
            fig.update_yaxes(title="")
            fig.update_xaxes(title="Wickets")
            st.plotly_chart(fig, use_container_width=True)
            scatter_df = bowling_df[bowling_df["wickets"] > 0].copy()
            fig = px.scatter(scatter_df, x="economy", y="bowling_avg", size="wickets", hover_name="player", title="Economy Rate vs Bowling Average", color="wickets", color_continuous_scale=[SECONDARY, PRIMARY])
            apply_dark_figure_style(fig)
            fig.update_xaxes(title="Economy")
            fig.update_yaxes(title="Bowling Average")
            st.plotly_chart(fig, use_container_width=True)
            display_columns = ["player", "wickets", "overs", "runs_conceded", "economy", "bowling_avg"]
            st.dataframe(filtered_bowling[display_columns].round({"overs": 2, "economy": 2, "bowling_avg": 2}), use_container_width=True, hide_index=True)

        with profile_tab:
            selected_team = render_team_filter("Profile Team Filter", available_teams, "profile_team")
            team_player_options = team_players.get(selected_team, [])
            profile_df = player_stats["profile"].copy()
            if not team_player_options:
                st.info("No players found for this team in `deliveries.csv`.")
            else:
                selected_player = st.selectbox("Select Player", team_player_options, key="profile_player")
                player_row = profile_df[profile_df["player"] == selected_player].iloc[0]
                st.markdown("<div class='section-label' style='font-size:1.25rem;'>Batting Metrics</div>", unsafe_allow_html=True)
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    render_stat_card("Total Runs", int(player_row["total_runs"]))
                with c2:
                    render_stat_card("Batting Avg", f"{player_row['batting_avg']:.2f}")
                with c3:
                    render_stat_card("Strike Rate", f"{player_row['strike_rate']:.2f}", f"linear-gradient(90deg, {SECONDARY}, #60A5FA)")
                with c4:
                    render_stat_card("Fours", int(player_row["fours"]))
                with c5:
                    render_stat_card("Sixes", int(player_row["sixes"]), f"linear-gradient(90deg, {SUCCESS}, #4ADE80)")
                st.markdown("<div class='section-label' style='font-size:1.25rem;'>Bowling Metrics</div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    render_stat_card("Wickets", int(player_row["wickets"]))
                with c2:
                    render_stat_card("Economy", f"{player_row['economy']:.2f}", f"linear-gradient(90deg, {SECONDARY}, #60A5FA)")
                with c3:
                    render_stat_card("Bowling Avg", f"{player_row['bowling_avg']:.2f}")
                with c4:
                    render_stat_card("Overs Bowled", f"{player_row['overs']:.2f}", f"linear-gradient(90deg, {SUCCESS}, #4ADE80)")
                st.markdown("<div class='section-label' style='font-size:1.25rem;'>Awards</div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    render_stat_card("Player of the Match Awards", int(player_row["pom_awards"]), f"linear-gradient(90deg, {PRIMARY}, #FB923C)")

elif page == "Win Probability":
    st.markdown("<div class='section-label'>Multi-Factor Win Probability Analyzer</div>", unsafe_allow_html=True)
    team1 = st.selectbox("Team 1", teams, index=0, key="wp_team1")
    team2_options = [team for team in teams if team != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0, key="wp_team2")
    team1_player_options = team_players.get(team1, []) if deliveries_df is not None else []
    team2_player_options = team_players.get(team2, []) if deliveries_df is not None else []
    with st.form("win_probability_form"):
        venue = st.selectbox("Venue", get_all_venues(matches_df))
        toss_winner_label = st.selectbox("Toss Winner", ["Team 1", "Team 2"])
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
        team1_players = st.multiselect("Team 1 Players", options=team1_player_options, default=[player for player in st.session_state.get("wp_team1_players", []) if player in team1_player_options], max_selections=5, disabled=deliveries_df is None, help="Optional. Select up to 5 players.", key="wp_team1_players")
        team2_players = st.multiselect("Team 2 Players", options=team2_player_options, default=[player for player in st.session_state.get("wp_team2_players", []) if player in team2_player_options], max_selections=5, disabled=deliveries_df is None, help="Optional. Select up to 5 players.", key="wp_team2_players")
        submitted = st.form_submit_button("Analyze Win Probability")

    if deliveries_df is None:
        st.info("`data/deliveries.csv` is not available, so the Player Strength factor defaults to 0.5 for both teams.")

    if submitted:
        toss_winner = team1 if toss_winner_label == "Team 1" else team2
        result = compute_win_probability(matches_df=matches_df, player_stats=player_stats, team1=team1, team2=team2, venue=venue, toss_winner=toss_winner, toss_decision=toss_decision, team1_players=team1_players, team2_players=team2_players)
        c1, c2 = st.columns(2)
        with c1:
            probability_card(team1, result["probability_team1"], PRIMARY)
        with c2:
            probability_card(team2, result["probability_team2"], SECONDARY)
        st.markdown("<div class='section-label'>Factor Breakdown</div>", unsafe_allow_html=True)
        render_factor_breakdown(result["factor_breakdown"], team1, team2)
        st.markdown("<div class='section-label'>Radar Comparison</div>", unsafe_allow_html=True)
        radar_fig = create_win_probability_radar(result)
        radar_fig.update_traces(line=dict(width=3), opacity=0.72)
        if len(radar_fig.data) >= 2:
            radar_fig.data[0].update(line=dict(color=PRIMARY, width=3), fillcolor="rgba(249,115,22,0.25)")
            radar_fig.data[1].update(line=dict(color=SECONDARY, width=3), fillcolor="rgba(59,130,246,0.25)")
        radar_fig.update_layout(template="plotly_dark", paper_bgcolor=BG_COLOR, plot_bgcolor=CHART_BG, font=dict(color=TEXT_MUTED, family="DM Sans"), title_font=dict(color=TEXT_PRIMARY, family="Rajdhani", size=22), polar=dict(bgcolor=CHART_BG, radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(139,148,158,0.15)", linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_MUTED)), angularaxis=dict(gridcolor="rgba(139,148,158,0.08)", linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_PRIMARY))))
        st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("<div class='section-label'>Head-to-Head History</div>", unsafe_allow_html=True)
        h2h_df = result["head_to_head_history"]
        if h2h_df.empty:
            st.info("No historical head-to-head matches found for this matchup.")
        else:
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)


