import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


TEAM_NAME_MAPPING = {
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "RCB": "Royal Challengers Bangalore",
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad",
}

CLEAN_BOWLING_DISMISSALS = {
    "bowled",
    "caught",
    "lbw",
    "stumped",
    "caught and bowled",
    "hit wicket",
}


def _data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "data", filename)


def standardize_teams_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    matches_columns = ["team1", "team2", "toss_winner", "winner"]
    deliveries_columns = ["batting_team", "bowling_team"]

    for column in matches_columns + deliveries_columns:
        if column in df.columns:
            df[column] = df[column].replace(TEAM_NAME_MAPPING)

    return df


@st.cache_data
def load_matches_data(path: str | None = None) -> pd.DataFrame:
    data_path = path or _data_path("matches.csv")
    df = pd.read_csv(data_path)
    return preprocess_matches(df)


@st.cache_data
def load_deliveries_data(path: str | None = None) -> pd.DataFrame | None:
    data_path = path or _data_path("deliveries.csv")
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    return preprocess_deliveries(df)


def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=["team1", "team2", "date"], inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "season" in df.columns:
        df["match_year"] = pd.to_numeric(df["season"], errors="coerce")
    else:
        df["match_year"] = df["date"].dt.year

    df = standardize_teams_all(df)

    if "player_of_match" in df.columns:
        df["player_of_match"] = df["player_of_match"].fillna("Unknown")

    if "win_by_runs" in df.columns and "win_by_wickets" in df.columns:
        win_by_runs = pd.to_numeric(df["win_by_runs"], errors="coerce").fillna(0)
        win_by_wickets = pd.to_numeric(df["win_by_wickets"], errors="coerce").fillna(0)
        df["win_margin"] = np.where(win_by_runs > 0, win_by_runs, win_by_wickets)
    else:
        df["win_margin"] = np.nan

    if "toss_winner" in df.columns and "winner" in df.columns:
        df["toss_win_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

    for column in ["umpire1", "umpire2", "umpire3"]:
        if column in df.columns:
            df.drop(columns=column, inplace=True)

    return df


def preprocess_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    if "batter" in df.columns and "batsman" not in df.columns:
        rename_map["batter"] = "batsman"
    if "ball_number" in df.columns and "ball" not in df.columns:
        rename_map["ball_number"] = "ball"
    if "matchId" in df.columns and "match_id" not in df.columns:
        rename_map["matchId"] = "match_id"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required = ["match_id", "batting_team", "bowling_team", "batsman", "bowler"]
    existing_required = [column for column in required if column in df.columns]
    if existing_required:
        df.dropna(subset=existing_required, inplace=True)

    df = standardize_teams_all(df)

    if "batsman" in df.columns:
        df["batsman"] = df["batsman"].astype(str).str.strip()
        df = df[df["batsman"].ne("") & df["batsman"].ne("nan")]

    if "bowler" in df.columns:
        df["bowler"] = df["bowler"].astype(str).str.strip()
        df = df[df["bowler"].ne("") & df["bowler"].ne("nan")]

    numeric_defaults = {
        "batsman_runs": 0,
        "total_runs": 0,
        "wide_runs": 0,
        "noball_runs": 0,
        "extra_runs": 0,
    }
    for column, default in numeric_defaults.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    if "inning" not in df.columns:
        df["inning"] = 1

    if "ball" not in df.columns:
        df["ball"] = 0

    if "player_dismissed" not in df.columns:
        df["player_dismissed"] = np.nan

    if "dismissal_kind" not in df.columns:
        df["dismissal_kind"] = np.nan

    df["is_legal_ball"] = 1
    if "wide_runs" in df.columns:
        df.loc[df["wide_runs"] > 0, "is_legal_ball"] = 0
    if "noball_runs" in df.columns:
        df.loc[df["noball_runs"] > 0, "is_legal_ball"] = 0

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return preprocess_matches(df)


def total_matches_by_team(df: pd.DataFrame) -> pd.DataFrame:
    teams = pd.concat([df["team1"], df["team2"]], axis=0)
    return teams.value_counts().rename_axis("team").reset_index(name="matches")


def total_wins_by_team(df: pd.DataFrame) -> pd.DataFrame:
    return df["winner"].value_counts().rename_axis("team").reset_index(name="wins")


def win_percentage(df: pd.DataFrame) -> pd.DataFrame:
    matches = total_matches_by_team(df)
    wins = total_wins_by_team(df)
    merged = matches.merge(wins, on="team", how="left").fillna(0)
    merged["win_percentage"] = (merged["wins"] / merged["matches"]) * 100
    return merged.sort_values("win_percentage", ascending=False)


def plot_matches_bar(df: pd.DataFrame):
    matches = total_matches_by_team(df)
    fig = px.bar(matches, x="team", y="matches", title="Total Matches Played by Each Team")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


def plot_wins_bar(df: pd.DataFrame):
    wins = total_wins_by_team(df)
    fig = px.bar(wins, x="team", y="wins", title="Total Wins by Each Team")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


def plot_win_percentage_pie(df: pd.DataFrame):
    wp = win_percentage(df)
    fig = px.pie(wp, names="team", values="win_percentage", title="Win Percentage of Each Team")
    fig.update_layout(template="plotly_white")
    return fig


def plot_toss_impact(df: pd.DataFrame):
    if "toss_win_match_win" not in df.columns:
        return go.Figure()
    agg = df.groupby("toss_win_match_win").size().reset_index(name="count")
    agg["label"] = agg["toss_win_match_win"].map({1: "Toss Winner also Match Winner", 0: "Toss Winner Lost Match"})
    fig = px.pie(agg, names="label", values="count", title="Toss Decision Impact (Toss win vs Match win)")
    fig.update_layout(template="plotly_white")
    return fig


def plot_venue_advantage(df: pd.DataFrame, top_n_venues: int = 6):
    if "venue" not in df.columns:
        return go.Figure()
    venue_counts = df["venue"].value_counts().nlargest(top_n_venues).index
    sub = df[df["venue"].isin(venue_counts)]
    grp = sub.groupby(["venue", "winner"]).size().reset_index(name="wins")
    fig = px.bar(grp, x="venue", y="wins", color="winner", title="Venue Advantage: Wins by Team at Top Venues")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


def player_of_match_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "player_of_match" not in df.columns:
        return pd.DataFrame(columns=["player", "awards"])
    return df["player_of_match"].value_counts().rename_axis("player").reset_index(name="awards")


def plot_player_of_match(df: pd.DataFrame, top_n: int = 15):
    pom = player_of_match_counts(df).head(top_n)
    fig = px.bar(pom, x="player", y="awards", title=f"Top {top_n} Player of the Match Awards")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


def season_winners_trend(df: pd.DataFrame):
    if "match_year" not in df.columns:
        return go.Figure(), pd.DataFrame()
    grp = df.groupby(["match_year", "winner"]).size().reset_index(name="wins")
    idx = grp.groupby("match_year")["wins"].idxmax()
    winners = grp.loc[idx].sort_values("match_year")
    fig = px.bar(winners, x="match_year", y="wins", color="winner", title="Season-wise Winners Trend (top team by wins per season)")
    fig.update_layout(template="plotly_white")
    return fig, winners


def top_venues(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "venue" not in df.columns:
        return pd.DataFrame()
    return df["venue"].value_counts().rename_axis("venue").reset_index(name="count").head(top_n)


def plot_top_venues(df: pd.DataFrame, top_n: int = 10):
    venues = top_venues(df, top_n)
    fig = px.bar(venues, x="venue", y="count", title=f"Top {top_n} Venues Hosting Matches")
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45)
    return fig


def _safe_divide(numerator, denominator):
    denominator_series = pd.Series(denominator)
    result = pd.Series(numerator).astype(float) / denominator_series.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@st.cache_data
def compute_player_stats(deliveries_df: pd.DataFrame | None, matches_df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    if deliveries_df is None or deliveries_df.empty:
        return None

    batting = deliveries_df.groupby("batsman").agg(
        total_runs=("batsman_runs", "sum"),
        balls_faced=("is_legal_ball", "sum"),
        innings=("match_id", "nunique"),
        fours=("batsman_runs", lambda s: int((s == 4).sum())),
        sixes=("batsman_runs", lambda s: int((s == 6).sum())),
    ).reset_index().rename(columns={"batsman": "player"})

    dismissals = deliveries_df.dropna(subset=["player_dismissed"]).groupby("player_dismissed").size().reset_index(name="dismissals")
    dismissals.rename(columns={"player_dismissed": "player"}, inplace=True)
    batting = batting.merge(dismissals, on="player", how="left").fillna({"dismissals": 0})
    batting["dismissals"] = batting["dismissals"].astype(int)
    batting["batting_avg"] = np.where(batting["dismissals"] > 0, batting["total_runs"] / batting["dismissals"], batting["total_runs"])
    batting["strike_rate"] = np.where(batting["balls_faced"] > 0, (batting["total_runs"] / batting["balls_faced"]) * 100, 0)
    batting = batting.sort_values("total_runs", ascending=False).reset_index(drop=True)

    bowling = deliveries_df.groupby("bowler").agg(
        legal_balls=("is_legal_ball", "sum"),
        runs_conceded=("total_runs", "sum"),
    ).reset_index().rename(columns={"bowler": "player"})

    wicket_df = deliveries_df[
        deliveries_df["dismissal_kind"].isin(CLEAN_BOWLING_DISMISSALS) & deliveries_df["player_dismissed"].notna()
    ]
    wickets = wicket_df.groupby("bowler").size().reset_index(name="wickets").rename(columns={"bowler": "player"})
    bowling = bowling.merge(wickets, on="player", how="left").fillna({"wickets": 0})
    bowling["wickets"] = bowling["wickets"].astype(int)
    bowling["overs"] = bowling["legal_balls"] / 6.0
    bowling["economy"] = np.where(bowling["overs"] > 0, bowling["runs_conceded"] / bowling["overs"], 0)
    bowling["bowling_avg"] = np.where(bowling["wickets"] > 0, bowling["runs_conceded"] / bowling["wickets"], 0)
    bowling = bowling.sort_values("wickets", ascending=False).reset_index(drop=True)

    pom = player_of_match_counts(matches_df).rename(columns={"player": "player", "awards": "pom_awards"})
    all_players = pd.DataFrame({
        "player": sorted(set(batting["player"]).union(set(bowling["player"])).union(set(pom["player"])))
    })

    profile = all_players.merge(batting, on="player", how="left").merge(bowling, on="player", how="left").merge(pom, on="player", how="left")
    numeric_cols = [column for column in profile.columns if column != "player"]
    profile[numeric_cols] = profile[numeric_cols].fillna(0)

    for column in ["total_runs", "balls_faced", "innings", "fours", "sixes", "dismissals", "legal_balls", "runs_conceded", "wickets", "pom_awards"]:
        if column in profile.columns:
            profile[column] = profile[column].astype(int)

    for column in ["batting_avg", "strike_rate", "overs", "economy", "bowling_avg"]:
        if column in profile.columns:
            profile[column] = profile[column].astype(float)

    return {
        "batting": batting,
        "bowling": bowling,
        "profile": profile.sort_values("player").reset_index(drop=True),
    }


def get_all_teams(matches_df: pd.DataFrame) -> list[str]:
    return sorted(pd.concat([matches_df["team1"], matches_df["team2"]]).dropna().unique().tolist())


def get_all_venues(matches_df: pd.DataFrame) -> list[str]:
    return sorted(matches_df["venue"].dropna().unique().tolist())


def get_all_players(player_stats: dict[str, pd.DataFrame] | None, matches_df: pd.DataFrame) -> list[str]:
    players = set(matches_df.get("player_of_match", pd.Series(dtype=str)).dropna().tolist())
    if player_stats is not None:
        players.update(player_stats["profile"]["player"].tolist())
    players.discard("Unknown")
    return sorted(players)


@st.cache_data
def get_team_players(deliveries_df: pd.DataFrame | None) -> dict[str, list[str]]:
    if deliveries_df is None or deliveries_df.empty:
        return {}

    required_columns = {"batting_team", "bowling_team", "batsman", "bowler"}
    if not required_columns.issubset(deliveries_df.columns):
        return {}

    batting_df = deliveries_df[["batting_team", "batsman"]].copy()
    bowling_df = deliveries_df[["bowling_team", "bowler"]].copy()

    batting_df["batting_team"] = batting_df["batting_team"].replace(TEAM_NAME_MAPPING)
    bowling_df["bowling_team"] = bowling_df["bowling_team"].replace(TEAM_NAME_MAPPING)

    batting_df = batting_df.dropna(subset=["batting_team", "batsman"])
    bowling_df = bowling_df.dropna(subset=["bowling_team", "bowler"])

    batting_df["batsman"] = batting_df["batsman"].astype(str).str.strip()
    bowling_df["bowler"] = bowling_df["bowler"].astype(str).str.strip()

    batting_df = batting_df[batting_df["batsman"].ne("") & batting_df["batsman"].ne("nan")]
    bowling_df = bowling_df[bowling_df["bowler"].ne("") & bowling_df["bowler"].ne("nan")]

    teams = sorted(
        set(batting_df["batting_team"].dropna().unique()).union(
            set(bowling_df["bowling_team"].dropna().unique())
        )
    )

    team_players: dict[str, list[str]] = {}
    for team in teams:
        batters = batting_df.loc[batting_df["batting_team"] == team, "batsman"].tolist()
        bowlers = bowling_df.loc[bowling_df["bowling_team"] == team, "bowler"].tolist()
        team_players[team] = sorted(set(batters).union(set(bowlers)))

    return team_players


def get_head_to_head_history(matches_df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    h2h = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ].copy()

    columns = [column for column in ["date", "season", "venue", "team1", "team2", "winner", "win_by_runs", "win_by_wickets", "player_of_match"] if column in h2h.columns]
    h2h = h2h[columns].sort_values("date", ascending=False)
    if "date" in h2h.columns:
        h2h["date"] = pd.to_datetime(h2h["date"], errors="coerce").dt.date
    return h2h.reset_index(drop=True)


def _overall_win_rate(matches_df: pd.DataFrame, team: str) -> float:
    played = matches_df[(matches_df["team1"] == team) | (matches_df["team2"] == team)]
    if played.empty:
        return 0.5
    wins = (played["winner"] == team).sum()
    return wins / len(played)


def _venue_win_rate(matches_df: pd.DataFrame, team: str, venue: str, overall_rate: float) -> float:
    venue_matches = matches_df[
        (matches_df["venue"] == venue) &
        ((matches_df["team1"] == team) | (matches_df["team2"] == team))
    ]
    if venue_matches.empty:
        return overall_rate
    wins = (venue_matches["winner"] == team).sum()
    return wins / len(venue_matches)


def _head_to_head_factor(matches_df: pd.DataFrame, team1: str, team2: str) -> float:
    h2h = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ]
    if h2h.empty:
        return 0.5
    return (h2h["winner"] == team1).sum() / len(h2h)


def _toss_factor(matches_df: pd.DataFrame, team: str, won_toss: bool) -> float:
    if not won_toss:
        return 0.45
    toss_matches = matches_df[matches_df["toss_winner"] == team]
    if toss_matches.empty:
        return 0.5
    return (toss_matches["winner"] == team).sum() / len(toss_matches)


def _player_strength_factor(
    player_stats: dict[str, pd.DataFrame] | None,
    team1_players: list[str],
    team2_players: list[str],
) -> tuple[float, float, pd.DataFrame]:
    if player_stats is None or (not team1_players and not team2_players):
        empty_scores = pd.DataFrame(columns=["player", "player_score"])
        return 0.5, 0.5, empty_scores

    profile = player_stats["profile"].copy()

    max_runs = max(profile["total_runs"].max(), 1)
    max_wickets = max(profile["wickets"].max(), 1)
    max_pom = max(profile["pom_awards"].max(), 1)

    profile["batting_score"] = (profile["total_runs"] / max_runs) * 0.5 + (profile["strike_rate"] / 200.0) * 0.3
    profile["bowling_score"] = (profile["wickets"] / max_wickets) * 0.5 + (1 - (profile["economy"] / 20.0).clip(lower=0, upper=1)) * 0.3
    profile["pom_score"] = (profile["pom_awards"] / max_pom) * 0.2
    profile["player_score"] = profile["batting_score"] + profile["bowling_score"] + profile["pom_score"]
    score_map = profile.set_index("player")["player_score"]

    team1_scores = score_map.reindex(team1_players).dropna()
    team2_scores = score_map.reindex(team2_players).dropna()

    if team1_scores.empty and team2_scores.empty:
        return 0.5, 0.5, profile[["player", "player_score"]]

    team1_avg = float(team1_scores.mean()) if not team1_scores.empty else 0.0
    team2_avg = float(team2_scores.mean()) if not team2_scores.empty else 0.0
    total = team1_avg + team2_avg
    if total <= 0:
        return 0.5, 0.5, profile[["player", "player_score"]]

    team1_factor = team1_avg / total
    return team1_factor, 1 - team1_factor, profile[["player", "player_score"]]


def compute_win_probability(
    matches_df: pd.DataFrame,
    player_stats: dict[str, pd.DataFrame] | None,
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    team1_players: list[str] | None = None,
    team2_players: list[str] | None = None,
) -> dict[str, object]:
    del toss_decision

    team1_players = team1_players or []
    team2_players = team2_players or []

    overall_t1 = _overall_win_rate(matches_df, team1)
    overall_t2 = _overall_win_rate(matches_df, team2)

    venue_t1 = _venue_win_rate(matches_df, team1, venue, overall_t1)
    venue_t2 = _venue_win_rate(matches_df, team2, venue, overall_t2)

    h2h_t1 = _head_to_head_factor(matches_df, team1, team2)
    h2h_t2 = 1 - h2h_t1

    toss_t1 = _toss_factor(matches_df, team1, toss_winner == team1)
    toss_t2 = _toss_factor(matches_df, team2, toss_winner == team2)

    player_t1, player_t2, player_score_table = _player_strength_factor(player_stats, team1_players, team2_players)

    weights = {
        "overall_win_rate": 0.20,
        "venue_win_rate": 0.25,
        "head_to_head": 0.20,
        "toss_factor": 0.15,
        "player_strength": 0.20,
    }

    factor_values = {
        "overall_win_rate": {team1: overall_t1, team2: overall_t2},
        "venue_win_rate": {team1: venue_t1, team2: venue_t2},
        "head_to_head": {team1: h2h_t1, team2: h2h_t2},
        "toss_factor": {team1: toss_t1, team2: toss_t2},
        "player_strength": {team1: player_t1, team2: player_t2},
    }

    score_t1 = sum(weights[key] * factor_values[key][team1] for key in weights)
    score_t2 = sum(weights[key] * factor_values[key][team2] for key in weights)
    total_score = score_t1 + score_t2

    if total_score <= 0:
        prob_t1 = 50.0
    else:
        prob_t1 = (score_t1 / total_score) * 100
    prob_t2 = 100 - prob_t1

    breakdown = pd.DataFrame([
        {
            "factor": "Overall Win Rate",
            "team1_value": factor_values["overall_win_rate"][team1],
            "team2_value": factor_values["overall_win_rate"][team2],
            "weight": weights["overall_win_rate"],
        },
        {
            "factor": "Venue Win Rate",
            "team1_value": factor_values["venue_win_rate"][team1],
            "team2_value": factor_values["venue_win_rate"][team2],
            "weight": weights["venue_win_rate"],
        },
        {
            "factor": "Head-to-Head",
            "team1_value": factor_values["head_to_head"][team1],
            "team2_value": factor_values["head_to_head"][team2],
            "weight": weights["head_to_head"],
        },
        {
            "factor": "Toss Factor",
            "team1_value": factor_values["toss_factor"][team1],
            "team2_value": factor_values["toss_factor"][team2],
            "weight": weights["toss_factor"],
        },
        {
            "factor": "Player Strength",
            "team1_value": factor_values["player_strength"][team1],
            "team2_value": factor_values["player_strength"][team2],
            "weight": weights["player_strength"],
        },
    ])

    return {
        "team1": team1,
        "team2": team2,
        "probability_team1": prob_t1,
        "probability_team2": prob_t2,
        "factor_breakdown": breakdown,
        "head_to_head_history": get_head_to_head_history(matches_df, team1, team2),
        "player_scores": player_score_table,
    }


def create_win_probability_radar(result: dict[str, object]):
    breakdown = result["factor_breakdown"]
    categories = breakdown["factor"].tolist()
    categories_closed = categories + [categories[0]]

    team1 = result["team1"]
    team2 = result["team2"]
    team1_values = breakdown["team1_value"].tolist()
    team2_values = breakdown["team2_value"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=team1_values + [team1_values[0]],
        theta=categories_closed,
        fill="toself",
        name=team1,
    ))
    fig.add_trace(go.Scatterpolar(
        r=team2_values + [team2_values[0]],
        theta=categories_closed,
        fill="toself",
        name=team2,
    ))
    fig.update_layout(
        template="plotly_white",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Factor Comparison Radar",
    )
    return fig
