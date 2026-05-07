import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder





st.set_page_config(
    page_title="CricketIQ",
    page_icon="🏏",
    layout="wide"
)

# Load model and encoder
model = joblib.load("models/random_forest_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.markdown("""
<style>

.big-title {
    font-size: 55px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0px;
}

.sub-text {
    font-size: 22px;
    color: #475569;
    margin-top: -10px;
    margin-bottom: 30px;
}

.metric-box {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.metric-title {
    font-size: 18px;
    color: #64748B;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #2563EB;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="big-title"> CricketIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">'
    'IPL Match Prediction & Analytics Platform'
    '</div>',
    unsafe_allow_html=True
)

# Metrics Row
col1, col2, col3 = st.columns(3)

st.markdown(
    """
    <style>
    div.block-container {
        padding-top: 2rem;
    }

    .stColumns {
        margin-bottom: -20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">83%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Algorithm</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-title">Records</div>
        <div class="metric-value">278K+</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.info(
    '''
    CricketIQ is an IPL Match Prediction &
    Analytics Platform using Machine Learning.
    '''
)

st.sidebar.success("Random Forest Accuracy: 83%")

st.sidebar.write("Algorithm: Random Forest")

st.sidebar.write("Clustering: K-Means")

st.sidebar.write("Dataset Size: 278,205+ records")

st.sidebar.write("Developed by Heem Desai")




teams = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad"
]

venues = [
    "Wankhede Stadium",
    "M Chinnaswamy Stadium",
    "Eden Gardens"
]

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox(
        "Select Team 1",
        teams
    )

with col2:
    team2 = st.selectbox(
        "Select Team 2",
        teams
    )

col3, col4 = st.columns(2)

with col3:
    venue = st.selectbox(
        "Select Venue",
        venues
    )

with col4:
    toss_winner = st.selectbox(
        "Select Toss Winner",
        teams
    )

toss_decision = st.selectbox(
    "Toss Decision",
    ["bat", "field"]
)

if team1 == team2:
    st.warning("⚠️ Team 1 and Team 2 cannot be same")
    st.stop()



if st.button("Predict Winner"):

    # Input Data
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'venue': [venue],
        'toss_match_win': [1]
    })

    # Encode Input
    for column in input_data.columns:
        input_data[column] = encoders[column].transform(
            input_data[column]
        )

    # Prediction
    prediction = model.predict(input_data)

    predicted_winner = encoders['winner'].inverse_transform(
        prediction
    )[0]

    # Probability
    probabilities = model.predict_proba(input_data)

    max_probability = round(
        probabilities.max() * 100,
        1
    )

    # Result Section
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Winner",
            value=predicted_winner
        )

    with col2:
        st.metric(
            label="Win Probability",
            value=f"{max_probability}%"
        )

    st.progress(int(max_probability))

    st.markdown("---")

    st.subheader("AI Match Insight")

    insight = f"""
    {predicted_winner} has a stronger probability of winning
    based on historical IPL records, toss impact,
    venue advantage, and team performance trends.

    The Random Forest Machine Learning model predicts this
    outcome with {max_probability}% confidence.

    Teams choosing to {toss_decision} first at
    {venue} have historically shown strong match-winning patterns.
    """

    st.info(insight)

st.subheader("IPL Team Clustering Analysis")

with st.expander("View IPL Team Clustering Analysis"):

    # Load dataset
    matches = pd.read_csv(
        "archive/matches_updated_ipl_upto_2025.csv"
    )

    # Team wins
    team_wins = matches['winner'].value_counts()

    team_stats = pd.DataFrame({
        'team': team_wins.index,
        'wins': team_wins.values
    })

    # Encode
    encoder_cluster = LabelEncoder()

    team_stats['team_encoded'] = (
        encoder_cluster.fit_transform(team_stats['team'])
    )

    # KMeans
    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    team_stats['cluster'] = kmeans.fit_predict(
        team_stats[['wins']]
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        team_stats['team_encoded'],
        team_stats['wins'],
        c=team_stats['cluster'],
        cmap='viridis',
        s=200
    )

    # Labels
    for i in range(len(team_stats)):
        ax.text(
            team_stats['team_encoded'][i],
            team_stats['wins'][i],
            team_stats['team'][i],
            fontsize=8
        )

    ax.set_title("IPL Team Clustering")

    ax.set_xlabel("Team Encoded")

    ax.set_ylabel("Wins")

    ax.grid(True)

    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

st.subheader("Toss Impact Analysis")

matches = pd.read_csv(
    "archive/matches_updated_ipl_upto_2025.csv"
)

# Toss win percentage
toss_match_win = round(
    (
        matches['toss_winner'] == matches['winner']
    ).mean() * 100,
    1
)

# Counts
bat_first_wins = matches[
    matches['toss_decision'] == 'bat'
]['winner'].count()

field_first_wins = matches[
    matches['toss_decision'] == 'field'
]['winner'].count()

# TOP METRICS
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Toss → Match Win %",
        f"{toss_match_win}%"
    )

with metric2:
    st.metric(
        "Bat First Matches",
        bat_first_wins
    )

with metric3:
    st.metric(
        "Field First Matches",
        field_first_wins
    )

# PIE CHART
col1, col2, col3 = st.columns([1,2,1])

with col2:

    fig, ax = plt.subplots(figsize=(3,3))

    ax.pie(
        [bat_first_wins, field_first_wins],
        labels=['Bat First', 'Field First'],
        autopct='%1.1f%%',
        textprops={'fontsize':8}
    )

    ax.set_title(
        "Toss Decision Distribution",
        fontsize=10
    )

    st.pyplot(fig)


    st.markdown("---")

st.subheader("Head-to-Head Analysis")

matches = pd.read_csv(
    "archive/matches_updated_ipl_upto_2025.csv"
)

# Matches between selected teams
h2h_matches = matches[
    (
        (matches['team1'] == team1) &
        (matches['team2'] == team2)
    )
    |
    (
        (matches['team1'] == team2) &
        (matches['team2'] == team1)
    )
]

total_matches = len(h2h_matches)

team1_wins = len(
    h2h_matches[h2h_matches['winner'] == team1]
)

team2_wins = len(
    h2h_matches[h2h_matches['winner'] == team2]
)

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Matches",
        total_matches
    )

with col2:
    st.metric(
        f"{team1} Wins",
        team1_wins
    )

with col3:
    st.metric(
        f"{team2} Wins",
        team2_wins
    )


    st.markdown("---")

st.subheader("Team Performance Comparison")

comparison_data = pd.DataFrame({
    "Teams": [team1, team2],
    "Wins": [team1_wins, team2_wins]
})

st.bar_chart(
    comparison_data.set_index("Teams")
)




st.markdown("---")

st.subheader("Advanced Player Intelligence Engine")

deliveries = pd.read_csv(
    "archive/deliveries_updated_mens_ipl.csv"
)

matches = pd.read_csv(
    "archive/matches_updated_ipl_upto_2025.csv"
)

players = sorted(
    deliveries['batsman'].dropna().unique()
)

selected_player = st.selectbox(
    "Search Any IPL Player",
    players
)

player_data = deliveries[
    deliveries['batsman'] == selected_player
]


total_runs = player_data['batsman_runs'].sum()

balls = len(player_data)

strike_rate = round(
    (total_runs / balls) * 100,
    2
)

fours = len(
    player_data[player_data['batsman_runs'] == 4]
)

sixes = len(
    player_data[player_data['batsman_runs'] == 6]
)

matches_played = player_data['matchId'].nunique()

dismissals = player_data[
    player_data['player_dismissed'] == selected_player
]


col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Runs", total_runs)

with col2:
    st.metric("Balls", balls)

with col3:
    st.metric("SR", strike_rate)

with col4:
    st.metric("4s", fours)

with col5:
    st.metric("6s", sixes)

with col6:
    st.metric("Matches", matches_played)

    st.markdown("---")

st.subheader("Match Phase Analysis")

powerplay = player_data[
    player_data['over'] <= 5
]['batsman_runs'].sum()

middle = player_data[
    (player_data['over'] > 5) &
    (player_data['over'] <= 15)
]['batsman_runs'].sum()

death = player_data[
    player_data['over'] > 15
]['batsman_runs'].sum()

phase_df = pd.DataFrame({
    'Phase': ['Powerplay', 'Middle', 'Death'],
    'Runs': [powerplay, middle, death]
})

st.bar_chart(
    phase_df.set_index('Phase')
)


st.markdown("---")

st.subheader("Dismissal Weakness Analysis")

dismissal_types = (
    dismissals['dismissal_kind']
    .value_counts()
)

st.bar_chart(dismissal_types)

most_common_dismissal = (
    dismissal_types.idxmax()
    if len(dismissal_types) > 0
    else "Not Available"
)

st.warning(
    f"Most Common Dismissal: {most_common_dismissal}"
)


st.markdown("---")

st.subheader("Bowler Matchup Engine")

bowler_stats = (
    player_data.groupby('bowler')
    ['batsman_runs']
    .sum()
    .reset_index()
)

bowler_stats = bowler_stats.sort_values(
    by='batsman_runs',
    ascending=False
)

st.dataframe(
    bowler_stats.head(10),
    use_container_width=True
)


st.markdown("---")

st.subheader("🔥 Clutch Performance Score")

boundary_runs = (
    (fours * 4) +
    (sixes * 6)
)

clutch_score = round(
    (
        strike_rate +
        boundary_runs / 10 +
        death / 5
    ),
    1
)

st.metric(
    "Clutch Score",
    clutch_score
)



st.markdown("---")

st.subheader("AI Insights")

best_phase = phase_df.sort_values(
    by='Runs',
    ascending=False
).iloc[0]['Phase']

st.info(f"""
{selected_player} performs strongest during the {best_phase} phase.

Strike Rate:
{strike_rate}

Boundary Ability:
{fours + sixes} boundaries scored.

Weakness Detected:
Most dismissals occur via {most_common_dismissal}.

Clutch Score:
{clutch_score}
""")



