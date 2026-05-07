import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


matches = pd.read_csv(
    "archive/matches_updated_ipl_upto_2025.csv"
)


team_wins = matches['winner'].value_counts()

team_stats = pd.DataFrame({
    'team': team_wins.index,
    'wins': team_wins.values
})


encoder = LabelEncoder()

team_stats['team_encoded'] = encoder.fit_transform(
    team_stats['team']
)


kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

team_stats['cluster'] = kmeans.fit_predict(
    team_stats[['wins']]
)


print(team_stats)


# Plot clusters

plt.figure(figsize=(10, 6))

plt.scatter(
    team_stats['team_encoded'],
    team_stats['wins'],
    c=team_stats['cluster'],
    cmap='viridis',
    s=200
)

# Team labels
for i in range(len(team_stats)):
    plt.text(
        team_stats['team_encoded'][i],
        team_stats['wins'][i],
        team_stats['team'][i],
        fontsize=8
    )

plt.title("IPL Team Clustering")

plt.xlabel("Team Encoded")

plt.ylabel("Wins")

plt.grid(True)

plt.show()