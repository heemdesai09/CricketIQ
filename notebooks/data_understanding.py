import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
matches = pd.read_csv("archive/matches_updated_ipl_upto_2025.csv")

deliveries = pd.read_csv("archive/deliveries_updated_ipl_upto_2025.csv")

# Show first rows
print(matches.head())

print(deliveries.head())

print("\nMatches Dataset Shape:")
print(matches.shape)

print("\nDeliveries Dataset Shape:")
print(deliveries.shape)

print("\nMatches Columns:\n")
print(matches.columns)

print("\nDeliveries Columns:\n")
print(deliveries.columns)

print("\nMissing Values in Matches:\n")
print(matches.isnull().sum())

print("\nMissing Values in Deliveries:\n")
print(deliveries.isnull().sum())

print("\nMatches Statistics:\n")
print(matches.describe())

print("\nDeliveries Statistics:\n")
print(deliveries.describe())

plt.figure(figsize=(12,6))

matches['season'].value_counts().sort_index().plot(kind='bar')

plt.title("IPL Matches Per Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")

plt.xticks(rotation=45)

plt.show()


plt.figure(figsize=(12,6))

matches['winner'].value_counts().head(10).plot(kind='bar')

plt.title("Top Winning Teams in IPL")
plt.xlabel("Teams")
plt.ylabel("Wins")

plt.xticks(rotation=45)

plt.show()

toss_match_win = matches[matches['toss_winner'] == matches['winner']]

toss_win_percentage = (len(toss_match_win) / len(matches)) * 100

print("\nToss Winner Also Won Match:")
print(f"{toss_win_percentage:.2f}%")


plt.figure(figsize=(12,6))

matches['venue'].value_counts().head(10).plot(kind='bar')

plt.title("Top IPL Venues by Matches")
plt.xlabel("Venue")
plt.ylabel("Number of Matches")

plt.xticks(rotation=90)

plt.show()

top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)

print(top_batsmen)

plt.figure(figsize=(12,6))

top_batsmen.plot(kind='bar')

plt.title("Top 10 Run Scorers in IPL")
plt.xlabel("Batsman")
plt.ylabel("Total Runs")

plt.xticks(rotation=45)

plt.show()

wickets = deliveries.dropna(subset=['player_dismissed'])

top_bowlers = wickets['bowler'].value_counts().head(10)

print(top_bowlers)

plt.figure(figsize=(12,6))

top_bowlers.plot(kind='bar')

plt.title("Top Wicket Takers in IPL")
plt.xlabel("Bowler")
plt.ylabel("Wickets")

plt.xticks(rotation=45)

plt.show()

team_wins = matches['winner'].value_counts().head(10)

plt.figure(figsize=(12,6))

team_wins.plot(kind='bar')

plt.title("Most Successful IPL Teams")
plt.xlabel("Teams")
plt.ylabel("Wins")

plt.xticks(rotation=45)

plt.show()

bat_first_wins = matches['winner_runs'].notnull().sum()

chasing_wins = matches['winner_wickets'].notnull().sum()

labels = ['Bat First Wins', 'Chasing Wins']

sizes = [bat_first_wins, chasing_wins]

plt.figure(figsize=(8,8))

plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.title("Bat First vs Chasing Wins")

plt.show()

top_venues = matches['venue'].value_counts().head(10)

plt.figure(figsize=(12,6))

top_venues.plot(kind='bar')

plt.title("Top IPL Venues")
plt.xlabel("Venue")
plt.ylabel("Matches Hosted")

plt.xticks(rotation=90)

plt.show()

player_consistency = deliveries.groupby('batsman')['batsman_runs'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))

player_consistency.plot(kind='bar')

plt.title("Most Consistent Batsmen")
plt.xlabel("Batsman")
plt.ylabel("Average Runs Per Ball")

plt.xticks(rotation=45)

plt.show()

