#1) IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


#2) LOAD DATA
df = pd.read_csv(r"C:\Users\asus\Downloads\GeneralEsportData.csv")

print("Shape:", df.shape)
print(df.head())


# 3) BASIC EDA
print("\n===== INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== STATISTICS =====")
print(df.describe())


#4) DATA CLEANING
df.columns = df.columns.str.strip().str.lower()

df = df.drop_duplicates()

df["percentoffline"] = df["percentoffline"].fillna(df["percentoffline"].median())



#5) KPI SUMMARY
print("\n===== KPI SUMMARY =====")
print("Total Games:", df["game"].nunique())
print("Total Earnings:", df["totalearnings"].sum())
print("Average Earnings:", df["totalearnings"].mean())
print("Top Game:", df.loc[df["totalearnings"].idxmax(), "game"])


#6) LINE CHART
year_trend = df.groupby("releasedate")["totalearnings"].sum().sort_index()
plt.figure()
sns.lineplot(x=year_trend.index, y=year_trend.values, marker="o")
plt.title("Earnings Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Total Earnings")
plt.show()


#7) BAR CHART
top_games = df.sort_values("totalearnings", ascending=False).head(10)

plt.figure()
sns.barplot(x="totalearnings", y="game", data=top_games)
plt.title("Top 10 Games by Earnings")
plt.show()


# 8) PIE CHARTS
top_genres = df["genre"].value_counts().head(5)

plt.figure()
plt.pie(top_genres, labels=top_genres.index, autopct="%1.1f%%")
plt.title("Top Genres Distribution")
plt.show()





# 9) SCATTER PLOTS
plt.figure()
sns.scatterplot(x="totalplayers", y="totalearnings", data=df)
plt.title("Players vs Earnings")
plt.show()

plt.figure()
sns.scatterplot(x="totaltournaments", y="totalearnings", data=df)
plt.title("Tournaments vs Earnings")
plt.show()


# 10) BOX PLOT
plt.figure()
sns.boxplot(x=df["totalearnings"])
plt.title("Boxplot of Earnings (Outliers)")
plt.show()



#12) CORRELATION HEATMAP
plt.figure()
corr = df[["totalearnings","totalplayers","totaltournaments"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()





#19) AREA PLOT (TREND)
year_trend = df.groupby("releasedate")["totalearnings"].sum().sort_index()

plt.figure()
year_trend.plot(kind="area")
plt.title("Area Plot of Earnings Over Years")
plt.xlabel("Year")
plt.ylabel("Earnings")
plt.show()


#20) STACKED BAR CHART
genre_year = df.pivot_table(
    values="totalearnings",
    index="releasedate",
    columns="genre",
    aggfunc="sum"
).fillna(0)

genre_year.tail(10).plot(kind="bar", stacked=True)
plt.title("Stacked Earnings by Genre (Recent Years)")
plt.show()



#24) TOP vs REST COMPARISON
df["category"] = np.where(
    df["totalearnings"] > df["totalearnings"].quantile(0.9),
    "Top 10%",
    "Others"
)

plt.figure()
sns.boxplot(x="category", y="totalearnings", data=df)
plt.title("Top 10% vs Others Earnings")
plt.show()

# Pearson Correlation
corr_val, p_val = stats.pearsonr(df["totalplayers"], df["totalearnings"])

print("\n===== STATISTICAL TEST =====")
print("Correlation:", corr_val)
print("P-value:", p_val)

# T-Test
high = df[df["totalearnings"] > df["totalearnings"].median()]["totalplayers"]
low = df[df["totalearnings"] <= df["totalearnings"].median()]["totalplayers"]

t_stat, p_val2 = stats.ttest_ind(high, low)

print("\nT-Test:")
print("T-stat:", t_stat)
print("P-value:", p_val2)


# 14) MACHINE LEARNING
X = df[["totalplayers","totaltournaments"]]
y = df["totalearnings"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))


# 15) FINAL INSIGHTS
print("""
--> Games with more players generate higher earnings
--> Tournaments strongly influence revenue
--> Few games dominate esports earnings
--> Data shows skewness and outliers
--> Strong correlation between players and earnings
--> Model predicts earnings with reasonable accuracy
""")
