import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



df = pd.read_csv("data.csv")
df_not_null = df[df["activity"] != 0]
df_not_null.drop(["devEUI", "infrared", "infrared_and_visible", "name", "floor", "deviceName", "time", "Building"], axis=1, inplace=True)
df_not_null = df_not_null[df_not_null["activity"] <= 500]
df_not_null = df_not_null[df_not_null["co2"] <= 3500]
df_not_null = df_not_null[:1500]
print(df_not_null.describe())

grid = sns.pairplot(df_not_null, corner=True)

target_name = "co2"
features_names = ["activity", "illumination", "temperature", "tvoc"]
X = df_not_null.drop(target_name, axis=1)
y = df_not_null[target_name]
X_train, X_test, y_train, y_test = train_test_split(X[features_names], y, test_size=0.2)

regressor_with_ability = LinearRegression()
regressor_with_ability.fit(X_train, y_train)
y_pred_w_ability = regressor_with_ability.predict(X_test)
R2_w_ability = r2_score(y_test, y_pred_w_ability)

print(f"R2 score with ability: {R2_w_ability:.3f}")
print(mean_absolute_error(y_test, y_pred_w_ability), mean_squared_error(y_test, y_pred_w_ability))

fig, ax = plt.subplots()
ax.plot(y_test, y_pred_w_ability, 'o')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('Measured vs Predicted')
plt.show()