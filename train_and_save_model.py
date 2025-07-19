import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load data
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'y':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target column
target_encoder = LabelEncoder()
df['y'] = target_encoder.fit_transform(df['y'])  # yes=1, no=0

# Features and target
X = df.drop('y', axis=1)
y = df['y']

# Train the model
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X, y)

# Save model and encoders
with open('model_decision_tree.pkl', 'wb') as f:
    pickle.dump((model, label_encoders, target_encoder, X.columns.tolist()), f)

print("✅ Model trained and saved as model_decision_tree.pkl")
