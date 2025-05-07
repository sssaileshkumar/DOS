import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import random
import os

def preprocess_and_select_features(data_file='C:/Users/saile/OneDrive/Desktop/0final_dos/revised_kddcup_dataset.csv', sample_size=50000):
    try:
        print("🚀 Starting model training process...")

        # Load dataset
        print("📂 Loading dataset...")
        if sample_size:
            total_rows = sum(1 for line in open(data_file)) - 1
            if total_rows > sample_size:
                skip = lambda i: i > 0 and random.random() > sample_size / total_rows
                df = pd.read_csv(data_file, skiprows=skip)
                print(f"🔢 Using sample of {len(df)} records")
            else:
                df = pd.read_csv(data_file)
                print(f"🔢 Using full dataset ({len(df)} records)")
        else:
            df = pd.read_csv(data_file)
            print(f"🔢 Using full dataset ({len(df)} records)")

        print("\n📊 Class Distribution:")
        print(df['result'].value_counts())

        # Prepare features and target
        print("🎯 Preparing target variable...")
        X = df.drop(columns=['result'])
        y = df['result'].apply(lambda x: 0 if x == 'normal.' else 1)

        # Encode categorical features
        print("🔠 Encoding categorical features...")
        encoders = {}
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            print(f"   - Encoded {col}")

        # Feature selection
        print("🔍 Starting feature selection with RFE...")
        start_time = time.time()
        temp_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        rfe = RFE(temp_model, n_features_to_select=10, step=5)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()
        print(f"✅ Feature selection completed in {time.time() - start_time:.2f}s")
        print(f"🏆 Top 10 features: {selected_features}")

        X_selected = X[selected_features]

        # Scale features
        print("⚖️ Scaling selected features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Final model
        print("🤖 Training final model...")
        final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # Model Accuracy
        print("\n📈 Model Evaluation:")
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc * 100:.2f}%")
        print("\n📄 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-validation score
        print("🔁 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(final_model, X_scaled, y, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("model/confusion_matrix.png")
        plt.close()

        # Feature importances
        importances = final_model.feature_importances_
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=selected_features)
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("model/feature_importances.png")
        plt.close()

        # Save model artifacts
        print("💾 Saving model artifacts...")
        os.makedirs('model', exist_ok=True)
        joblib.dump(final_model, 'model/rf_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(encoders, 'model/encoders.pkl')
        joblib.dump(selected_features, 'model/features.pkl')

        print("🎉 All artifacts and graphs saved to 'model/'")
        return selected_features

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    selected_features = preprocess_and_select_features(sample_size=10000)
    if selected_features:
        print("\n✨ Model training successful! ✨")
        print(f"🔑 Selected features: {selected_features}")
    else:
        print("\n❌ Model training failed")
