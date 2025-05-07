import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import time
import random
import os

def preprocess_and_select_features(data_file='C:/Users/saile/OneDrive/Desktop/0final_dos/revised_kddcup_dataset.csv', sample_size=50000):
    try:
        print("🚀 Starting model training process...")
        
        # Load data with sampling for large datasets
        print("📂 Loading dataset...")
        if sample_size:
            total_rows = sum(1 for line in open(data_file)) - 1  # Count rows excluding header
            if total_rows > sample_size:
                skip = lambda i: i>0 and random.random() > sample_size/total_rows
                df = pd.read_csv(data_file, skiprows=skip)
                print(f"🔢 Using sample of {len(df)} records (target: {sample_size})")
            else:
                df = pd.read_csv(data_file)
                print(f"🔢 Using full dataset ({len(df)} records)")
        else:
            df = pd.read_csv(data_file)
            print(f"🔢 Using full dataset ({len(df)} records)")

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

        # Feature selection with RFE
        print("🔍 Starting feature selection with RFE...")
        start_time = time.time()
        
        # Use smaller model for faster RFE
        temp_model = RandomForestClassifier(n_estimators=30, max_depth=5, 
                                         random_state=42, n_jobs=-1)
        rfe = RFE(temp_model, n_features_to_select=10, step=5)
        rfe.fit(X, y)  # Use non-scaled data for RFE
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        print(f"✅ Feature selection completed in {time.time()-start_time:.2f}s")
        print(f"🏆 Top 10 features: {selected_features}")

        # Prepare final training data with selected features
        X_selected = X[selected_features]
        
        # Scale only the selected features
        print("⚖️ Scaling selected features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Train final model
        print("🤖 Training final model...")
        final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(X_scaled, y)
        print("🎉 Final model trained!")

        # Save artifacts
        print("💾 Saving model artifacts...")
        os.makedirs('model', exist_ok=True)
        
        joblib.dump(final_model, 'model/rf_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(encoders, 'model/encoders.pkl')
        joblib.dump(selected_features, 'model/features.pkl')
        
        print("💿 All artifacts saved to 'model' directory")
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