import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE / 'ml_model' / 'stud_training.csv'
MODEL_OUT = BASE / 'ml_model' / 'majormatch_rf_model.pkl'
ENC_OUT = BASE / 'ml_model' / 'majormatch_encoder.pkl'
FEATS_OUT = BASE / 'ml_model' / 'majormatch_features.pkl'


def main():
    print('Loading deduped training data...')
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=['Courses'])
    y = df['Courses']

    print(f'Training rows: {X.shape[0]}, features: {X.shape[1]}')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    print('Training RandomForestClassifier...')
    clf.fit(X, y_enc)

    print('Saving artifacts to ml_model/...')
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(le, ENC_OUT)
    joblib.dump(list(X.columns), FEATS_OUT)

    print('Saved:')
    print(f' - {MODEL_OUT}')
    print(f' - {ENC_OUT}')
    print(f' - {FEATS_OUT}')


if __name__ == '__main__':
    main()
