'''

Purpose of this model: 
The purpose of this model is to generate the output in terms of coefficient which tells us how much that course score influences 
the probability of a candidate getting hired, after accounting for all other courses.
The bigger the number, the stronger the relationship between that course score and getting hired.

We adopted a logistic regression model as the output variable that we are looking at is a binary one where 1 (hire) and 0 (non-hired)

How to read the outputs:

============================================================
TRACK: ABC123
============================================================
Candidates in track : ??
Hired               : ?? (??%)
Not hired           : ??

Train : ?? candidates (?? hired, ?? not hired) (Following 80/20 rule)
Test  : ?? candidates (?? hired, ?? not hired)

Learned Coefficients for ABC123:

  Course                          Coefficient  Interpretation
  -----------------------------------------------------------------
  capstone_score                       1.0247  Strong — employers value this highly
  sql_score                            1.0137  Strong — employers value this highly
  python_score                         0.7707  Strong — employers value this highly
  power_bi_score                       0.5652  Strong — employers value this highly
  data_viz_score                       0.2621  Moderate — some hiring impact

'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load placement data
from pathlib import Path
DATA_PATH = Path(__file__).parent.parent / 'data' / 'placement_data.csv'
df = pd.read_csv(DATA_PATH)

# Intro
print("-" * 60)
print("SKILLBRIDGE — MODEL 1: LOGISTIC REGRESSION PER ROLE TRACK")
print("-" * 60)
print(f"\nTotal candidates : {len(df)}")
print(f"Role tracks      : {df['role_track'].unique()}")
print(f"Overall hire rate: {df['hired'].mean()*100:.1f}%")
 
# Course to jobscope matching
ALL_COURSE_COLS = [
    'sql_score', 'python_score', 'power_bi_score', 'data_viz_score',
    'excel_score', 'financial_modelling_score', 'compliance_score',
    'reporting_score', 'process_mapping_score', 'capstone_score'
]

TRACK_COURSES = {}
for track in df['role_track'].unique():
    track_df = df[df['role_track'] == track]
    # Keep only columns where this track has actual scores (not all NaN)
    relevant = [col for col in ALL_COURSE_COLS
                if track_df[col].notna().any()]
    TRACK_COURSES[track] = relevant
    print(f"{track}: {relevant}")

# Model 1
scaler  = StandardScaler()
results = {}
 
for track, courses in TRACK_COURSES.items():
    print("\n" + "-" * 60)
    print(f"TRACK: {track}")
    print("-" * 60)
 
    # Filter to this track only
    track_df = df[df['role_track'] == track].copy()
    print(f"Candidates in track : {len(track_df)}")
    print(f"Hired               : {track_df['hired'].sum()} "
          f"({track_df['hired'].mean()*100:.1f}%)")
    print(f"Not hired           : {(track_df['hired']==0).sum()}")
 
    # Skip if not enough data
    if len(track_df) < 15:
        print(f"Insufficient data for {track} — need at least 15 candidates")
        continue
 
    if track_df['hired'].nunique() < 2:
        print(f"Only one class in {track} data — cannot train logistic regression")
        continue
 
    # Features and target
    X = track_df[courses]
    y = track_df['hired']
 
    # Scale
    X_scaled = scaler.fit_transform(X)
 
    # Train test split — stratify ensures both hired and not hired in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(y_train)} candidates "
          f"({y_train.sum()} hired, {(y_train==0).sum()} not hired)")
    print(f"Test  : {len(y_test)} candidates "
          f"({y_test.sum()} hired, {(y_test==0).sum()} not hired)")
 
    # Train model — coefficients learned from this track's data only
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    #Learned coefficients
    coef_df = pd.DataFrame({
        'course'      : courses,
        'coefficient' : model.coef_[0]
    }).sort_values('coefficient', ascending=False).reset_index(drop=True)
 
    print(f"\nLearned Coefficients for {track}:")
    print(f"\n  {'Course':<30} {'Coefficient':>12}  {'Interpretation'}")
    print(f"  {'-'*65}")
    for _, row in coef_df.iterrows():
        if row['coefficient'] > 0.3:
            interp = "Strong — employers value this highly"
        elif row['coefficient'] > 0.1:
            interp = "Moderate — some hiring impact"
        elif row['coefficient'] > -0.1:
            interp = "Weak — little hiring impact"
        else:
            interp = "Negative — review this course"
        print(f"  {row['course']:<30} {row['coefficient']:>12.4f}  {interp}")
    
    # Storing result
    results[track] = {
    'model'      : model,
    'coef_df'    : coef_df,
    'y_test'     : y_test,
    'y_pred'     : y_pred,
    'courses'    : courses,
    'n'          : len(track_df),
    'hire_rate'  : track_df['hired'].mean() * 100
    }

# Summary
print("\n" + "-" * 60)
print("SUMMARY — MODEL 1 ACROSS ALL TRACKS")
print("-" * 60)
 
print(f"\n{'Track':<20} {'N':>5} {'Hire%':>7} "
      f"{'Top Course':<30} {'Weakest Course'}")
print("-" * 85)
 
for track, res in results.items():
    top    = res['coef_df'].iloc[0]['course']
    bottom = res['coef_df'].iloc[-1]['course']
    print(f"{track:<20} {res['n']:>5} {res['hire_rate']:>6.1f}% "
          f"{top:<30} {bottom}")
 
print(f"\nCurriculum Recommendations:")
print("-" * 60)
for track, res in results.items():
    top    = res['coef_df'].iloc[0]['course']
    bottom = res['coef_df'].iloc[-1]['course']
    print(f"\n  {track}:")
    print(f"    Prioritise : {top} — strongest hiring signal")
    print(f"    Review     : {bottom} — weakest hiring signal")
 
print("\n" + "-" * 60)
print("MODEL 1 COMPLETE")
print("-" * 60)