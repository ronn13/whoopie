import pandas as pd
import json

df = pd.read_csv('data/raw/data_aviation.csv')

def map_event_to_adrep(event_str, narrative_str=''):
    # Combine events + narrative for keyword matching
    combined = ' '.join([
        event_str.upper() if isinstance(event_str, str) else '',
        narrative_str.upper() if isinstance(narrative_str, str) else '',
    ])

    if not combined.strip():
        return 'UNK'

    # Priority mapping based on keywords (events take priority, narrative catches the rest)
    if 'CFIT' in combined or 'CFTT' in combined:
        return 'CFIT'
    elif ('NMAC' in combined or 'AIRBORNE CONFLICT' in combined
          or 'NEAR MID-AIR' in combined or 'NEAR MIDAIR' in combined
          or 'NEAR MISS' in combined or 'NEAR-MISS' in combined
          or 'NEAR COLLISION' in combined or 'TCAS' in combined
          or 'RESOLUTION ADVISORY' in combined or 'TRAFFIC ADVISORY' in combined
          or 'TRAFFIC ALERT' in combined):
        return 'MAC'
    elif 'GROUND CONFLICT' in combined or 'GROUND INCURSION' in combined:
        return 'GCOL'
    elif 'RUNWAY EXCURSION' in combined:
        return 'RE'
    elif 'LOSS OF AIRCRAFT CONTROL' in combined or 'LOSS OF CONTROL' in combined:
        return 'LOC-I'
    elif 'UNAUTHORIZED FLIGHT' in combined or 'AIRSPACE VIOLATION' in combined or 'UAS' in combined:
        return 'SEC'
    elif 'WEATHER' in combined or 'TURBULENCE' in combined:
        return 'TURB'
    elif 'ATC ISSUE' in combined or 'CLEARANCE' in combined:
        return 'ATM'
    elif 'ALTITUDE OVERSHOOT' in combined or 'UNDERSHOOT' in combined:
        return 'USOS'
    else:
        return 'OTHR'

df['adrep_category'] = df.apply(
    lambda row: map_event_to_adrep(row['events'], row['narrative_1']), axis=1
)

print("Mapped categories count:")
print(df['adrep_category'].value_counts())

df.to_csv('data/reference/labeled_hard_cases.csv', index=False)
print("\\nSaved labeled dataset to data/reference/labeled_hard_cases.csv")
