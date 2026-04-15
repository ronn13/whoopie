import pandas as pd
import json

df = pd.read_csv('data/raw/data_aviation.csv')

def map_event_to_adrep(event_str):
    if not isinstance(event_str, str):
        return 'UNK'
    event_str = event_str.upper()
    
    # Priority mapping based on keywords
    if 'CFIT' in event_str or 'CFTT' in event_str:
        return 'CFIT'
    elif 'NMAC' in event_str or 'AIRBORNE CONFLICT' in event_str:
        return 'MAC'
    elif 'GROUND CONFLICT' in event_str or 'GROUND INCURSION' in event_str:
        return 'GCOL' # Ground Collision
    elif 'RUNWAY EXCURSION' in event_str:
        return 'RE'
    elif 'LOSS OF AIRCRAFT CONTROL' in event_str:
        return 'LOC-I'
    elif 'UNAUTHORIZED FLIGHT' in event_str or 'AIRSPACE VIOLATION' in event_str or 'UAS' in event_str:
        return 'SEC' # Security related / Unauthorized UAS
    elif 'WEATHER' in event_str or 'TURBULENCE' in event_str:
        return 'TURB'
    elif 'ATC ISSUE' in event_str or 'CLEARANCE' in event_str:
        return 'ATM' # ATM/COM related
    elif 'ALTITUDE OVERSHOOT' in event_str or 'UNDERSHOOT' in event_str:
        return 'USOS'
    else:
        return 'OTHR'

df['adrep_category'] = df['events'].apply(map_event_to_adrep)

print("Mapped categories count:")
print(df['adrep_category'].value_counts())

df.to_csv('data/reference/labeled_hard_cases.csv', index=False)
print("\\nSaved labeled dataset to data/reference/labeled_hard_cases.csv")
