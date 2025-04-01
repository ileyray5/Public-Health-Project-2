import json
import pandas as pd
import re
import folium
from geopy.distance import geodesic
import googlemaps

# Loading Data
with open('location-history.json', "r") as f:
    data = json.load(f)

# Parsing Data to make easier to filter
visits = []
for entry in data:
    if "visit" in entry:
        place = entry["visit"]["topCandidate"]
        lat_lng_match = re.search(r"geo:([0-9\.\-]+),([0-9\.\-]+)", place["placeLocation"])
        if lat_lng_match:
            lat = float(lat_lng_match.group(1))
            lng = float(lat_lng_match.group(2))
            visits.append({
                "start": pd.to_datetime(entry["startTime"]),
                "end": pd.to_datetime(entry["endTime"]),
                "duration_min": (pd.to_datetime(entry["endTime"]) - pd.to_datetime(
                    entry["startTime"])).total_seconds() / 60,
                "placeID": place["placeID"],
                "lat": lat,
                "lng": lng
            })

df_visits = pd.DataFrame(visits)


# Smarter merging algorithm
# Merges to more dominant place based on time overlap and number of visits and based on distance of significant locations
def smart_merge(df_visits, place_summary, distance_threshold_m=130, min_visits=2):
    merged_clusters = []
    used = set()

    for i, place1 in place_summary.iterrows():
        if place1['placeID'] in used:
            continue
        merged_group = [place1['placeID']]
        for j, place2 in place_summary.iterrows():
            if place1['placeID'] == place2['placeID'] or place2['placeID'] in used:
                continue
            dist = geodesic((place1['lat'], place1['lng']), (place2['lat'], place2['lng'])).meters
            if dist <= distance_threshold_m:
                visits1 = df_visits[df_visits['placeID'] == place1['placeID']]
                visits2 = df_visits[df_visits['placeID'] == place2['placeID']]
                if len(visits2) < min_visits:
                    merged_group.append(place2['placeID'])
                    used.add(place2['placeID'])
                else:
                    # If both are high-duration locations, merge them anyway
                    total1 = visits1['duration_min'].sum()
                    total2 = visits2['duration_min'].sum()
                    if total1 > 700 and total2 > 700:
                        merged_group.append(place2['placeID'])
                        used.add(place2['placeID'])
                    else:
                        # Otherwise check for time overlap
                        overlap = False
                        for v1 in visits1.itertuples():
                            for v2 in visits2.itertuples():
                                latest_start = max(v1.start, v2.start)
                                earliest_end = min(v1.end, v2.end)
                                if latest_start < earliest_end:
                                    merged_group.append(place2['placeID'])
                                    used.add(place2['placeID'])
                                    overlap = True
                                    break
                            if overlap:
                                break
        merged_clusters.append(merged_group)
    return merged_clusters


# Get the center of the cluster and its info
place_summary = df_visits.groupby('placeID').agg({
    'lat': 'mean',
    'lng': 'mean',
    'duration_min': ['count', 'sum']
}).reset_index()

# Clean up the columns
place_summary.columns = ['placeID', 'lat', 'lng', 'num_visits', 'total_duration_min']

# Call the smarter merge function on the place summary info
merged_groups = smart_merge(df_visits, place_summary)

place_id_merge_map = {}
for i, group in enumerate(merged_groups):
    for pid in group:
        place_id_merge_map[pid] = f"merged_{i}"

# Apply the new merged placeID column
df_visits['merged_placeID'] = df_visits['placeID'].map(place_id_merge_map)


def collapse_visits(group):
    group = group.sort_values('start')
    merged = []
    current_start, current_end = None, None

    for visit in group.itertuples():
        if current_start is None:
            current_start, current_end = visit.start, visit.end
        elif visit.start <= current_end:  # Overlapping or continuous
            current_end = max(current_end, visit.end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = visit.start, visit.end

    # Add the final visit window
    if current_start is not None:
        merged.append((current_start, current_end))

    total_minutes = sum((end - start).total_seconds() / 60 for start, end in merged)

    return pd.Series({
        'lat': group['lat'].mean(),
        'lng': group['lng'].mean(),
        'num_visits': len(merged),
        'total_duration_min': total_minutes
    })


# Make new summary to map
merged_summary = (
    df_visits
    .drop(columns='merged_placeID')  # Drop the group column to avoid the warning
    .groupby(df_visits['merged_placeID'], group_keys=False)
    .apply(collapse_visits)
    .reset_index()
)
# Only include places visited more than 2 times
merged_summary = merged_summary[merged_summary['num_visits'] > 2]

# Google Places API key
gmaps = googlemaps.Client(key="omitted key for public repository")

# Labels
labels = []

for _, row in merged_summary.iterrows():
    latlng = (row['lat'], row['lng'])
    label = "Unknown"

    try:
        results = gmaps.places_nearby(location=latlng, radius=50)
        if results['results']:
            place = results['results'][0]
            types = place.get("types", [])
            name = place.get("name", "")

            # Try assigning label from place types
            if "university" in types or "school" in types:
                label = "University"
            elif "gym" in types:
                label = "Gym"
            elif "cafe" in types or "restaurant" in types or "bar" in types:
                label = "Restaurant"
            elif "lodging" in types or "travel_agency" in types or "establishment" in types and "inn" in name.lower():
                label = "Hotel"
            elif "locality" in types or "sublocality" in types or "premise" in types:
                label = "Residential"
            else:
                if any(char.isdigit() for char in name) and any(word in name.lower() for word in ["st", "road", "rd", "blvd", "dr", "avenue", "ave", "street", "lane", "ln", "pike", "way", "highway", "hwy", "bus"]):
                    label = "Residential"
                else:
                    label = name
    except Exception as e:
        print(f"Error at {latlng}: {e}")
        label = "Error"

    labels.append(label)

merged_summary['label'] = labels
# Assign home to place with the longest duration time
home_index = merged_summary['total_duration_min'].idxmax()
merged_summary.loc[home_index, 'label'] = "Home"
# Colors
label_colors = {
    "University": "darkred",
    "Home": "purple",
    "Gym": "cadetblue",
    "Restaurant": "blue",
    "Hotel": "beige",
    "Residential": "green",
    "Error": "gray",
    "Unknown": "gray"
}

# Creating the Map

# Base map centered around the average lat/lng
m = folium.Map(location=[38.0293, -78.4767], zoom_start=14)

# adding marker for each cluster
for _, row in merged_summary.iterrows():
    hours = int(row['total_duration_min'] // 60)
    minutes = int(row['total_duration_min'] % 60)
    label = row['label']
    color = label_colors.get(label, "gray")
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=(
            f"<b>Label:</b> {label}<br>"
            f"<b>Visits:</b> {int(row['num_visits'])}<br>"
            f"<b> Total Time:</b> {hours} hr {minutes} min"
        ),
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)

# Save the map
m.save('significant_locations_map.html')
print("map has saved")
