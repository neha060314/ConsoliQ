import json, requests, time

bad = [
    (18.6792, 72.2252),
    (20.7601, 72.4789),
    (13.0874, 80.3028),
]

headers = {"User-Agent": "Lorri-Freight-Clustering/1.0 (logistics@lorri.in)"}

for lat, lng in bad:
    time.sleep(1.2)
    resp = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lng, "format": "json", "addressdetails": 1, "zoom": 5},
        headers=headers,
        timeout=10,
    )
    data = resp.json()
    print(f"\n({lat}, {lng})")
    print(f"  display_name : {data.get('display_name', 'N/A')}")
    print(f"  address      : {data.get('address', {})}")