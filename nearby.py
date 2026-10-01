import requests
from math import radians, sin, cos, sqrt, atan2


# ══════════════════════════════════════════════════════════
# GEOCODING
# ══════════════════════════════════════════════════════════

def geocode_city(city_name: str):
    """Convert city/area name to latitude/longitude using Nominatim."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city_name, "format": "json", "limit": 1},
            headers={"User-Agent": "MindScan/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return (
                float(data[0]["lat"]),
                float(data[0]["lon"]),
                data[0]["display_name"],
            )
        return None, None, None
    except Exception as e:
        print(f"Geocode error: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════
# DISTANCE
# ══════════════════════════════════════════════════════════

def haversine(lat1, lng1, lat2, lng2):
    """Calculate distance in meters between two coordinates."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    )
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ══════════════════════════════════════════════════════════
# NEARBY PLACES
# ══════════════════════════════════════════════════════════

def get_nearby_places(
    lat: float,
    lng: float,
    search_type: str = "both",
    radius: int = 5000,
):
    """
    Search hospitals, clinics and doctors using OpenStreetMap
    Overpass API. Never raises — returns [] on total failure.
    """
    filters = {
        "hospital": '["amenity"="hospital"]',
        "doctor":   '["amenity"~"doctors|clinic"]',
        "both":     '["amenity"~"hospital|clinic|doctors"]',
    }
    f = filters.get(search_type, filters["both"])

    # NOTE: `out center;` is the valid Overpass QL.
    # `out center tags;` is INVALID and makes Overpass reject the query.
    query = f"""
    [out:json][timeout:30];
    (
      node{f}(around:{radius},{lat},{lng});
      way{f}(around:{radius},{lat},{lng});
    );
    out center;
    """

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]

    last_error = None

    for server in servers:
        try:
            print(f"[Overpass] trying {server}")
            r = requests.post(
                server,
                data=query,
                headers={
                    "User-Agent": "MindScan/1.0 (mental-health-resource-finder)"
                },
                timeout=40,
            )

            if r.status_code != 200:
                last_error = f"HTTP {r.status_code} from {server}"
                print(f"[Overpass] {last_error}")
                continue

            try:
                data = r.json()
            except ValueError:
                last_error = f"Invalid JSON from {server}"
                print(f"[Overpass] {last_error}")
                continue

            places = []

            for el in data.get("elements", []):
                tags = el.get("tags", {}) or {}
                name = tags.get("name")
                if not name:
                    continue

                if el["type"] == "node":
                    p_lat = el.get("lat")
                    p_lng = el.get("lon")
                else:
                    center = el.get("center", {}) or {}
                    p_lat = center.get("lat")
                    p_lng = center.get("lon")

                if p_lat is None or p_lng is None:
                    continue

                try:
                    dist = haversine(lat, lng, float(p_lat), float(p_lng))
                except Exception:
                    continue

                street = tags.get("addr:street", "")
                city = tags.get("addr:city", "")
                full = tags.get("addr:full", "")
                address = (
                    full
                    or ", ".join(x for x in [street, city] if x)
                    or "Address not listed"
                )

                amenity = tags.get("amenity", "facility")
                place_type = amenity.replace("_", " ").title()

                places.append({
                    "name":    name,
                    "address": address,
                    "phone":   tags.get("phone") or tags.get("contact:phone") or "",
                    "type":    place_type,
                    "lat":     float(p_lat),
                    "lng":     float(p_lng),
                    "dist_m":  int(dist),
                    "dist_km": round(dist / 1000, 1),
                })

            places.sort(key=lambda x: x["dist_m"])

            unique, seen = [], set()
            for p in places:
                key = (
                    p["name"].lower().strip(),
                    round(p["lat"], 4),
                    round(p["lng"], 4),
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(p)

            print(f"[Overpass] {len(unique)} places from {server}")
            return unique[:15]

        except requests.Timeout:
            last_error = f"Timeout from {server}"
            print(f"[Overpass] {last_error}")
            continue
        except Exception as e:
            last_error = f"{server} → {e}"
            print(f"[Overpass] {last_error}")
            continue

    print(f"[Overpass] all servers failed. Last error: {last_error}")
    return []


# ══════════════════════════════════════════════════════════
# LEAFLET MAP
# ══════════════════════════════════════════════════════════

def build_map_html(
    lat: float,
    lng: float,
    places: list,
    selected_idx: int = None,
) -> str:

    markers_js = ""

    for i, p in enumerate(places):
        is_sel = (i == selected_idx)
        color = "#a78bfa" if is_sel else "#38bdf8"
        marker_radius = 11 if is_sel else 8
        open_popup = ".openPopup()" if is_sel else ""

        phone_html = ""
        if p["phone"]:
            phone_html = (
                "<br>"
                "<span style='font-size:12px;color:#555'>"
                "📞 " + p["phone"] + "</span>"
            )

        popup = (
            "<div style='font-family:sans-serif;min-width:180px'>"
            "<strong style='color:#1a1a2e'>" + p["name"] + "</strong>"
            "<br>"
            "<span style='font-size:12px;color:#555'>🏷️ " + p["type"] + "</span>"
            "<br>"
            "<span style='font-size:12px;color:#555'>📍 " + p["address"] + "</span>"
            "<br>"
            "<span style='font-size:12px;color:#7c3aed'>🚶 " + str(p["dist_km"]) + " km away</span>"
            + phone_html +
            "</div>"
        )

        markers_js += f"""
        L.circleMarker(
            [{p['lat']}, {p['lng']}],
            {{
                radius: {marker_radius},
                fillColor: "{color}",
                color: "#ffffff",
                weight: 2,
                opacity: 1,
                fillOpacity: 0.95
            }}
        )
        .addTo(map)
        .bindPopup(`{popup}`)
        {open_popup};
        """

    route_js = ""
    if selected_idx is not None and selected_idx < len(places):
        sel = places[selected_idx]
        route_js = f"""
        L.polyline(
            [[{lat},{lng}], [{sel['lat']},{sel['lng']}]],
            {{
                color: '#a78bfa',
                weight: 3,
                dashArray: '10 6',
                opacity: 0.75
            }}
        ).addTo(map);
        """

    empty_message = ""
    if not places:
        empty_message = """
        <div style="
            position:absolute;
            top:16px;
            left:50%;
            transform:translateX(-50%);
            z-index:1000;
            background:rgba(7,6,26,0.92);
            color:white;
            padding:10px 16px;
            border-radius:12px;
            font-family:sans-serif;
            font-size:13px;
            border:1px solid rgba(255,255,255,0.12);
        ">
            🔍 No places found yet — press a search button
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #07061a;
            }}
            #map {{
                width: 100%;
                height: 420px;
                border-radius: 16px;
            }}
            .leaflet-popup-content-wrapper {{
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.3);
            }}
            .leaflet-popup-tip {{
                background: #ffffff;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        {empty_message}
        <script>
            var map = L.map('map', {{
                center: [{lat},{lng}],
                zoom: 14
            }});

            L.tileLayer(
                'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                {{
                    attribution: '© OpenStreetMap © CARTO',
                    maxZoom: 19,
                    subdomains: 'abcd'
                }}
            ).addTo(map);

            L.circleMarker(
                [{lat},{lng}],
                {{
                    radius: 13,
                    fillColor: "#7c3aed",
                    color: "#fff",
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 1
                }}
            )
            .addTo(map)
            .bindPopup("<strong>📍 You are here</strong>");

            {markers_js}
            {route_js}

            setTimeout(function() {{ map.invalidateSize(); }}, 300);
        </script>
    </body>
    </html>
    """


# ══════════════════════════════════════════════════════════
# BROWSER LOCATION DETECTOR
# ══════════════════════════════════════════════════════════

def build_location_detector_html() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                background: transparent;
                font-family: 'DM Sans', sans-serif;
                padding: 0;
            }
            .btn {
                width: 100%;
                padding: 14px 20px;
                background: linear-gradient(135deg, #7c3aed, #4f46e5);
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 15px;
                font-weight: 700;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                letter-spacing: 0.03em;
                transition: opacity 0.2s;
            }
            .btn:hover { opacity: 0.88; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }
            #msg {
                margin-top: 10px;
                font-size: 13px;
                color: rgba(255,255,255,0.5);
                text-align: center;
                min-height: 20px;
            }
            #coords {
                margin-top: 6px;
                font-size: 12px;
                color: #a78bfa;
                text-align: center;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <button class="btn" id="locBtn" onclick="detect()">
            📍 Allow Location Access
        </button>
        <div id="msg">
            Tap to detect your current location automatically
        </div>
        <div id="coords"></div>
        <script>
            function detect() {
                var btn = document.getElementById('locBtn');
                var msg = document.getElementById('msg');
                var coords = document.getElementById('coords');

                btn.disabled = true;
                btn.innerHTML = '⏳ Detecting...';
                msg.innerText = 'Please allow location access in your browser popup...';

                if (!navigator.geolocation) {
                    msg.innerText = '❌ Geolocation not supported. Use manual input below.';
                    btn.disabled = false;
                    btn.innerHTML = '📍 Allow Location Access';
                    return;
                }

                navigator.geolocation.getCurrentPosition(
                    function(pos) {
                        var lat = pos.coords.latitude.toFixed(6);
                        var lng = pos.coords.longitude.toFixed(6);

                        btn.innerHTML = '✅ Location Detected!';
                        msg.innerText = 'Location found! Copy the coordinates below.';
                        coords.innerText = 'Lat: ' + lat + '  ·  Lng: ' + lng;
                    },
                    function(err) {
                        btn.disabled = false;
                        btn.innerHTML = '📍 Allow Location Access';

                        if (err.code === 1) {
                            msg.innerText = '❌ Permission denied. Use manual input below.';
                        } else if (err.code === 2) {
                            msg.innerText = '❌ Location unavailable. Use manual input below.';
                        } else {
                            msg.innerText = '❌ Timeout. Use manual input below.';
                        }
                    },
                    { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
                );
            }
        </script>
    </body>
    </html>
    """
