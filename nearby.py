import os
import time
import requests
from functools import lru_cache
from math import radians, sin, cos, sqrt, atan2


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

USER_AGENT = "MindScan/1.0 (mental-health-resource-finder)"

_NOMINATIM_LAST = {"t": 0.0}


_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _nominatim_throttle():
    """Enforce Nominatim's 1 req/sec rule."""
    elapsed = time.time() - _NOMINATIM_LAST["t"]
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _NOMINATIM_LAST["t"] = time.time()


# ══════════════════════════════════════════════════════════
# GEOCODING (Nominatim)
# ══════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def geocode_city(city_name: str):
    """
    Convert city/area name → (lat, lng, display_name) using Nominatim.
    Returns (None, None, None) on failure. Cached per-process.
    """
    if not city_name or not city_name.strip():
        return None, None, None

    try:
        _nominatim_throttle()

        r = _session.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q":      city_name.strip(),
                "format": "jsonv2",
                "limit":  1,
                "addressdetails": 0,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        if data:
            return (
                float(data[0]["lat"]),
                float(data[0]["lon"]),
                data[0].get("display_name", city_name),
            )
        return None, None, None

    except requests.HTTPError as e:
        print(f"[Nominatim] HTTP error: {e}")
        return None, None, None
    except Exception as e:
        print(f"[Nominatim] error: {e}")
        return None, None, None


def reverse_geocode(lat: float, lng: float):
    """Optional: lat/lng → human-readable address."""
    try:
        _nominatim_throttle()
        r = _session.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "jsonv2"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("display_name", "")
    except Exception as e:
        print(f"[Nominatim reverse] error: {e}")
        return ""


# ══════════════════════════════════════════════════════════
# DISTANCE
# ══════════════════════════════════════════════════════════

def haversine(lat1, lng1, lat2, lng2):
    """Great-circle distance in meters."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    )
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ══════════════════════════════════════════════════════════
# NEARBY PLACES (Overpass API)
# ══════════════════════════════════════════════════════════


_OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# OSM tags for healthcare facilities
_FILTERS = {
    # Hospitals only (including psychiatric hospitals)
    "hospital": '["amenity"="hospital"]',

    # Clinics + doctors + psychiatric clinics
    "doctor":   '["amenity"~"^(clinic|doctors)$"]',

    # Everything healthcare-related
    "both":     '["amenity"~"^(hospital|clinic|doctors)$"]',
}


def get_nearby_places(
    lat: float,
    lng: float,
    search_type: str = "both",
    radius: int = 5000,
):
    """
    Search healthcare facilities via OpenStreetMap Overpass API.
    Never raises. Returns [] on total failure.

    search_type: "hospital" | "doctor" | "both"
    radius:      meters (max ~50000 recommended)
    """
    f = _FILTERS.get(search_type, _FILTERS["both"])

    # NOTE: `out center;` is the correct Overpass QL.
    # `out center tags;` is INVALID and rejects the entire query.
    query = f"""
    [out:json][timeout:30];
    (
      node{f}(around:{radius},{lat},{lng});
      way{f}(around:{radius},{lat},{lng});
    );
    out center;
    """

    last_error = None

    for server in _OVERPASS_SERVERS:
        try:
            print(f"[Overpass] trying {server}")
            r = _session.post(
                server,
                data=query,
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
                continue

            places = _parse_overpass(data, lat, lng)
            places = _dedupe(places)
            print(f"[Overpass] {len(places)} places from {server}")
            return places[:15]

        except requests.Timeout:
            last_error = f"Timeout from {server}"
            continue
        except Exception as e:
            last_error = f"{server} → {e}"
            continue

    print(f"[Overpass] all servers failed. Last: {last_error}")
    return []


def _parse_overpass(data, user_lat, user_lng):
    """Turn raw Overpass JSON into clean place dicts."""
    places = []

    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue

        # Coordinates
        if el["type"] == "node":
            p_lat, p_lng = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {}) or {}
            p_lat, p_lng = c.get("lat"), c.get("lon")

        if p_lat is None or p_lng is None:
            continue

        try:
            dist = haversine(user_lat, user_lng, float(p_lat), float(p_lng))
        except Exception:
            continue

        # Address assembly (best-effort)
        addr_bits = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:suburb", ""),
            tags.get("addr:city", ""),
        ]
        address = tags.get("addr:full") or ", ".join(b for b in addr_bits if b)
        if not address:
            address = "Address not listed"

        amenity = tags.get("amenity", "facility")
        place_type = amenity.replace("_", " ").title()

        places.append({
            "name":     name,
            "address":  address,
            "phone":    tags.get("phone") or tags.get("contact:phone") or "",
            "website":  tags.get("website") or tags.get("contact:website") or "",
            "type":     place_type,
            "lat":      float(p_lat),
            "lng":      float(p_lng),
            "dist_m":   int(dist),
            "dist_km":  round(dist / 1000, 1),
            "emergency": tags.get("emergency") == "yes",
        })

    places.sort(key=lambda x: x["dist_m"])
    return places


def _dedupe(places):
    """Remove duplicates by name + rounded coords."""
    seen, unique = set(), []
    for p in places:
        key = (
            p["name"].lower().strip(),
            round(p["lat"], 4),
            round(p["lng"], 4),
        )
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ══════════════════════════════════════════════════════════
# LEAFLET MAP RENDERER (OpenStreetMap tiles)
# ══════════════════════════════════════════════════════════

def build_map_html(lat, lng, places, selected_idx=None):
    """
    Render a Leaflet map using OpenStreetMap data.
    Two tile layers available: Dark Matter (default) and Standard OSM.
    """

    markers_js = ""

    for i, p in enumerate(places):
        is_sel = (i == selected_idx)
        color = "#a78bfa" if is_sel else "#38bdf8"
        r_size = 11 if is_sel else 8
        open_popup = ".openPopup()" if is_sel else ""

        phone_html = (
            f"<br><span style='font-size:12px;color:#555'>📞 {p['phone']}</span>"
            if p.get("phone") else ""
        )
        web_html = (
            f"<br><a href='{p['website']}' target='_blank' "
            f"style='font-size:12px;color:#7c3aed'>🌐 Website</a>"
            if p.get("website") else ""
        )
        emg_html = (
            "<br><span style='font-size:11px;color:#ef4444;font-weight:700'>"
            "🚨 Emergency</span>"
            if p.get("emergency") else ""
        )

        popup = (
            "<div style='font-family:sans-serif;min-width:190px'>"
            f"<strong style='color:#1a1a2e'>{p['name']}</strong>"
            f"<br><span style='font-size:12px;color:#555'>🏷️ {p['type']}</span>"
            f"<br><span style='font-size:12px;color:#555'>📍 {p['address']}</span>"
            f"<br><span style='font-size:12px;color:#7c3aed'>"
            f"🚶 {p['dist_km']} km away</span>"
            f"{phone_html}{web_html}{emg_html}"
            "</div>"
        )

        markers_js += f"""
        L.circleMarker([{p['lat']}, {p['lng']}], {{
            radius: {r_size},
            fillColor: "{color}",
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.95
        }})
        .addTo(map)
        .bindPopup(`{popup}`)
        {open_popup};
        """

    # Route line to selected place
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

    # Empty state message
    empty_msg = ""
    if not places:
        empty_msg = """
        <div style="
            position:absolute;top:16px;left:50%;transform:translateX(-50%);
            z-index:1000;background:rgba(7,6,26,0.92);color:white;
            padding:10px 16px;border-radius:12px;font-family:sans-serif;
            font-size:13px;border:1px solid rgba(255,255,255,0.12);">
            🔍 No places found yet — press a search button
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <link rel="stylesheet"
              href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin:0; padding:0; background:#07061a; }}
            #map {{ width:100%; height:420px; border-radius:16px; }}
            .leaflet-popup-content-wrapper {{
                background:#fff; border-radius:12px;
                box-shadow:0 4px 24px rgba(0,0,0,0.3);
            }}
            .leaflet-popup-tip {{ background:#fff; }}
            .leaflet-control-layers {{
                background:rgba(20,18,40,0.92) !important;
                color:white !important;
                border:1px solid rgba(255,255,255,0.12) !important;
                border-radius:10px !important;
            }}
            .leaflet-control-layers label {{
                color:white !important;
                font-family:sans-serif;
                font-size:12px;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        {empty_msg}
        <script>
            var map = L.map('map', {{
                center: [{lat},{lng}],
                zoom: 14,
                zoomControl: true
            }});

            // Dark theme (CartoDB Dark Matter, based on OSM data)
            var darkLayer = L.tileLayer(
                'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                {{
                    attribution:
                        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> ' +
                        '© <a href="https://carto.com/attributions">CARTO</a>',
                    maxZoom: 19,
                    subdomains: 'abcd'
                }}
            );

            // Standard OSM (light) — user can toggle
            var osmLayer = L.tileLayer(
                'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                {{
                    attribution:
                        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19,
                    subdomains: 'abc'
                }}
            );

            darkLayer.addTo(map);

            L.control.layers(
                {{ "🌙 Dark": darkLayer, "☀️ Light": osmLayer }},
                null,
                {{ position: 'topright', collapsed: true }}
            ).addTo(map);

            // "You are here" pin
            L.circleMarker([{lat},{lng}], {{
                radius: 13,
                fillColor: "#7c3aed",
                color: "#fff",
                weight: 3,
                opacity: 1,
                fillOpacity: 1
            }})
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
# BROWSER LOCATION DETECTOR (unchanged)
# ══════════════════════════════════════════════════════════

def build_location_detector_html() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: transparent; font-family: 'DM Sans', sans-serif; padding: 0; }
            .btn {
                width: 100%; padding: 14px 20px;
                background: linear-gradient(135deg, #7c3aed, #4f46e5);
                color: white; border: none; border-radius: 14px;
                font-size: 15px; font-weight: 700; cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                gap: 8px; letter-spacing: 0.03em; transition: opacity 0.2s;
            }
            .btn:hover { opacity: 0.88; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; }
            #msg { margin-top: 10px; font-size: 13px;
                   color: rgba(255,255,255,0.5);
                   text-align: center; min-height: 20px; }
            #coords { margin-top: 6px; font-size: 12px;
                      color: #a78bfa; text-align: center; font-weight: 600; }
        </style>
    </head>
    <body>
        <button class="btn" id="locBtn" onclick="detect()">
            📍 Allow Location Access
        </button>
        <div id="msg">Tap to detect your current location automatically</div>
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
