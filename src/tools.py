from langchain.tools import tool
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
import json, re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote
import pandas as pd
from datetime import datetime

# --- Pydantic validation (added) ---
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

# Small helpers (won't change behavior)
def _to_float(x, default: float = 0.0) -> float:
    try:
        if x in (None, "-", ""):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

class StationId(BaseModel):
    """BoM station id like 'IDV60901.95936'"""
    value: str = Field(...)

    @field_validator("value")
    @classmethod
    def check_format(cls, v: str) -> str:
        # Keep this permissive but catch obvious typos
        if "." not in v or len(v.split(".")) != 2:
            raise ValueError("station_id must look like 'IDxNNNNN.NNNNN'")
        prefix, suffix = v.split(".")
        if not prefix.startswith("ID"):
            raise ValueError("station_id should start with 'ID*'")
        # Suffix is often numeric; allow non-numeric if BoM uses one, but warn if empty
        if not suffix:
            raise ValueError("station_id suffix is empty")
        return v

class BBox(BaseModel):
    """Bounding box: west,south,east,north (floats)"""
    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def check_order(self) -> "BBox":
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bbox invalid: require west < east and south < north")
        return self

    @classmethod
    def parse(cls, bbox_str: str) -> "BBox":
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must have four comma-separated numbers")
        return cls(west=parts[0], south=parts[1], east=parts[2], north=parts[3])

class LatLon(BaseModel):
    lat: float
    lon: float


END_POINTS = {
    "melbourne": {
        "lat": -37.81, "lon": 144.96,
        "weather_station_id": "IDV60901.95936",
        "fuel_search_suburb": "Melbourne"
    },
    "sydney": {
        "lat": -33.86, "lon": 151.20,
        "weather_station_id": "IDN60901.94768",
        "fuel_search_suburb": "Sydney"
    }
}

WAYPOINTS = {
    "Hume_Highway": {
        "albury": {
            "lat": -36.08, "lon": 146.91,
            "weather_station_id": "IDN60901.94925",
            "fuel_search_suburb": "Albury"
        },
        "goulburn": {
            "lat": -34.75, "lon": 149.71,
            "weather_station_id": "IDN60903.94926",
            "fuel_search_suburb": "Goulburn"
        }
    },
    "Inland_Route": {
        "shepparton": {
            "lat": -36.38, "lon": 145.40,
            "weather_station_id": "IDV60801.94875",
            "fuel_search_suburb": "Shepparton"
        },
        "dubbo": {
            "lat": -32.25, "lon": 148.61,
            "weather_station_id": "IDN60801.95719",
            "fuel_search_suburb": "Dubbo"
        }
    }
}
CSV_FILE_PATH = 'freight_data_hume_highway.csv'

@tool
def get_weather_data(station_id):
    """
    Fetches the latest weather observation data from a specific BoM station.
    """
    # --- Pydantic validation (added) ---
    try:
        station = StationId(value=str(station_id)).value
    except ValidationError as e:
        return {"error": f"Invalid station_id: {e.errors()[0]['msg']}"}
    except Exception as e:
        return {"error": f"Invalid station_id: {e}"}

    url = f"http://www.bom.gov.au/fwo/{station.split('.')[0]}/{station}.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers ,timeout=10)
        response.raise_for_status()
        data = response.json()
        latest_obs = data.get("observations", {}).get("data", [])[0]
        if not latest_obs:
            return {"error": "No observation data found."}
        # Keep same keys, just coerce numbers safely
        return {
            "location": latest_obs.get("name"),
            "time_utc": latest_obs.get("aifstime_utc"),
            "temperature": _to_float(latest_obs.get("air_temp"), default=float("nan")),
            "rainfall_since_9am": _to_float(latest_obs.get("rain_trace"), default=0.0),
            "wind_speed_kmh": _to_float(latest_obs.get("wind_spd_kmh"), default=0.0),
            "wind_direction": latest_obs.get("wind_dir"),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch weather data: {e}"}
    except (KeyError, IndexError) as e:
        return {"error": f"Could not parse weather data structure: {e}"}
    
@tool
def get_traffic_data(bounding_box: str):
    """ Getting live traffic data"""
    TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
    if not TOMTOM_API_KEY:
        return {"error": "TomTom API key not found. Please check your .env file."}
    
    # --- Pydantic validation (added) ---
    try:
        _ = BBox.parse(bounding_box)
    except Exception as e:
        return {"error": f"{e}"}

    base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"

    # Minimal valid projection (one line, no newlines or spaces)
    fields_raw = "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,length,magnitudeOfDelay,delay,roadNumbers,from,to,events{description,code}}}}"
    fields_encoded = quote(fields_raw, safe="")

    url = (
        f"{base_url}?key={TOMTOM_API_KEY}"
        f"&bbox={bounding_box}"
        f"&fields={fields_encoded}"
        f"&language=en-GB"
        f"&timeValidityFilter=present"
    )

    try:
        resp = requests.get(url, timeout=20, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        sample_url = url.replace(TOMTOM_API_KEY, "****")  # mask key in error
        return {"error": f"Failed to fetch TomTom traffic data: {e}. url: {sample_url}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error contacting TomTom: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

    incidents_out = []
    for inc in data.get("incidents", []):
        props = inc.get("properties", {}) or {}
        events = props.get("events") or []
        desc = None
        if events and isinstance(events, list) and isinstance(events[0], dict):
            desc = events[0].get("description")
        incidents_out.append({
            "category": props.get("iconCategory"),
            "description": desc,
            "from": props.get("from"),
            "to": props.get("to"),
            "road_numbers": props.get("roadNumbers"),
            # Coerce numerics but keep original keys
            "incident_length_m": _to_float(props.get("length"), default=0.0),
            "incident_delay": _to_float(props.get("delay"), default=0.0),
            "incident_magnitude_delay": _to_float(props.get("magnitudeOfDelay"), default=0.0)
        })

    if not incidents_out:
        return {"status": "No incidents returned for this area."}

    return incidents_out

@tool
def get_travel_time_osrm(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    waypoints: Optional[List[Tuple[float, float]]] = None,
    *,
    optimize_order: bool = False,
    server: str = "https://router.project-osrm.org"
) -> Dict[str, Any]:
    """
    Compute driving time with OSRM. origin/waypoints/destination are (lat, lon).
    If optimize_order=True, uses OSRM's trip service (TSP) with fixed start & end.
    Returns seconds and per-leg durations; no live traffic.
    """
    # --- light Pydantic validation for coordinates (added) ---
    try:
        _ = LatLon(lat=float(origin[0]), lon=float(origin[1]))
        _ = LatLon(lat=float(destination[0]), lon=float(destination[1]))
        for wp in (waypoints or []):
            _ = LatLon(lat=float(wp[0]), lon=float(wp[1]))
    except Exception as e:
        return {"error": f"Invalid coordinates: {e}"}

    coords = [origin] + (waypoints or []) + [destination]
    # OSRM expects lon,lat order
    lonlat = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)

    if optimize_order:
        # Fix start=first and end=last, let OSRM reorder the intermediates
        url = (f"{server}/trip/v1/driving/{lonlat}"
               "?source=first&destination=last&roundtrip=false&overview=false&steps=false")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        trips = data.get("trips", [])
        if not trips:
            return {"error": data.get("message", "No trip found")}
        trip = trips[0]
        legs = trip.get("legs", [])
        leg_secs = [int(leg.get("duration", 0)) for leg in legs]
        # Return OSRM's optimized order for the intermediates
        # waypoints[0] and waypoints[-1] are start/end; the rest have an "waypoint_index"
        optimized_order = []
        for wp in data.get("waypoints", []):
            if wp.get("trips_index", 0) == 0 and wp.get("waypoint_index") not in (0, len(coords)-1):
                optimized_order.append(wp["waypoint_index"] - 1)  # shift to 0-based among intermediates
        return {
            "total_duration_sec": int(trip.get("duration", 0)),
            "distance_meters": int(trip.get("distance", 0)),
            "leg_durations_sec": leg_secs,
            "num_waypoints": len(waypoints or []),
            "optimized_intermediate_order": sorted(optimized_order) or None
        }
    else:
        url = (f"{server}/route/v1/driving/{lonlat}"
               "?overview=false&steps=false&annotations=duration")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        routes = data.get("routes", [])
        if not routes:
            return {"error": data.get("message", "No route found")}
        route = routes[0]
        leg_secs = [int(leg.get("duration", 0)) for leg in route.get("legs", [])]
        return {
            "total_duration_sec": int(route.get("duration", 0)),
            "distance_meters": int(route.get("distance", 0)),
            "leg_durations_sec": leg_secs,
            "num_waypoints": len(waypoints or []),
            "optimized_intermediate_order": None
        }