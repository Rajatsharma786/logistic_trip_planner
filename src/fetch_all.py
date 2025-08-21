import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# import your tools and constants (unchanged behavior)
from src.tools import (
    END_POINTS,
    WAYPOINTS,
    get_weather_data,
    get_traffic_data,
)

if __name__ == "__main__":
    load_dotenv()
    print("--- Agentic AI Logistics Planner: Multi-Route Data Ingestion Script ---")
    print(f"--- Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    all_data = {}

    for route_name, route_wps in WAYPOINTS.items():
        print(f"\nFetching Data for: {route_name.replace('_',' ')}")
        print("=" * 40)
        all_data[route_name] = {}

        # lookup dict only (unordered is fine for lookups)
        full_route_locations = {**END_POINTS, **route_wps}

        # ordered node list for segment math
        node_names = ["melbourne"] + list(route_wps.keys()) + ["sydney"]

        # origin / intermediates / destination in declared order
        origin = (END_POINTS["melbourne"]["lat"], END_POINTS["melbourne"]["lon"])
        destination = (END_POINTS["sydney"]["lat"], END_POINTS["sydney"]["lon"])
        waypoints_list = [(route_wps[n]["lat"], route_wps[n]["lon"]) for n in route_wps.keys()]

        # ---- Free routing (OSRM public server). Fixed order; no traffic. ----
        coords = [origin] + waypoints_list + [destination]
        lonlat = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords)
        osrm_url = f"https://router.project-osrm.org/route/v1/driving/{lonlat}?overview=false&steps=false"

        leg_secs, leg_dists = [], []
        route_travel = {}
        try:
            rr = requests.get(osrm_url, timeout=30)
            rr.raise_for_status()
            osrm = rr.json()
            routes = osrm.get("routes") or []
            if routes:
                route = routes[0]
                leg_secs  = [int(leg.get("duration", 0)) for leg in route.get("legs", [])]
                leg_dists = [int(leg.get("distance", 0)) for leg in route.get("legs", [])]
                route_travel = {
                    "total_duration_sec": int(route.get("duration", 0)),
                    "distance_meters": int(route.get("distance", 0)),
                    "leg_durations_sec": leg_secs,
                    "leg_distances_meters": leg_dists
                }
            else:
                route_travel = {"error": osrm.get("message", "No route returned")}
        except Exception as e:
            route_travel = {"error": f"OSRM call failed: {e}"}

        # cumulative time to each node
        cum_time = [0]
        for s in leg_secs:
            cum_time.append(cum_time[-1] + s)

        # ---- Per-location fetch + segment attachment ----
        for idx, loc_name in enumerate(node_names):
            print(f"  Processing Location: {loc_name.capitalize()}")
            cfg = full_route_locations[loc_name]
            lat, lon = cfg["lat"], cfg["lon"]

            bbox_size = 0.25
            min_lon, min_lat = lon - bbox_size, lat - bbox_size
            max_lon, max_lat = lon + bbox_size, lat + bbox_size
            location_bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

            all_data[route_name][loc_name] = {}

            weather = get_weather_data(cfg["weather_station_id"])
            all_data[route_name][loc_name]["weather"] = weather

            traffic = get_traffic_data(location_bbox)
            all_data[route_name][loc_name]["traffic"] = traffic

            # segment info for this node
            inbound = None if idx == 0 else {
                "from": node_names[idx - 1],
                "to": loc_name,
                "duration_sec": leg_secs[idx - 1] if idx - 1 < len(leg_secs) else None,
                "distance_m":  leg_dists[idx - 1] if idx - 1 < len(leg_dists) else None,
            }
            outbound = None if idx == len(node_names) - 1 else {
                "from": loc_name,
                "to": node_names[idx + 1],
                "duration_sec": leg_secs[idx] if idx < len(leg_secs) else None,
                "distance_m":  leg_dists[idx] if idx < len(leg_dists) else None,
            }

            all_data[route_name][loc_name]["segment"] = {
                "inbound": inbound,
                "outbound": outbound,
                "cumulative_time_to_here_sec": cum_time[idx] if idx < len(cum_time) else None,
                "remaining_time_from_here_sec": (cum_time[-1] - cum_time[idx]) if idx < len(cum_time) else None,
            }

            # keep this for backward-compat with your processing
            all_data[route_name][loc_name]["route_travel"] = route_travel

        # route-level summary once
        all_data[route_name]["__route_summary__"] = {
            "total_duration_sec": route_travel.get("total_duration_sec"),
            "total_distance_meters": route_travel.get("distance_meters"),
            "num_waypoints": len(waypoints_list),
        }

    print("\n\n--- COMPLETE DATA PAYLOAD ---")
    print(json.dumps(all_data, indent=2))