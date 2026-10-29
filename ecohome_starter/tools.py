"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.
 
    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7). Day 0 is today, day 1 is tomorrow.
 
    Returns:
        Dict[str, Any]: Simulated forecast with the current conditions, a daily summary
        (min/max temperature, condition, sunniest hours) and 24 hourly entries per day
        (temperature_c, condition, solar_irradiance in W/m2, humidity, wind_speed).
        Use the "daily" section to pick a date, and the "hourly" section for hour-by-hour timing.
    """
    import math
 
    if not location or not str(location).strip():
        return {"error": "A location is required, for example 'San Francisco, CA'."}
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 3
    days = max(1, min(7, days))
 
    now = datetime.now()
    loc_key = str(location).strip().lower()
 
    # Rough climates for common cities: (annual mean temp C, summer/winter swing C, base humidity %)
    known_climates = {
        "san francisco": (14.5, 3.5, 70), "los angeles": (18.5, 4.0, 60), "seattle": (11.0, 6.5, 72),
        "portland": (12.0, 7.0, 70), "phoenix": (23.5, 10.5, 30), "denver": (10.5, 10.5, 45),
        "austin": (21.0, 8.5, 62), "chicago": (10.0, 13.0, 68), "new york": (13.0, 11.5, 62),
        "boston": (10.5, 12.0, 65), "miami": (25.0, 3.5, 74), "atlanta": (17.5, 10.0, 65),
    }
    climate = random.Random(f"climate|{loc_key}")      # any other place gets a stable climate from its name
    wind_base = climate.uniform(6, 18)                 # km/h
    match = next((v for name, v in known_climates.items() if name in loc_key), None)
    if match:
        mean_temp, seasonal_amp, humidity_base = match
    else:
        mean_temp = climate.uniform(8, 24)
        seasonal_amp = climate.uniform(5, 11)
        humidity_base = climate.uniform(45, 75)
 
    conditions = ["sunny", "partly_cloudy", "cloudy"]
    sun_factor = {"sunny": 1.0, "partly_cloudy": 0.6, "cloudy": 0.25}   # share of clear-sky sunshine
 
    hourly, daily = [], []
    for d in range(days):
        day = now + timedelta(days=d)
        date_str = day.strftime("%Y-%m-%d")
        doy = day.timetuple().tm_yday
        rng = random.Random(f"weather|{loc_key}|{date_str}")   # same place and date -> same forecast
 
        # Northern hemisphere: the sun is highest at the June solstice, temperatures peak about a month later
        sun_wave = math.sin(2 * math.pi * (doy - 80) / 365)
        temp_wave = math.sin(2 * math.pi * (doy - 110) / 365)
        season_temp = seasonal_amp * temp_wave
        season_sun = 0.85 + 0.15 * sun_wave
        day_offset = rng.uniform(-3, 3)
        day_condition = rng.choices(conditions, weights=[0.5, 0.3, 0.2])[0]
 
        day_hours = []
        for hour in range(24):
            diurnal = 5.5 * math.sin(2 * math.pi * (hour - 9) / 24)   # coolest ~3 AM, warmest ~3 PM
            temp = mean_temp + season_temp + day_offset + diurnal + rng.uniform(-0.6, 0.6)
 
            idx = conditions.index(day_condition)
            if rng.random() < 0.15:                                    # some hourly variation in cloud cover
                idx = max(0, min(2, idx + rng.choice([-1, 1])))
            condition = conditions[idx]
 
            if 6 <= hour <= 18:
                clear_sky = math.sin(math.pi * (hour - 6) / 12) * 1000 * season_sun
                irradiance = clear_sky * sun_factor[condition] * rng.uniform(0.92, 1.0)
            else:
                irradiance = 0.0
 
            humidity = humidity_base - 0.9 * diurnal + (8 if condition == "cloudy" else 0) + rng.uniform(-3, 3)
            wind = wind_base + 2 * math.sin(2 * math.pi * (hour - 12) / 24) + rng.uniform(-4, 4)
 
            entry = {
                "date": date_str,
                "hour": hour,
                "temperature_c": round(temp, 1),
                "condition": condition,
                "solar_irradiance": round(irradiance, 1),
                "humidity": round(max(20, min(98, humidity))),
                "wind_speed": round(max(0, wind), 1),
            }
            day_hours.append(entry)
            hourly.append(entry)
 
        temps = [h["temperature_c"] for h in day_hours]
        irr = [h["solar_irradiance"] for h in day_hours]
        peak_irr = max(irr)
        daily.append({
            "date": date_str,
            "day_of_week": day.strftime("%A"),
            "condition": day_condition,
            "min_temp_c": min(temps),
            "max_temp_c": max(temps),
            "avg_temp_c": round(sum(temps) / 24, 1),
            "total_solar_kwh_per_m2": round(sum(irr) / 1000, 2),
            "sunniest_hours": sorted(
                h["hour"] for h in sorted(day_hours, key=lambda x: x["solar_irradiance"], reverse=True)[:4]
                if h["solar_irradiance"] > 0
            ),
        })
 
    current_hour = hourly[now.hour]
    return {
        "location": location,
        "forecast_days": days,
        "note": "Simulated forecast data for demonstration.",
        "units": {"temperature": "C", "solar_irradiance": "W/m2", "humidity": "%", "wind_speed": "km/h"},
        "current": {
            "temperature_c": current_hour["temperature_c"],
            "condition": current_hour["condition"],
            "humidity": current_hour["humidity"],
            "wind_speed": current_hour["wind_speed"],
        },
        "daily": daily,
        "hourly": hourly,
    }
 
 
# Implemented: get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.
 
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
 
    Returns:
        Dict[str, Any]: Simulated time-of-use pricing with 24 hourly rates in USD per kWh.
        Each hour has a "period" ("peak" from 06:00 to 22:00, "off_peak" otherwise) and a
        "demand_charge" (an extra USD per kWh surcharge during peak hours, 0 when off-peak).
        The evening hours (17:00-22:00) are the most expensive. A "summary" gives the
        average peak and off-peak rates and the cheapest and most expensive hours.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    try:
        day = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date '{date}'. Use the YYYY-MM-DD format."}
 
    base_rate = 0.12                                  # USD per kWh
    rng = random.Random(f"prices|{date}")             # same date -> same prices
    is_weekend = day.weekday() >= 5
    season_mult = 1.15 if day.month in (6, 7, 8, 9) else 1.0    # summer cooling demand
    weekend_mult = 0.9 if is_weekend else 1.0                   # weekends are a little cheaper at peak
 
    hourly_rates = []
    for hour in range(24):
        if 6 <= hour < 22:
            period = "peak"
            if 17 <= hour < 22:
                tier, demand_charge = 1.6, 0.04       # evening peak
            elif 6 <= hour < 10:
                tier, demand_charge = 1.3, 0.02       # morning ramp
            else:
                tier, demand_charge = 1.1, 0.02       # midday
            multiplier = tier * season_mult * weekend_mult
        else:
            period, demand_charge, multiplier = "off_peak", 0.0, 0.7
 
        rate = round(base_rate * multiplier * rng.uniform(0.97, 1.03), 4)
        hourly_rates.append({
            "hour": hour,
            "rate": rate,
            "period": period,
            "demand_charge": demand_charge,
        })
 
    peak = [h["rate"] for h in hourly_rates if h["period"] == "peak"]
    off_peak = [h["rate"] for h in hourly_rates if h["period"] == "off_peak"]
    cheapest = min(hourly_rates, key=lambda h: h["rate"])
    priciest = max(hourly_rates, key=lambda h: h["rate"])
 
    return {
        "date": date,
        "day_of_week": day.strftime("%A"),
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "demand_charge_unit": "USD per kWh surcharge during peak hours (added to rate)",
        "note": "Simulated pricing data for demonstration.",
        "summary": {
            "peak_hours": "06:00-22:00",
            "off_peak_hours": "22:00-06:00",
            "evening_peak_hours": "17:00-22:00",
            "avg_peak_rate": round(sum(peak) / len(peak), 4),
            "avg_off_peak_rate": round(sum(off_peak) / len(off_peak), 4),
            "cheapest_hour": cheapest["hour"],
            "cheapest_rate": cheapest["rate"],
            "most_expensive_hour": priciest["hour"],
            "most_expensive_rate": priciest["rate"],
        },
        "hourly_rates": hourly_rates,
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
 
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
 
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices. Each tip has a rank, the text,
        its source file and a relevance_score between 0 and 1 (higher is more relevant).
    """
    import glob
    import hashlib
 
    try:
        if not query or not str(query).strip():
            return {"error": "A search query is required."}
        try:
            max_results = max(1, min(10, int(max_results)))
        except (TypeError, ValueError):
            max_results = 5
 
        documents_dir = "data/documents"
        persist_directory = "data/vectorstore"
 
        # Every .txt file in data/documents is indexed, so new tip files are picked up automatically
        doc_paths = sorted(glob.glob(os.path.join(documents_dir, "*.txt")))
        if not doc_paths:
            return {"error": f"No tip documents found in '{documents_dir}'. Add .txt files and try again."}
 
        # The collection name changes whenever the set of files (or their contents) changes,
        # so an index built from older documents is never reused.
        signature = json.dumps(
            [[os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in doc_paths]
        )
        collection_name = "energy_tips_" + hashlib.sha1(signature.encode()).hexdigest()[:10]
 
        # 1. Initialize the vector store (the folder is created on first use)
        os.makedirs(persist_directory, exist_ok=True)
        embeddings = OpenAIEmbeddings(
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
 
        # 2. Load and split the documents only if this collection is still empty.
        # 3. Otherwise the existing vector store is reused as is.
        if not vectorstore.get(limit=1)["ids"]:
            documents = []
            for path in doc_paths:
                documents.extend(TextLoader(path, encoding="utf-8").load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            vectorstore.add_documents(splits)
 
        # 4. Search for relevant documents, with a real similarity score for each one
        try:
            hits = vectorstore.similarity_search_with_relevance_scores(query, k=max_results)
        except Exception:
            hits = [(doc, None) for doc in vectorstore.similarity_search(query, k=max_results)]
 
        # 5. Return a results dict
        results = {
            "query": query,
            "total_results": len(hits),
            "tips": []
        }
        for i, (doc, score) in enumerate(hits):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "relevance_score": round(score, 3) if score is not None else None
            })
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
