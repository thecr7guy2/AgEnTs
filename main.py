from __future__ import annotations as _annotations
import asyncio
import os
from dataclasses import dataclass
from typing import Any
from devtools import debug
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext
from dotenv import load_dotenv
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic import BaseModel
from typing import List


load_dotenv()


class struct_output(BaseModel):
    Temperature: str
    Description: str
    AirQualityIndex: int


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None
    aqi_index_key:str | None

model = GeminiModel(
    'gemini-2.0-flash-exp', provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
)

# model = GeminiModel(
#     'gemma-3', provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
# )

weather_agent = Agent(
    model,
    system_prompt=(
    'When asked about the weather for one or more locations, '
    'handle each location separately. '
    'For each location, first use the `get_lat_lng` tool to find latitude and longitude, '
    'then use the `get_weather` tool to get the weather, '
    'then use the `get_aqi` tool to get the Air Quality Index. '
    'Be concise, one sentence per location. '
    '**If the input is not a weather/location-related query, refuse politely.**'
    ),
    deps_type=Deps,
    retries=2,
    instrument=True,
    output_type=List[struct_output],
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }

    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    else:
        raise ModelRetry('Could not find the location')




@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        return {'temperature': '21 °C',}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }

    r = await ctx.deps.client.get(
        'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()

    values = data['data']['values']
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }


@weather_agent.tool
async def get_aqi(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the the Air Quality Index given the georaphical coordinates of a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.aqi_index_key is None:
        return {'AirQualityIndex': 10000} 

    r = await ctx.deps.client.get(
        f'https://api.waqi.info/feed/geo:{lat};{lng}/?token={ctx.deps.aqi_index_key}',
    )
    r.raise_for_status()
    data = r.json()

    if data:
        return {'AirQualityIndex': data['data']['aqi']}
    else:
        raise ModelRetry('Could not find the location')


async def main():
    async with AsyncClient() as client:
        geo_api_key = os.getenv("LOCATION_API_KEY")
        weather_api_key = os.getenv("WAETHER_API_KEY")
        aqi_api_key = os.getenv("AQI_API_KEY")
        deps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key, aqi_index_key = aqi_api_key
        )
        result = await weather_agent.run(
            '', deps=deps
        )
        debug(result)
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())