/**
 * WeatherTool - Fetches weather data from Open-Meteo API
 */
import { Tool } from './Tool';

type WeatherMode = 'current' | 'forecast';

interface WeatherParams {
    latitude: string | number;
    longitude: string | number;
    mode?: WeatherMode;
}

interface WeatherToolContent {
    content?: string;
    latitude?: string | number;
    longitude?: string | number;
    mode?: WeatherMode;
}

// WMO Weather interpretation codes
const WEATHER_CODES: Record<number, string> = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Foggy',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Moderate drizzle',
    55: 'Dense drizzle',
    61: 'Slight rain',
    63: 'Moderate rain',
    65: 'Heavy rain',
    66: 'Light freezing rain',
    67: 'Heavy freezing rain',
    71: 'Slight snow',
    73: 'Moderate snow',
    75: 'Heavy snow',
    77: 'Snow grains',
    80: 'Slight rain showers',
    81: 'Moderate rain showers',
    82: 'Violent rain showers',
    85: 'Slight snow showers',
    86: 'Heavy snow showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with slight hail',
    99: 'Thunderstorm with heavy hail'
};

function parseParams(params: unknown): WeatherParams | null {
    const content = params as WeatherToolContent;
    
    if (content?.latitude !== undefined && content?.longitude !== undefined) {
        return { 
            latitude: content.latitude, 
            longitude: content.longitude,
            mode: content.mode || 'current'
        };
    }
    
    return null;
}

async function fetchCurrentWeather(latitude: number, longitude: number): Promise<object> {
    const url = `https://api.open-meteo.com/v1/forecast?` +
        `latitude=${latitude}&longitude=${longitude}` +
        `&current=temperature_2m,relative_humidity_2m,cloud_cover,weather_code` +
        `&timezone=auto`;

    const response = await fetch(url, {
        headers: { 'User-Agent': 'NovaSonicVoicebot/1.0', 'Accept': 'application/json' }
    });

    if (!response.ok) throw new Error(`Weather API returned ${response.status}`);

    const data = await response.json();
    const current = data.current;

    return {
        location: { latitude: data.latitude, longitude: data.longitude, timezone: data.timezone },
        current: {
            temperature: current.temperature_2m,
            humidity: current.relative_humidity_2m,
            cloud_cover: current.cloud_cover,
            conditions: WEATHER_CODES[current.weather_code] || 'Unknown'
        },
        units: { temperature: '°C', humidity: '%', cloud_cover: '%' }
    };
}

async function fetchForecast(latitude: number, longitude: number): Promise<object> {
    const url = `https://api.open-meteo.com/v1/forecast?` +
        `latitude=${latitude}&longitude=${longitude}` +
        `&daily=weather_code,temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min` +
        `&forecast_days=7&timezone=auto`;

    const response = await fetch(url, {
        headers: { 'User-Agent': 'NovaSonicVoicebot/1.0', 'Accept': 'application/json' }
    });

    if (!response.ok) throw new Error(`Weather API returned ${response.status}`);

    const data = await response.json();
    const daily = data.daily;

    return {
        location: { latitude: data.latitude, longitude: data.longitude, timezone: data.timezone },
        forecast: daily.time.map((date: string, i: number) => ({
            date,
            conditions: WEATHER_CODES[daily.weather_code[i]] || 'Unknown',
            temp_high: daily.temperature_2m_max[i],
            temp_low: daily.temperature_2m_min[i],
            humidity_high: daily.relative_humidity_2m_max[i],
            humidity_low: daily.relative_humidity_2m_min[i]
        })),
        units: { temperature: '°C', humidity: '%' }
    };
}

export const WeatherTool: Tool = {
    name: 'getWeatherTool',
    description: 'Get weather for a location. Use mode "current" for current conditions or "forecast" for 7-day forecast.',
    inputSchema: {
        type: 'object',
        properties: {
            latitude: {
                type: 'string',
                description: 'Geographical WGS84 latitude of the location.'
            },
            longitude: {
                type: 'string',
                description: 'Geographical WGS84 longitude of the location.'
            },
            mode: {
                type: 'string',
                enum: ['current', 'forecast'],
                description: 'Weather mode: "current" for current conditions (default), "forecast" for 7-day forecast.'
            }
        },
        required: ['latitude', 'longitude']
    },

    async execute(params: unknown): Promise<object> {
        const parsed = parseParams(params);
        
        if (!parsed) {
            throw new Error('Invalid weather tool parameters: latitude and longitude required');
        }

        const lat = typeof parsed.latitude === 'string' ? parseFloat(parsed.latitude) : parsed.latitude;
        const lon = typeof parsed.longitude === 'string' ? parseFloat(parsed.longitude) : parsed.longitude;
        const mode = parsed.mode || 'current';

        if (isNaN(lat) || isNaN(lon)) {
            throw new Error('Invalid coordinates: latitude and longitude must be numbers');
        }

        console.log(`Fetching ${mode} weather for coordinates: ${lat}, ${lon}`);
        
        return mode === 'forecast' 
            ? fetchForecast(lat, lon) 
            : fetchCurrentWeather(lat, lon);
    }
};
