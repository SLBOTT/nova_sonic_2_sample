/**
 * LocationSearchTool - Searches for location coordinates using Open-Meteo Geocoding API
 */
import { Tool } from './Tool';

interface LocationSearchParams {
    query: string;
    count?: number;
}

interface LocationSearchContent {
    content?: string;
    query?: string;
    count?: number;
}

interface GeocodingResult {
    name: string;
    latitude: number;
    longitude: number;
    country: string;
}

function parseParams(params: unknown): LocationSearchParams | null {
    const content = params as LocationSearchContent;
    
    if (content?.query) {
        return {
            query: content.query,
            count: content.count
        };
    }
    
    return null;
}

async function searchLocation(query: string, count: number = 3): Promise<object> {
    const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=${count}&language=en`;

    const response = await fetch(url, {
        headers: {
            'User-Agent': 'NovaSonicVoicebot/1.0',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error(`Geocoding API returned ${response.status}`);
    }

    const data = await response.json();
    
    // Return only the fields we need to save LLM tokens
    const results: GeocodingResult[] = (data.results || []).map((r: any) => ({
        name: r.name,
        latitude: r.latitude,
        longitude: r.longitude,
        country: r.country
    }));

    return { locations: results };
}

export const LocationSearchTool: Tool = {
    name: 'searchLocationTool',
    description: 'Search for a city or country location to get its coordinates (latitude/longitude). Query must be in English. Use this to find coordinates before getting weather data.',
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'City name, country name, or postal code to search for. Must be in English.'
            },
            count: {
                type: 'number',
                description: 'Number of results to return (1-10). Default is 3.'
            }
        },
        required: ['query']
    },

    async execute(params: unknown): Promise<object> {
        const parsed = parseParams(params);
        
        if (!parsed || !parsed.query) {
            throw new Error('Invalid location search parameters: query is required');
        }

        const count = Math.min(Math.max(parsed.count || 3, 1), 10);
        
        console.log(`Searching location: "${parsed.query}" (count: ${count})`);
        return searchLocation(parsed.query, count);
    }
};
