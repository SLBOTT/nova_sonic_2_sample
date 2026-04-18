/**
 * WikipediaTool - Search and retrieve content from Wikipedia
 * Uses Wikipedia's action=query API which handles multilingual queries
 */
import { Tool } from './Tool';
import { DefaultToolConfiguration } from '../consts';

const MAX_CONTENT_LENGTH = DefaultToolConfiguration.maxResultLength - 1000;

interface WikipediaParams {
    query: string;
    mode?: 'search' | 'summary' | 'content';
    limit?: number;
}

interface WikipediaToolContent {
    content?: string;
    query?: string;
    mode?: 'search' | 'summary' | 'content';
    limit?: number;
}

interface SearchResult {
    title: string;
    snippet: string;
    url: string;
}

interface WikiSearchResult {
    title: string;
    pageid: number;
    snippet: string;
}

function parseParams(params: unknown): WikipediaParams | null {
    const content = params as WikipediaToolContent;
    
    if (content?.query) {
        return { 
            query: content.query,
            mode: content.mode || 'search',
            limit: content.limit || 5
        };
    }
    
    return null;
}

/**
 * Search Wikipedia using action=query&list=search API
 */
async function searchWikipedia(query: string, limit: number): Promise<SearchResult[]> {
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&prop=info&origin=*&srlimit=${limit}&utf8=&format=json&srsearch=${encodeURIComponent(query)}`;

    const response = await fetch(url, {
        headers: { 
            'User-Agent': 'NovaSonicVoicebot/1.0',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error(`Wikipedia API returned ${response.status}`);
    }

    const data = await response.json();
    const results = data.query?.search || [];

    return results.map((result: WikiSearchResult) => ({
        title: result.title,
        snippet: result.snippet.replace(/<[^>]*>/g, ''), // Strip HTML tags
        url: `https://en.wikipedia.org/wiki/${encodeURI(result.title)}`
    }));
}

/**
 * Get Wikipedia article summary
 */
async function getWikipediaSummary(query: string): Promise<object> {
    const searchResults = await searchWikipedia(query, 1);
    
    if (searchResults.length === 0) {
        throw new Error(`No Wikipedia articles found for "${query}"`);
    }

    const title = searchResults[0].title;
    const encodedTitle = encodeURIComponent(title.replace(/ /g, '_'));
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodedTitle}`;

    const response = await fetch(url, {
        headers: { 
            'User-Agent': 'NovaSonicVoicebot/1.0',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        return {
            title: searchResults[0].title,
            extract: searchResults[0].snippet,
            url: searchResults[0].url
        };
    }

    const data = await response.json();
    return {
        title: data.title,
        extract: data.extract || 'No summary available',
        url: data.content_urls?.desktop?.page || searchResults[0].url
    };
}

/**
 * Get full Wikipedia article content (plain text)
 */
async function getWikipediaContent(query: string): Promise<object> {
    const searchResults = await searchWikipedia(query, 1);
    
    if (searchResults.length === 0) {
        throw new Error(`No Wikipedia articles found for "${query}"`);
    }

    const title = searchResults[0].title;
    const url = `https://en.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(title)}&prop=extracts&explaintext=1&exsectionformat=plain&format=json&origin=*`;

    const response = await fetch(url, {
        headers: { 
            'User-Agent': 'NovaSonicVoicebot/1.0',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error(`Wikipedia API returned ${response.status}`);
    }

    const data = await response.json();
    const pages = data.query?.pages || {};
    const page = Object.values(pages)[0] as { title: string; extract: string; pageid: number };

    if (!page || !page.extract) {
        return {
            title,
            content: searchResults[0].snippet,
            url: searchResults[0].url
        };
    }

    // Truncate if too long
    const content = page.extract.length > MAX_CONTENT_LENGTH
        ? page.extract.substring(0, MAX_CONTENT_LENGTH) + '...' 
        : page.extract;

    return {
        title: page.title,
        content,
        url: `https://wikipedia.org/wiki/${encodeURI(title)}`
    };
}

export const WikipediaTool: Tool = {
    name: 'searchWikipedia',
    description: `Look up factual information on Wikipedia. Use this tool for questions about people, places, events, science, technology, companies, movies, music, books, or any topic requiring accurate details. Your training data may be outdated, so verify facts here first. Use search mode to find relevant articles, summary mode for quick facts, or content mode for detailed information. For non-English names or terms, preserve the original characters without encode.`,
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'Search query. MUST preserve original script for non-English terms (Chinese, Japanese, Korean, etc.). Do NOT romanize.'
            },
            mode: {
                type: 'string',
                enum: ['search', 'summary', 'content'],
                description: '"search": list matching articles; "summary": brief 2-3 sentence overview; "content": full detailed article text.'
            },
            limit: {
                type: 'number',
                description: 'Max search results (1-10, default 5). Only for search mode.'
            }
        },
        required: ['query']
    },

    async execute(params: unknown): Promise<object> {
        const parsed = parseParams(params);
        
        if (!parsed || !parsed.query) {
            throw new Error('Invalid Wikipedia tool parameters: query is required');
        }

        const { query, mode = 'search', limit = 5 } = parsed;
        const clampedLimit = Math.min(Math.max(1, limit), 10);

        console.log(`Wikipedia ${mode}: "${query}"${mode === 'search' ? ` (limit: ${clampedLimit})` : ''}`);

        if (mode === 'summary') {
            return await getWikipediaSummary(query);
        }

        if (mode === 'content') {
            return await getWikipediaContent(query);
        }

        const results = await searchWikipedia(query, clampedLimit);
        return {
            query,
            count: results.length,
            results
        };
    }
};
