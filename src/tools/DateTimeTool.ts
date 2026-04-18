/**
 * DateTimeTool - Comprehensive date and time utility for conversational AI
 * 
 * Supports:
 * - Current time in any timezone
 * - Date calculations (days between dates, add/subtract time)
 * - Time conversions between timezones
 * - Relative date queries (days until event, time since)
 */
import { Tool } from './Tool';

interface DateTimeParams {
    content?: string;
    action?: 'current' | 'convert' | 'calculate' | 'difference';
    timezone?: string;
    fromTimezone?: string;
    toTimezone?: string;
    date?: string;
    targetDate?: string;
    amount?: number;
    unit?: 'days' | 'hours' | 'minutes' | 'weeks' | 'months' | 'years';
    operation?: 'add' | 'subtract';
}

// Server's local timezone (detected at runtime)
const SERVER_TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone;

// Common timezone mappings for natural language
const TIMEZONE_ALIASES: Record<string, string> = {
    // US
    'pst': 'America/Los_Angeles',
    'pacific': 'America/Los_Angeles',
    'pdt': 'America/Los_Angeles',
    'mst': 'America/Denver',
    'mountain': 'America/Denver',
    'cst': 'America/Chicago',
    'central': 'America/Chicago',
    'est': 'America/New_York',
    'eastern': 'America/New_York',
    'edt': 'America/New_York',
    
    // International
    'gmt': 'Europe/London',
    'utc': 'UTC',
    'london': 'Europe/London',
    'paris': 'Europe/Paris',
    'berlin': 'Europe/Berlin',
    'tokyo': 'Asia/Tokyo',
    'jst': 'Asia/Tokyo',
    'seoul': 'Asia/Seoul',
    'kst': 'Asia/Seoul',
    'beijing': 'Asia/Shanghai',
    'shanghai': 'Asia/Shanghai',
    'hong kong': 'Asia/Hong_Kong',
    'singapore': 'Asia/Singapore',
    'sydney': 'Australia/Sydney',
    'aest': 'Australia/Sydney',
    'mumbai': 'Asia/Kolkata',
    'ist': 'Asia/Kolkata',
    'dubai': 'Asia/Dubai',
    'moscow': 'Europe/Moscow',
    'sao paulo': 'America/Sao_Paulo',
    'new york': 'America/New_York',
    'los angeles': 'America/Los_Angeles',
    'chicago': 'America/Chicago',
    'denver': 'America/Denver',
};

function resolveTimezone(tz?: string): string {
    if (!tz) return SERVER_TIMEZONE;
    const normalized = tz.toLowerCase().trim();
    return TIMEZONE_ALIASES[normalized] || tz;
}

function parseParams(params: unknown): DateTimeParams {
    const content = params as DateTimeParams;
    return content || {};
}

function formatDateTime(date: Date, timezone: string): object {
    const options: Intl.DateTimeFormatOptions = { timeZone: timezone };
    
    const formatted = {
        iso: date.toISOString(),
        date: date.toLocaleDateString('en-US', { ...options, dateStyle: 'full' }),
        time: date.toLocaleTimeString('en-US', { 
            ...options, 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        }),
        time24: date.toLocaleTimeString('en-US', { 
            ...options, 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false 
        }),
        year: parseInt(date.toLocaleDateString('en-US', { ...options, year: 'numeric' })),
        month: date.toLocaleDateString('en-US', { ...options, month: 'long' }),
        monthNumber: parseInt(date.toLocaleDateString('en-US', { ...options, month: 'numeric' })),
        day: parseInt(date.toLocaleDateString('en-US', { ...options, day: 'numeric' })),
        dayOfWeek: date.toLocaleDateString('en-US', { ...options, weekday: 'long' }),
        timezone: timezone,
        timezoneAbbr: date.toLocaleTimeString('en-US', { ...options, timeZoneName: 'short' }).split(' ').pop(),
    };
    
    return formatted;
}

function getCurrentTime(timezone: string): object {
    const now = new Date();
    const isServerTimezone = timezone === SERVER_TIMEZONE;
    return {
        action: 'current',
        ...formatDateTime(now, timezone),
        isServerLocalTime: isServerTimezone,
        serverTimezone: SERVER_TIMEZONE
    };
}

function convertTime(fromTz: string, toTz: string, dateStr?: string): object {
    const date = dateStr ? new Date(dateStr) : new Date();
    
    if (isNaN(date.getTime())) {
        throw new Error(`Invalid date: ${dateStr}`);
    }
    
    return {
        action: 'convert',
        from: formatDateTime(date, fromTz),
        to: formatDateTime(date, toTz)
    };
}

function calculateDate(
    baseDate: string | undefined, 
    amount: number, 
    unit: string, 
    operation: 'add' | 'subtract',
    timezone: string
): object {
    const date = baseDate ? new Date(baseDate) : new Date();
    
    if (isNaN(date.getTime())) {
        throw new Error(`Invalid date: ${baseDate}`);
    }
    
    const multiplier = operation === 'subtract' ? -1 : 1;
    const adjustedAmount = amount * multiplier;
    
    switch (unit) {
        case 'minutes':
            date.setMinutes(date.getMinutes() + adjustedAmount);
            break;
        case 'hours':
            date.setHours(date.getHours() + adjustedAmount);
            break;
        case 'days':
            date.setDate(date.getDate() + adjustedAmount);
            break;
        case 'weeks':
            date.setDate(date.getDate() + (adjustedAmount * 7));
            break;
        case 'months':
            date.setMonth(date.getMonth() + adjustedAmount);
            break;
        case 'years':
            date.setFullYear(date.getFullYear() + adjustedAmount);
            break;
        default:
            throw new Error(`Unknown unit: ${unit}`);
    }
    
    return {
        action: 'calculate',
        operation,
        amount,
        unit,
        result: formatDateTime(date, timezone)
    };
}

function dateDifference(date1Str: string | undefined, date2Str: string, timezone: string): object {
    const date1 = date1Str ? new Date(date1Str) : new Date();
    const date2 = new Date(date2Str);
    
    if (isNaN(date1.getTime())) {
        throw new Error(`Invalid start date: ${date1Str || 'now'}`);
    }
    if (isNaN(date2.getTime())) {
        throw new Error(`Invalid target date: ${date2Str}`);
    }
    
    const diffMs = date2.getTime() - date1.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    
    const absDays = Math.abs(diffDays);
    const weeks = Math.floor(absDays / 7);
    const remainingDays = absDays % 7;
    
    const isPast = diffMs < 0;
    
    return {
        action: 'difference',
        from: formatDateTime(date1, timezone),
        to: formatDateTime(date2, timezone),
        difference: {
            totalDays: diffDays,
            totalHours: diffHours,
            totalMinutes: diffMinutes,
            weeks,
            remainingDays,
            isPast,
            humanReadable: isPast 
                ? `${absDays} days ago` 
                : `${absDays} days from now`
        }
    };
}

export const DateTimeTool: Tool = {
    name: 'getDateAndTimeTool',
    description: `Get current date/time, convert between timezones, calculate future/past dates, or find the difference between dates. 
    
Actions:
- "current": Get current time in a timezone (default: server local timezone). Use for "What time is it?" or "What time is it in Tokyo?"
- "convert": Convert time between timezones. Use for "What time is 3pm EST in Tokyo?"
- "calculate": Add or subtract time from a date. Use for "What date is 2 weeks from now?" or "What was the date 30 days ago?"
- "difference": Find days/time between two dates. Use for "How many days until Christmas?" or "How long since July 4th?"

Supported timezones: PST, EST, CST, MST, UTC, GMT, Tokyo, Seoul, Beijing, Singapore, Sydney, London, Paris, Berlin, Mumbai, Dubai, Moscow, or any IANA timezone.`,
    
    inputSchema: {
        type: 'object',
        properties: {
            action: {
                type: 'string',
                enum: ['current', 'convert', 'calculate', 'difference'],
                description: 'The operation to perform'
            },
            timezone: {
                type: 'string',
                description: 'Timezone for current time or calculations (e.g., "Tokyo", "PST", "America/New_York"). If not provided, uses server local timezone.'
            },
            fromTimezone: {
                type: 'string',
                description: 'Source timezone for conversion'
            },
            toTimezone: {
                type: 'string',
                description: 'Target timezone for conversion'
            },
            date: {
                type: 'string',
                description: 'Base date for calculations (ISO format or natural like "2024-12-25"). Defaults to now.'
            },
            targetDate: {
                type: 'string',
                description: 'Target date for difference calculation (e.g., "2024-12-25" for Christmas)'
            },
            amount: {
                type: 'number',
                description: 'Amount of time units to add/subtract'
            },
            unit: {
                type: 'string',
                enum: ['minutes', 'hours', 'days', 'weeks', 'months', 'years'],
                description: 'Time unit for calculations'
            },
            operation: {
                type: 'string',
                enum: ['add', 'subtract'],
                description: 'Whether to add or subtract time'
            }
        },
        required: []
    },

    async execute(params: unknown): Promise<object> {
        const parsed = parseParams(params);
        const action = parsed.action || 'current';
        
        try {
            switch (action) {
                case 'current': {
                    const tz = resolveTimezone(parsed.timezone);
                    return getCurrentTime(tz);
                }
                
                case 'convert': {
                    const fromTz = resolveTimezone(parsed.fromTimezone || parsed.timezone);
                    const toTz = resolveTimezone(parsed.toTimezone);
                    return convertTime(fromTz, toTz, parsed.date);
                }
                
                case 'calculate': {
                    if (!parsed.amount || !parsed.unit) {
                        throw new Error('Calculate action requires amount and unit');
                    }
                    const tz = resolveTimezone(parsed.timezone);
                    return calculateDate(
                        parsed.date, 
                        parsed.amount, 
                        parsed.unit, 
                        parsed.operation || 'add',
                        tz
                    );
                }
                
                case 'difference': {
                    if (!parsed.targetDate) {
                        throw new Error('Difference action requires targetDate');
                    }
                    const tz = resolveTimezone(parsed.timezone);
                    return dateDifference(parsed.date, parsed.targetDate, tz);
                }
                
                default:
                    throw new Error(`Unknown action: ${action}`);
            }
        } catch (error) {
            return {
                error: true,
                message: error instanceof Error ? error.message : 'Unknown error',
                action
            };
        }
    }
};
