/**
 * Event type definitions for Nova Sonic bidirectional streaming
 */

// Content types
export type ContentType = 'TEXT' | 'AUDIO' | 'TOOL';
export type Role = 'USER' | 'ASSISTANT' | 'SYSTEM' | 'TOOL';
export type StopReason = 'END_TURN' | 'INTERRUPTED' | 'MAX_TOKENS' | 'TOOL_USE';

// Session events
export interface SessionStartEvent {
    inferenceConfiguration: {
        maxTokens: number;
        topP: number;
        temperature: number;
    };
    turnDetectionConfiguration?: {
        endpointingSensitivity: 'HIGH' | 'MEDIUM' | 'LOW';
    };
}

export interface SessionEndEvent {
    // Empty object signals session end
}

// Content events
export interface ContentStartEvent {
    promptName: string;
    contentName: string;
    type: ContentType;
    role: Role;
    interactive?: boolean;
    textInputConfiguration?: {
        mediaType: 'text/plain';
    };
    audioInputConfiguration?: AudioInputConfiguration;
    additionalModelFields?: string; // JSON string with generationStage etc.
}

export interface ContentEndEvent {
    promptName: string;
    contentName: string;
    type?: ContentType;
    stopReason?: StopReason;
}

// Audio events
export interface AudioInputConfiguration {
    audioType: 'SPEECH';
    encoding: string;
    mediaType: 'audio/lpcm';
    sampleRateHertz: number;
    sampleSizeBits: number;
    channelCount: number;
}

export interface AudioOutputConfiguration {
    mediaType: 'audio/lpcm';
    sampleRateHertz: number;
    sampleSizeBits: number;
    channelCount: number;
    voiceId: string;
}


// Text events
export interface TextInputEvent {
    promptName: string;
    contentName: string;
    content: string;
}

export interface TextOutputEvent {
    role: Role;
    content: string;
}

// Audio events
export interface AudioInputEvent {
    promptName: string;
    contentName: string;
    content: string; // Base64 encoded audio
}

export interface AudioOutputEvent {
    content: string; // Base64 encoded audio
}

// Tool events
export interface ToolUseEvent {
    toolUseId: string;
    toolName: string;
    content?: string; // JSON string of tool parameters
}

export interface ToolResultEvent {
    toolUseId: string;
    result: unknown;
}

// Prompt events
export interface PromptStartEvent {
    promptName: string;
    textOutputConfiguration: {
        mediaType: 'text/plain';
    };
    audioOutputConfiguration: AudioOutputConfiguration;
    toolUseOutputConfiguration: {
        mediaType: 'application/json';
    };
    toolConfiguration?: {
        tools: Array<{
            toolSpec: {
                name: string;
                description: string;
                inputSchema: { json: string };
            };
        }>;
    };
}

export interface PromptEndEvent {
    promptName: string;
}

// Error events
export interface StreamErrorEvent {
    type: 'modelStreamErrorException' | 'internalServerException' | 'error';
    source?: string;
    message?: string;
    details?: unknown;
}

// Barge-in event
export interface BargeInEvent {
    interrupted: boolean;
}

// Usage event
export interface UsageEvent {
    inputTokens?: number;
    outputTokens?: number;
}

// Completion event
export interface CompletionStartEvent {
    // Signals completion phase started
}

// Stream complete event
export interface StreamCompleteEvent {
    timestamp: string;
}

/**
 * Union type of all possible events from Nova Sonic
 */
export type NovaSonicEvent =
    | { type: 'sessionStart'; data: SessionStartEvent }
    | { type: 'sessionEnd'; data: SessionEndEvent }
    | { type: 'contentStart'; data: ContentStartEvent }
    | { type: 'contentEnd'; data: ContentEndEvent }
    | { type: 'textOutput'; data: TextOutputEvent }
    | { type: 'audioOutput'; data: AudioOutputEvent }
    | { type: 'toolUse'; data: ToolUseEvent }
    | { type: 'toolResult'; data: ToolResultEvent }
    | { type: 'bargeIn'; data: BargeInEvent }
    | { type: 'error'; data: StreamErrorEvent }
    | { type: 'streamComplete'; data: StreamCompleteEvent }
    | { type: 'usageEvent'; data: UsageEvent }
    | { type: 'completionStart'; data: CompletionStartEvent };

/**
 * Event handler type for session events
 */
export type EventHandler<T = unknown> = (data: T) => void;

/**
 * Map of event types to their handler signatures
 */
export interface EventHandlerMap {
    sessionStart: EventHandler<SessionStartEvent>;
    sessionEnd: EventHandler<SessionEndEvent>;
    contentStart: EventHandler<ContentStartEvent>;
    contentEnd: EventHandler<ContentEndEvent>;
    textOutput: EventHandler<TextOutputEvent>;
    audioOutput: EventHandler<AudioOutputEvent>;
    toolUse: EventHandler<ToolUseEvent>;
    toolResult: EventHandler<ToolResultEvent>;
    toolEnd: EventHandler<{ toolUseContent: unknown; toolUseId: string; toolName: string }>;
    bargeIn: EventHandler<BargeInEvent>;
    error: EventHandler<StreamErrorEvent>;
    streamComplete: EventHandler<StreamCompleteEvent>;
    usageEvent: EventHandler<UsageEvent>;
    completionStart: EventHandler<CompletionStartEvent>;
    any: EventHandler<NovaSonicEvent>;
}
