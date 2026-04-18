export type EndpointingSensitivity = "HIGH" | "MEDIUM" | "LOW";

/**
 * Tool choice configuration per AWS Nova Sonic docs
 * - auto: Model decides when to use tools (default)
 * - any: Model must use at least one tool
 * - tool: Force use of a specific tool by name
 */
export type ToolChoice = 
  | { auto: {} }
  | { any: {} }
  | { tool: { name: string } };

export interface InferenceConfig {
  readonly maxTokens: number;
  readonly topP: number;
  readonly temperature: number;
}

export interface TurnDetectionConfig {
  readonly endpointingSensitivity: EndpointingSensitivity;
}

export interface SessionConfig {
  readonly inferenceConfig?: InferenceConfig;
  readonly turnDetectionConfig?: TurnDetectionConfig;
  readonly toolChoice?: ToolChoice;
}

export type ContentType = "AUDIO" | "TEXT" | "TOOL";
export type AudioType = "SPEECH";
export type AudioMediaType = "audio/lpcm"
export type TextMediaType = "text/plain" | "application/json";


export interface AudioConfiguration {
  readonly audioType: AudioType;
  readonly mediaType: AudioMediaType;
  readonly sampleRateHertz: number;
  readonly sampleSizeBits: number;
  readonly channelCount: number;
  readonly encoding: string;
  readonly voiceId?: string;
}

export interface TextConfiguration {
  readonly mediaType: TextMediaType;
}

export interface ToolConfiguration {
  readonly toolUseId: string;
  readonly type: "TEXT";
  readonly textInputConfiguration: {
    readonly mediaType: "text/plain";
  };
}
