import { AudioType, AudioMediaType, TextMediaType } from "./types";

// AWS Configuration
export const AWSConfig = {
  profile: process.env.AWS_PROFILE || 'bedrock-test',
  defaultRegion: 'ap-northeast-1',
  availableRegions: ['ap-northeast-1', 'us-east-1', 'us-west-2', 'eu-north-1']
};

// Model Configuration
export const NovaSonicModelId = 'amazon.nova-2-sonic-v1:0';

export const ToolModels = {
  reasoning: {
    modelId: 'global.anthropic.claude-sonnet-4-20250514-v1:0',
    // modelId: 'us.amazon.nova-2-lite-v1:0',
    // modelId: 'us.amazon.nova-premier-v1:0',
    // region: 'us-east-1',
    region: 'ap-northeast-1',
    // Nova-specific features (only work with Nova models like us.amazon.nova-2-lite-v1:0)
    extendedThinking: false,
    maxReasoningEffort: 'low' as 'low' | 'medium' | 'high',
    webGrounding: false,
  },
  transcriptCorrection: {
    modelId: 'global.anthropic.claude-sonnet-4-20250514-v1:0',
    // modelId: 'us.amazon.nova-2-lite-v1:0',
    // modelId: 'us.amazon.nova-premier-v1:0',
    // region: 'us-east-1',
    region: 'ap-northeast-1',
    // Nova-specific features (only work with Nova models like us.amazon.nova-2-lite-v1:0)
    extendedThinking: false,
    maxReasoningEffort: 'low' as 'low' | 'medium' | 'high',
    webGrounding: false,
  }
};

export const DefaultInferenceConfiguration = {
  maxTokens: 1024,
  topP: 0.9,
  temperature: 0.7,
};

export const DefaultTurnDetectionConfiguration = {
  responseTiming: 'medium'
};

export const DefaultAudioInputConfiguration = {
  audioType: "SPEECH" as AudioType,
  encoding: "base64",
  mediaType: "audio/lpcm" as AudioMediaType,
  sampleRateHertz: 16000,
  sampleSizeBits: 16,
  channelCount: 1,
};

export const DefaultTextConfiguration = { mediaType: "text/plain" as TextMediaType };

export const DefaultAudioOutputConfiguration = {
  mediaType: "audio/lpcm" as AudioMediaType,
  sampleRateHertz: 24000,
  sampleSizeBits: 16,
  channelCount: 1,
  voiceId: "tiffany",
  encoding: "base64",
  audioType: "SPEECH" as AudioType,
  bufferMs: 200,
};

// Tool Configuration
export const DefaultToolConfiguration = {
  maxResultLength: 20480,
};
