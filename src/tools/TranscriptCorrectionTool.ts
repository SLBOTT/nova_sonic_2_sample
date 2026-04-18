/**
 * TranscriptCorrectionTool - Auto-corrects transcript reading problems using LLM
 */
import { Tool, ToolExecutionContext } from './Tool';
import { BedrockRuntimeClient, ConverseCommand, Message } from '@aws-sdk/client-bedrock-runtime';
import { ToolModels } from '../consts';

interface ConversationMessage {
    role: 'user' | 'assistant';
    content: string;
}

interface TranscriptCorrectionParams {
    content?: string;
    unclearTerm: string;
    conversations?: ConversationMessage[];
}

// Lazy-initialized Bedrock client
let bedrockClient: BedrockRuntimeClient | null = null;

function getBedrockClient(): BedrockRuntimeClient {
    if (!bedrockClient) {
        bedrockClient = new BedrockRuntimeClient({
            region: ToolModels.transcriptCorrection.region
            // credentials omitted - SDK uses default chain (env vars, profile, IAM role, etc.)
        });
    }
    return bedrockClient;
}

function parseParams(params: unknown): TranscriptCorrectionParams {
    if (!params || typeof params !== 'object') {
        return { unclearTerm: '' };
    }
    
    const content = params as TranscriptCorrectionParams;
    return content;
}

function formatConversationContext(conversations?: ConversationMessage[]): string {
    if (!conversations || conversations.length === 0) {
        return 'No recent conversation context available.';
    }
    
    return conversations
        .slice(-3) // Ensure we only use last 3 messages
        .map(msg => `${msg.role.toUpperCase()}: ${msg.content}`)
        .join('\n');
}

async function invokeCorrectionModel(
    unclearTerm: string, 
    conversations?: ConversationMessage[],
    inferenceConfig?: ToolExecutionContext['inferenceConfig']
): Promise<object> {
    const client = getBedrockClient();
    
    const conversationContext = formatConversationContext(conversations);
    
    const systemPrompt = `You are a speech-to-text correction assistant specialized in identifying misheard terms from audio transcripts.`;

    const userMessage = `In an audio transcript, users mention "${unclearTerm}" and I believe there are some transcript problem on similar phonetic or pronounciation. Consider the pronounciations for words with similar phonetic. Suggest a list of possible corrections for this term.

## CONSIDERATIONS ##
- The results MUST have similiar pronounciations. Remove any results that do not share similar pronounications
- Consider the conversation for more background
- Look for some popular and well-known terms
- Consider the result must be meaningful and human understandable

## Output ##
List a bullet point of at most 5 most-likely possible words without any other texts or explanations

<conversation>${conversationContext}</conversation>`;

    const temperature = inferenceConfig?.temperature ?? 0.3; // Lower temp for more focused suggestions
    const topP = inferenceConfig?.topP ?? 0.9;

    // Build messages for Converse API
    const messages: Message[] = [
        {
            role: 'user',
            content: [{ text: userMessage }]
        }
    ];

    try {
        const config = ToolModels.transcriptCorrection;
        const isNovaModel = config.modelId.includes('nova');
        
        // Build base command input
        const commandInput: Record<string, unknown> = {
            modelId: config.modelId,
            messages,
            system: [{ text: systemPrompt }],
            inferenceConfig: {
                maxTokens: 1024,
                temperature,
                topP
            }
        };

        // Add web grounding tool config (Nova only)
        if (isNovaModel && config.webGrounding) {
            commandInput.toolConfig = {
                tools: [{ systemTool: { name: 'nova_grounding' } }]
            };
        }

        // Add extended thinking config (Nova only)
        if (isNovaModel && config.extendedThinking) {
            commandInput.additionalModelRequestFields = {
                reasoningConfig: {
                    type: 'enabled',
                    maxReasoningEffort: config.maxReasoningEffort
                }
            };
        }
        
        const command = new ConverseCommand(commandInput as any);

        const response = await client.send(command);
        
        // Extract text from Converse API response
        const outputText = response.output?.message?.content?.[0]?.text || '{}';
        
        console.log(`Transcript correction response for "${unclearTerm}"`);

        // Return the raw suggestions list
        return {
            unclearTerm,
            suggestions: outputText
        };
    } catch (error) {
        console.error('Error invoking transcript correction model:', error);
        throw error;
    }
}

export const TranscriptCorrectionTool: Tool = {
    name: 'transcriptCorrectionTool',
    description: `Fixes speech recognition errors by analyzing phonetic similarities. Use this tool when a user corrects you, repeats themselves, or when a name, place, or term does not match any known entity. Also use it when the user sounds frustrated, when proper nouns seem misspelled, or when the conversation context suggests a different word than what was transcribed. Input the unclear term and recent conversation for context. The tool returns likely corrections based on similar pronunciations.`,
    
    inputSchema: {
        type: 'object',
        properties: {
            unclearTerm: {
                type: 'string',
                description: 'The unclear or potentially misheard term that needs correction'
            },
            conversations: {
                type: 'array',
                description: 'Recent conversation messages (last 3) for context',
                items: {
                    type: 'object',
                    properties: {
                        role: {
                            type: 'string',
                            enum: ['user', 'assistant'],
                            description: 'Who said this message'
                        },
                        content: {
                            type: 'string',
                            description: 'The message content'
                        }
                    },
                    required: ['role', 'content']
                }
            }
        },
        required: ['unclearTerm']
    },

    async execute(params: unknown, context?: ToolExecutionContext): Promise<object> {
        const parsed = parseParams(params);
        
        if (!parsed.unclearTerm) {
            return {
                error: true,
                message: 'An unclear term is required for correction'
            };
        }

        try {
            return await invokeCorrectionModel(
                parsed.unclearTerm, 
                parsed.conversations,
                context?.inferenceConfig
            );
        } catch (error) {
            return {
                error: true,
                message: error instanceof Error ? error.message : 'Unknown error occurred',
                unclearTerm: parsed.unclearTerm
            };
        }
    }
};
