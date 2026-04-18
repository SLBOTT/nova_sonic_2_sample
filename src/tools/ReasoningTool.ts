/**
 * ReasoningTool - Uses a language model for complex reasoning tasks
 * This tool invokes a reasoning model via Bedrock Converse API for advanced problem-solving
 */
import { Tool, ToolExecutionContext } from './Tool';
import { BedrockRuntimeClient, ConverseCommand, Message } from '@aws-sdk/client-bedrock-runtime';
import { ToolModels } from '../consts';

interface ReasoningParams {
    content?: string;
    question?: string;
    context?: string;
    task?: 'reason' | 'analyze' | 'solve' | 'explain' | 'verify' | 'brainstorm' | 'summarize';
}

// Lazy-initialized Bedrock client
let bedrockClient: BedrockRuntimeClient | null = null;

function getBedrockClient(): BedrockRuntimeClient {
    if (!bedrockClient) {
        bedrockClient = new BedrockRuntimeClient({
            region: ToolModels.reasoning.region
            // credentials omitted - SDK uses default chain (env vars, profile, IAM role, etc.)
        });
    }
    return bedrockClient;
}

function parseParams(params: unknown): ReasoningParams {
    const content = params as ReasoningParams;
    return content || {};
}

async function invokeReasoningModel(question: string, context?: string, task?: string, inferenceConfig?: ToolExecutionContext['inferenceConfig']): Promise<object> {
    const client = getBedrockClient();
    
    // Build the prompt based on task type
    let systemPrompt = `You are an advanced reasoning assistant embedded within a voice-based AI system. Your role is to provide deeper analysis, fact-checking, and complex problem-solving support when the primary voice assistant needs backup.

Be accurate and thoughtful since the voice assistant is relying on you for correctness. Acknowledge uncertainty when appropriate. Provide structured thinking when helpful, but keep it concise.

Never make up facts, statistics, dates, names, or any specific data. If you do not know something with certainty, say you are not sure or that you do not have that information. Do not invent sources, citations, or references. When uncertain, clearly state your confidence level. Prefer saying you do not know over providing potentially false information.

`;
    
    switch (task) {
        case 'reason':
            systemPrompt += 'Task: Break down the problem step by step using logical reasoning. Show your work clearly.';
            break;
        case 'analyze':
            systemPrompt += 'Task: Provide thorough analysis considering multiple perspectives, trade-offs, and implications.';
            break;
        case 'solve':
            systemPrompt += 'Task: Focus on finding practical solutions with actionable steps and recommendations.';
            break;
        case 'explain':
            systemPrompt += 'Task: Explain concepts clearly with examples and analogies, suitable for learning.';
            break;
        case 'verify':
            systemPrompt += 'Task: Fact-check and verify the information. Point out any errors, misconceptions, or areas of uncertainty.';
            break;
        case 'brainstorm':
            systemPrompt += 'Task: Generate creative ideas and alternative approaches. Think outside the box.';
            break;
        case 'summarize':
            systemPrompt += 'Task: Distill the key points into a clear, organized summary.';
            break;
        default:
            systemPrompt += 'Task: Provide a comprehensive, well-reasoned response with your best thinking.';
    }
    
    // Keep responses concise for voice output
    systemPrompt += '\n\nIMPORTANT: Keep your response concise (2-4 sentences) since this will be spoken aloud. Be direct and conversational.';

    // Build the user message
    let userMessage = question;
    if (context) {
        userMessage = `Context: ${context}\n\nQuestion: ${question}`;
    }

    // Use config values if provided, otherwise use defaults
    const temperature = inferenceConfig?.temperature ?? 0.7;
    const topP = inferenceConfig?.topP ?? 0.9;

    // Build messages for Converse API
    const messages: Message[] = [
        {
            role: 'user',
            content: [{ text: userMessage }]
        }
    ];

    try {
        const config = ToolModels.reasoning;
        const isNovaModel = config.modelId.includes('nova');
        
        // Build base command input
        const commandInput: Record<string, unknown> = {
            modelId: config.modelId,
            messages,
            system: [{ text: systemPrompt }],
            inferenceConfig: {
                maxTokens: 2048,
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
        const contentList = response.output?.message?.content || [];
        let outputText = '';
        let reasoningText = '';
        
        for (const item of contentList) {
            const anyItem = item as any;
            if ('reasoningContent' in anyItem && anyItem.reasoningContent) {
                reasoningText = anyItem.reasoningContent.reasoningText?.text || '';
            } else if ('text' in anyItem) {
                outputText += anyItem.text;
            }
        }
        
        if (!outputText) {
            outputText = 'No response generated';
        }
        
        console.log(`Reasoning model response (${outputText.length} chars)`);

        return {
            answer: outputText,
            ...(reasoningText && { reasoning: reasoningText })
        };
    } catch (error) {
        console.error('Error invoking reasoning model:', error);
        throw error;
    }
}

export const ReasoningTool: Tool = {
    name: 'reasoningTool',
    description: `Use this tool for complex reasoning, fact-checking, and deep thinking. It calls a more powerful reasoning model to help with challenging questions, complex math or logic problems, multi-step analysis, pros and cons comparisons, creative brainstorming, or when you want to verify your answer. Better to be accurate than fast.`,
    
    inputSchema: {
        type: 'object',
        properties: {
            question: {
                type: 'string',
                description: 'The question, problem, or topic to think deeply about'
            },
            context: {
                type: 'string',
                description: 'Relevant background info, conversation history, or constraints'
            },
            task: {
                type: 'string',
                enum: ['reason', 'analyze', 'solve', 'explain', 'verify', 'brainstorm', 'summarize'],
                description: 'Type of thinking needed: reason (step-by-step logic), analyze (multi-perspective), solve (find solutions), explain (teach clearly), verify (fact-check), brainstorm (creative ideas), summarize (distill key points)'
            }
        },
        required: ['question']
    },

    async execute(params: unknown, context?: ToolExecutionContext): Promise<object> {
        const parsed = parseParams(params);
        
        if (!parsed.question) {
            return {
                error: true,
                message: 'A question is required for the reasoning model to process'
            };
        }

        try {
            return await invokeReasoningModel(parsed.question, parsed.context, parsed.task, context?.inferenceConfig);
        } catch (error) {
            return {
                error: true,
                message: error instanceof Error ? error.message : 'Unknown error occurred',
                question: parsed.question
            };
        }
    }
};
