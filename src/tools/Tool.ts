/**
 * Tool interface and registry for Nova Sonic function calling
 */

export interface ToolSpec {
    name: string;
    description: string;
    inputSchema: {
        json: string; // JSON stringified schema
    };
}

export interface ToolExecutionContext {
    /** Inference config from the session (temperature, topP, maxTokens) */
    inferenceConfig?: {
        maxTokens: number;
        topP: number;
        temperature: number;
    };
}

export interface Tool {
    /** Tool name (case-insensitive matching) */
    name: string;
    /** Human-readable description for the model */
    description: string;
    /** JSON Schema for input parameters */
    inputSchema: object;
    /** Execute the tool with parsed parameters and optional context */
    execute(params: unknown, context?: ToolExecutionContext): Promise<unknown>;
}

export class ToolRegistry {
    private tools = new Map<string, Tool>();

    register(tool: Tool): void {
        this.tools.set(tool.name.toLowerCase(), tool);
    }

    get(name: string): Tool | undefined {
        return this.tools.get(name.toLowerCase());
    }

    has(name: string): boolean {
        return this.tools.has(name.toLowerCase());
    }

    /**
     * Get all tool specs formatted for Nova Sonic promptStart event
     * Note: Nova Sonic expects inputSchema.json as a stringified JSON string
     */
    getToolSpecs(): Array<{ toolSpec: ToolSpec }> {
        return Array.from(this.tools.values()).map(tool => ({
            toolSpec: {
                name: tool.name,
                description: tool.description,
                inputSchema: {
                    json: JSON.stringify(tool.inputSchema)
                }
            }
        }));
    }

    /**
     * Execute a tool by name with optional context
     */
    async execute(name: string, params: unknown, context?: ToolExecutionContext): Promise<unknown> {
        const tool = this.get(name);
        if (!tool) {
            throw new Error(`Tool "${name}" not found in registry`);
        }
        return tool.execute(params, context);
    }
}
