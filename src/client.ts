import {
  BedrockRuntimeClient,
  BedrockRuntimeClientConfig,
  InvokeModelWithBidirectionalStreamCommand,
  InvokeModelWithBidirectionalStreamInput,
} from "@aws-sdk/client-bedrock-runtime";
import {
  NodeHttp2Handler,
  NodeHttp2HandlerOptions,
} from "@smithy/node-http-handler";
import { Provider } from "@smithy/types";
import { Buffer } from "node:buffer";
import { randomUUID } from "node:crypto";
import { InferenceConfig, TurnDetectionConfig, EndpointingSensitivity, ToolChoice } from "./types";
import { Subject } from 'rxjs';
import { take } from 'rxjs/operators';
import { firstValueFrom } from 'rxjs';
import {
  DefaultAudioInputConfiguration,
  DefaultAudioOutputConfiguration,
  DefaultTextConfiguration,
  DefaultToolConfiguration,
  NovaSonicModelId,
} from "./consts";
import { ToolRegistry, createDefaultToolRegistry } from "./tools";

export interface NovaSonicBidirectionalStreamClientConfig {
  requestHandlerConfig?:
  | NodeHttp2HandlerOptions
  | Provider<NodeHttp2HandlerOptions | void>;
  clientConfig: Partial<BedrockRuntimeClientConfig>;
  inferenceConfig?: InferenceConfig;
  turnDetectionConfig?: TurnDetectionConfig;
}

export class StreamSession {
  private audioBufferQueue: Buffer[] = [];
  private maxQueueSize = 200; // Maximum number of audio chunks to queue
  private isProcessingAudio = false;
  private isActive = true;

  constructor(
    private sessionId: string,
    private client: NovaSonicBidirectionalStreamClient
  ) { }

  // Register event handlers for this specific session
  public onEvent(eventType: string, handler: (data: any) => void): StreamSession {
    this.client.registerEventHandler(this.sessionId, eventType, handler);
    return this; // For chaining
  }

  private voiceId?: string;
  private outputSampleRate: number = 24000;

  public async setupSessionAndPromptStart(voiceId?: string, outputSampleRate: number = 24000): Promise<void> {
    this.voiceId = voiceId;
    this.outputSampleRate = outputSampleRate;
    this.client.setupSessionStartEvent(this.sessionId);
    this.client.setupPromptStartEvent(this.sessionId, voiceId, outputSampleRate);
  }

  public async setupSystemPrompt(
    textConfig: typeof DefaultTextConfiguration = DefaultTextConfiguration,
    systemPromptContent: string,
    voiceId?: string): Promise<void> {
    // If voiceId provided here, update the stored value
    if (voiceId) {
      this.voiceId = voiceId;
    }
    this.client.setupSystemPromptEvent(this.sessionId, textConfig, systemPromptContent);
  }

  public async setupStartAudio(
    audioConfig: typeof DefaultAudioInputConfiguration = DefaultAudioInputConfiguration
  ): Promise<void> {
    this.client.setupStartAudioEvent(this.sessionId, audioConfig);
  }

  // Send text input (for typing mode)
  public async sendTextInput(textContent: string): Promise<void> {
    if (!textContent || !textContent.trim()) {
      throw new Error('Text content is required');
    }
    this.client.sendTextInputEvent(this.sessionId, textContent.trim());
  }


  // Stream audio for this session
  public async streamAudio(audioData: Buffer): Promise<void> {
    // Check queue size to avoid memory issues
    if (this.audioBufferQueue.length >= this.maxQueueSize) {
      // Queue is full, drop oldest chunk
      this.audioBufferQueue.shift();
      console.log("Audio queue full, dropping oldest chunk");
    }

    // Queue the audio chunk for streaming
    this.audioBufferQueue.push(audioData);
    this.processAudioQueue();
  }

  // Process audio queue for continuous streaming
  private async processAudioQueue() {
    if (this.isProcessingAudio || this.audioBufferQueue.length === 0 || !this.isActive) return;

    this.isProcessingAudio = true;
    try {
      // Process all chunks in the queue, up to a reasonable limit
      let processedChunks = 0;
      const maxChunksPerBatch = 5; // Process max 5 chunks at a time to avoid overload

      while (this.audioBufferQueue.length > 0 && processedChunks < maxChunksPerBatch && this.isActive) {
        const audioChunk = this.audioBufferQueue.shift();
        if (audioChunk) {
          await this.client.streamAudioChunk(this.sessionId, audioChunk);
          processedChunks++;
        }
      }
    } finally {
      this.isProcessingAudio = false;

      // If there are still items in the queue, schedule the next processing using setTimeout
      if (this.audioBufferQueue.length > 0 && this.isActive) {
        setTimeout(() => this.processAudioQueue(), 0);
      }
    }
  }
  // Get session ID
  public getSessionId(): string {
    return this.sessionId;
  }

  public async endAudioContent(): Promise<void> {
    if (!this.isActive) return;
    await this.client.sendContentEnd(this.sessionId);
  }

  public async endPrompt(): Promise<void> {
    if (!this.isActive) return;
    await this.client.sendPromptEnd(this.sessionId);
  }

  public async close(): Promise<void> {
    if (!this.isActive) return;

    this.isActive = false;
    this.audioBufferQueue = []; // Clear any pending audio

    await this.client.sendSessionEnd(this.sessionId);
    console.log(`Session ${this.sessionId} close completed`);
  }
}

// Session data type
interface SessionData {
  queue: Array<any>;
  queueSignal: Subject<void>;
  closeSignal: Subject<void>;
  responseSubject: Subject<any>;
  toolUseContent: any;
  toolUseId: string;
  toolName: string;
  responseHandlers: Map<string, (data: any) => void>;
  promptName: string;
  inferenceConfig: InferenceConfig;
  turnDetectionConfig?: TurnDetectionConfig;
  toolChoice?: ToolChoice;
  enabledTools?: string[];
  isActive: boolean;
  isPromptStartSent: boolean;
  isAudioContentStartSent: boolean;
  audioContentId: string;
}

export class NovaSonicBidirectionalStreamClient {
  private bedrockRuntimeClient: BedrockRuntimeClient;
  private inferenceConfig: InferenceConfig;
  private turnDetectionConfig?: TurnDetectionConfig;
  private activeSessions: Map<string, SessionData> = new Map();
  private sessionLastActivity: Map<string, number> = new Map();
  private sessionCleanupInProgress = new Set<string>();
  private toolRegistry: ToolRegistry;


  constructor(config: NovaSonicBidirectionalStreamClientConfig) {
    const nodeHttp2Handler = new NodeHttp2Handler({
      requestTimeout: 300000,
      sessionTimeout: 300000,
      disableConcurrentStreams: false,
      maxConcurrentStreams: 20,
      ...config.requestHandlerConfig,
    });

    // Allow credentials to be undefined - SDK will use default credential chain
    // (env vars, shared credentials, ECS/EC2 IAM roles)
    this.bedrockRuntimeClient = new BedrockRuntimeClient({
      ...config.clientConfig,
      region: config.clientConfig.region || "ap-northeast-1",
      requestHandler: nodeHttp2Handler
    });

    this.inferenceConfig = config.inferenceConfig ?? {
      maxTokens: 1024,
      topP: 0.9,
      temperature: 0.7,
    };

    this.turnDetectionConfig = config.turnDetectionConfig;
    this.toolRegistry = createDefaultToolRegistry();
  }

  /**
   * Get the tool registry for registering custom tools
   */
  public getToolRegistry(): ToolRegistry {
    return this.toolRegistry;
  }

  public isSessionActive(sessionId: string): boolean {
    const session = this.activeSessions.get(sessionId);
    return !!session && session.isActive;
  }

  public getActiveSessions(): string[] {
    return Array.from(this.activeSessions.keys());
  }

  public getLastActivityTime(sessionId: string): number {
    return this.sessionLastActivity.get(sessionId) || 0;
  }

  private updateSessionActivity(sessionId: string): void {
    this.sessionLastActivity.set(sessionId, Date.now());
  }

  public isCleanupInProgress(sessionId: string): boolean {
    return this.sessionCleanupInProgress.has(sessionId);
  }


  // Create a new streaming session
  public createStreamSession(sessionId: string = randomUUID(), config?: { inferenceConfig?: InferenceConfig; turnDetectionConfig?: TurnDetectionConfig; toolChoice?: ToolChoice; enabledTools?: string[] }): StreamSession {
    if (this.activeSessions.has(sessionId)) {
      throw new Error(`Stream session with ID ${sessionId} already exists`);
    }

    const session: SessionData = {
      queue: [],
      queueSignal: new Subject<void>(),
      closeSignal: new Subject<void>(),
      responseSubject: new Subject<any>(),
      toolUseContent: null,
      toolUseId: "",
      toolName: "",
      responseHandlers: new Map(),
      promptName: randomUUID(),
      inferenceConfig: config?.inferenceConfig ?? this.inferenceConfig,
      turnDetectionConfig: config?.turnDetectionConfig ?? this.turnDetectionConfig,
      toolChoice: config?.toolChoice,
      enabledTools: config?.enabledTools,
      isActive: true,
      isPromptStartSent: false,
      isAudioContentStartSent: false,
      audioContentId: randomUUID()
    };

    this.activeSessions.set(sessionId, session);

    return new StreamSession(sessionId, this);
  }

  private async processToolUse(sessionId: string, toolName: string, toolUseContent: object): Promise<object> {
    const shortSessionId = sessionId.substring(0, 8);
    console.log(`[Tool:${toolName}] Starting execution for session ${shortSessionId}`);
    
    if (!this.toolRegistry.has(toolName)) {
      console.log(`[Tool:${toolName}] ❌ Not found in registry`);
      throw new Error(`Tool "${toolName}" not supported`);
    }

    try {
      // Get session's inference config to pass to tools
      const session = this.activeSessions.get(sessionId);
      const context = session ? { inferenceConfig: session.inferenceConfig } : undefined;
      
      // Parse tool parameters from the toolUse event
      // AWS Nova Sonic sends content as a JSON string in the toolUse event
      let toolParams: unknown = toolUseContent;
      const toolUseEvent = toolUseContent as { content?: string };
      
      if (toolUseEvent.content && typeof toolUseEvent.content === 'string') {
        try {
          toolParams = JSON.parse(toolUseEvent.content);
          console.log('[Tool:%s] Parsed params:', toolName, JSON.stringify(toolParams).substring(0, 200));
        } catch (parseError) {
          console.log(`[Tool:${toolName}] Using raw content as params`);
          toolParams = { content: toolUseEvent.content };
        }
      }
      
      const startTime = Date.now();
      const result = await this.toolRegistry.execute(toolName, toolParams, context);
      const duration = Date.now() - startTime;
      
      console.log(`[Tool:${toolName}] ✓ Completed in ${duration}ms`);
      return result as object;
    } catch (error) {
      console.error('[Tool:%s] ❌ Execution failed:', toolName, error instanceof Error ? error.message : error);
      throw error;
    }
  }

  // Execute tool asynchronously without blocking the response stream
  private async executeToolAsync(sessionId: string, toolUseId: string, toolName: string, toolUseContent: object): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const startTime = Date.now();
    
    try {
      const toolResult = await this.processToolUse(sessionId, toolName, toolUseContent);
      const duration = Date.now() - startTime;
      
      // Check if session is still active before sending result
      if (!session.isActive) {
        console.log(`[Tool:${toolName}] ⚠ Session inactive, cannot send result`);
        return;
      }
      
      // Send tool result
      await this.sendToolResult(sessionId, toolUseId, toolResult);
      
      console.log(`┌─── Tool Result ───────────────────────────────`);
      console.log(`│ Tool: ${toolName}`);
      console.log(`│ Status: ✓ Success`);
      console.log(`│ Duration: ${duration}ms`);
      console.log(`└───────────────────────────────────────────────\n`);
      
      // Dispatch event about tool result
      this.dispatchEvent(sessionId, 'toolResult', {
        toolUseId: toolUseId,
        result: toolResult,
        executionTimeMs: duration
      });
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMsg = error instanceof Error ? error.message : 'Tool execution failed';
      
      console.log(`┌─── Tool Result ───────────────────────────────`);
      console.log(`│ Tool: ${toolName}`);
      console.log(`│ Status: ❌ Failed`);
      console.log(`│ Error: ${errorMsg}`);
      console.log(`│ Duration: ${duration}ms`);
      console.log(`└───────────────────────────────────────────────\n`);
      
      const errorResult = {
        error: true,
        message: errorMsg
      };
      
      if (session.isActive) {
        await this.sendToolResult(sessionId, toolUseId, errorResult);
      }
      
      // Dispatch event about tool error result so UI can update
      this.dispatchEvent(sessionId, 'toolResult', {
        toolUseId: toolUseId,
        result: errorResult,
        executionTimeMs: duration,
        error: true
      });
    }
  }

  // Stream audio for a specific session
  public async initiateBidirectionalStreaming(sessionId: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Stream session ${sessionId} not found`);
    }

    try {

      // Create the bidirectional stream with session-specific async iterator
      const asyncIterable = this.createSessionAsyncIterable(sessionId);

      console.log(`Starting bidirectional stream for session ${sessionId}...`);

      const response = await this.bedrockRuntimeClient.send(
        new InvokeModelWithBidirectionalStreamCommand({
          modelId: NovaSonicModelId,
          body: asyncIterable,
        })
      );

      console.log(`Stream established for session ${sessionId}, processing responses...`);

      // Process responses for this session
      await this.processResponseStream(sessionId, response);

    } catch (error) {
      console.error('Error in session %s:', sessionId, error);
      this.dispatchEventForSession(sessionId, 'error', {
        source: 'bidirectionalStream',
        error
      });

      // Make sure to clean up if there's an error
      if (session.isActive) {
        this.closeSession(sessionId);
      }
    }
  }

  // Dispatch events to handlers for a specific session
  private dispatchEventForSession(sessionId: string, eventType: string, data: any): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const handler = session.responseHandlers.get(eventType);
    if (handler) {
      try {
        handler(data);
      } catch (e) {
        console.error('Error in %s handler for session %s:', eventType, sessionId, e);
      }
    }

    // Also dispatch to "any" handlers
    const anyHandler = session.responseHandlers.get('any');
    if (anyHandler) {
      try {
        anyHandler({ type: eventType, data });
      } catch (e) {
        console.error("Error in 'any' handler for session %s:", sessionId, e);
      }
    }
  }

  private createSessionAsyncIterable(sessionId: string): AsyncIterable<InvokeModelWithBidirectionalStreamInput> {

    if (!this.isSessionActive(sessionId)) {
      console.log(`Cannot create async iterable: Session ${sessionId} not active`);
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ value: undefined, done: true })
        })
      };
    }

    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Cannot create async iterable: Session ${sessionId} not found`);
    }

    let eventCount = 0;

    return {
      [Symbol.asyncIterator]: () => {
        console.log(`AsyncIterable iterator requested for session ${sessionId}`);

        return {
          next: async (): Promise<IteratorResult<InvokeModelWithBidirectionalStreamInput>> => {
            try {
              // Check if session is still active
              if (!session.isActive || !this.activeSessions.has(sessionId)) {
                console.log(`Iterator closing for session ${sessionId}, done = true`);
                return { value: undefined, done: true };
              }
              // Wait for items in the queue or close signal
              if (session.queue.length === 0) {
                try {
                  await Promise.race([
                    firstValueFrom(session.queueSignal.pipe(take(1))),
                    firstValueFrom(session.closeSignal.pipe(take(1))).then(() => {
                      throw new Error("Stream closed");
                    })
                  ]);
                } catch (error) {
                  if (error instanceof Error) {
                    if (error.message === "Stream closed" || !session.isActive) {
                      // This is an expected condition when closing the session
                      if (this.activeSessions.has(sessionId)) {
                        console.log(`Session ${ sessionId } closed during wait`);
                      }
                      return { value: undefined, done: true };
                    }
                  }
                  else {
                    console.error(`Error on event close`, error)
                  }
                }
              }

              // If queue is still empty or session is inactive, we're done
              if (session.queue.length === 0 || !session.isActive) {
                console.log(`Queue empty or session inactive: ${sessionId} `);
                return { value: undefined, done: true };
              }

              // Get next item from the session's queue
              const nextEvent = session.queue.shift();
              eventCount++;

              //console.log(`Sending event #${ eventCount } for session ${ sessionId }: ${ JSON.stringify(nextEvent).substring(0, 100) }...`);

              return {
                value: {
                  chunk: {
                    bytes: new TextEncoder().encode(JSON.stringify(nextEvent))
                  }
                },
                done: false
              };
            } catch (error) {
              console.error(`Error in session ${sessionId} iterator: `, error);
              session.isActive = false;
              return { value: undefined, done: true };
            }
          },

          return: async (): Promise<IteratorResult<InvokeModelWithBidirectionalStreamInput>> => {
            console.log(`Iterator return () called for session ${sessionId}`);
            session.isActive = false;
            return { value: undefined, done: true };
          },

          throw: async (error: any): Promise<IteratorResult<InvokeModelWithBidirectionalStreamInput>> => {
            console.log(`Iterator throw () called for session ${sessionId} with error: `, error);
            session.isActive = false;
            throw error;
          }
        };
      }
    };
  }

  // Process the response stream from AWS Bedrock
  private async processResponseStream(sessionId: string, response: any): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    try {
      for await (const event of response.body) {
        if (!session.isActive) {
          console.log(`Session ${sessionId} is no longer active, stopping response processing`);
          break;
        }
        if (event.chunk?.bytes) {
          try {
            this.updateSessionActivity(sessionId);
            const textResponse = new TextDecoder().decode(event.chunk.bytes);

            try {
              const jsonResponse = JSON.parse(textResponse);
              if (jsonResponse.event?.contentStart) {
                this.dispatchEvent(sessionId, 'contentStart', jsonResponse.event.contentStart);
              } else if (jsonResponse.event?.textOutput) {
                // Check for barge-in indicator in text content
                const textContent = jsonResponse.event.textOutput.content || '';
                if (textContent.includes('{ "interrupted" : true }') || textContent.includes('{"interrupted":true}')) {
                  console.log(`Barge-in detected for session ${sessionId}`);
                  this.dispatchEvent(sessionId, 'bargeIn', { interrupted: true });
                }
                this.dispatchEvent(sessionId, 'textOutput', jsonResponse.event.textOutput);
              } else if (jsonResponse.event?.audioOutput) {
                this.dispatchEvent(sessionId, 'audioOutput', jsonResponse.event.audioOutput);
              } else if (jsonResponse.event?.toolUse) {
                this.dispatchEvent(sessionId, 'toolUse', jsonResponse.event.toolUse);

                // Store tool use information for later
                session.toolUseContent = jsonResponse.event.toolUse;
                session.toolUseId = jsonResponse.event.toolUse.toolUseId;
                session.toolName = jsonResponse.event.toolUse.toolName;
              } else if (jsonResponse.event?.contentEnd &&
                jsonResponse.event?.contentEnd?.type === 'TOOL') {

                const toolUseId = session.toolUseId;
                const toolName = session.toolName;
                const toolUseContent = session.toolUseContent;
                const shortSessionId = sessionId.substring(0, 8);
                
                console.log(`\n┌─── Tool Invocation ───────────────────────────`);
                console.log(`│ Tool: ${toolName}`);
                console.log(`│ Session: ${shortSessionId}...`);
                console.log(`│ Tool Use ID: ${toolUseId}`);
                console.log(`└───────────────────────────────────────────────`);

                this.dispatchEvent(sessionId, 'toolEnd', {
                  toolUseContent: session.toolUseContent,
                  toolUseId: session.toolUseId,
                  toolName: session.toolName
                });

                // Execute tool asynchronously to avoid blocking the response stream
                // This prevents timeout errors during long-running tool executions
                this.executeToolAsync(sessionId, toolUseId, toolName, toolUseContent).catch(err => {
                  console.error('[Tool:%s] Async execution error:', toolName, err);
                })
                  
              } else if (jsonResponse.event?.contentEnd) {
                this.dispatchEvent(sessionId, 'contentEnd', jsonResponse.event.contentEnd);
              }
              else {
                // Handle other events
                const eventKeys = Object.keys(jsonResponse.event || {});
                if (eventKeys.length > 0) {
                  this.dispatchEvent(sessionId, eventKeys[0], jsonResponse.event);
                } else if (Object.keys(jsonResponse).length > 0) {
                  this.dispatchEvent(sessionId, 'unknown', jsonResponse);
                }
              }
            } catch (e) {
              console.log('Raw text response for session %s (parse error):', sessionId, textResponse);
            }
          } catch (e) {
            console.error('Error processing response chunk for session %s:', sessionId, e);
          }
        } else if (event.modelStreamErrorException) {
          console.error('Model stream error for session %s:', sessionId, event.modelStreamErrorException);
          const exceptionDetails = event.modelStreamErrorException?.message || JSON.stringify(event.modelStreamErrorException);
          this.dispatchEvent(sessionId, 'error', {
            type: 'modelStreamErrorException',
            source: 'responseStream',
            details: exceptionDetails
          });
        } else if (event.internalServerException) {
          console.error('Internal server error for session %s:', sessionId, event.internalServerException);
          const exceptionDetails = event.internalServerException?.message || JSON.stringify(event.internalServerException);
          this.dispatchEvent(sessionId, 'error', {
            type: 'internalServerException',
            source: 'responseStream',
            details: exceptionDetails
          });
        }
      }

      console.log(`Response stream processing complete for session ${sessionId}`);
      this.dispatchEvent(sessionId, 'streamComplete', {
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Error processing response stream for session %s:', sessionId, error);
      // Extract error message properly - handle Error instances, objects with message property, or stringify
      let errorDetails: string;
      if (error instanceof Error) {
        errorDetails = error.message;
      } else if (error && typeof error === 'object' && 'message' in error) {
        errorDetails = String((error as { message: unknown }).message);
      } else {
        errorDetails = JSON.stringify(error);
      }
      this.dispatchEvent(sessionId, 'error', {
        source: 'responseStream',
        message: 'Error processing response stream',
        details: errorDetails
      });
    }
  }

  // Add an event to a session's queue
  private addEventToSessionQueue(sessionId: string, event: any): void {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isActive) return;

    this.updateSessionActivity(sessionId);
    session.queue.push(event);
    session.queueSignal.next();
  }


  // Set up initial events for a session
  public setupSessionStartEvent(sessionId: string): void {
    console.log(`Setting up initial events for session ${sessionId}...`);
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    // Build session start event with inference config and turn detection
    const sessionStartEvent: any = {
      event: {
        sessionStart: {
          inferenceConfiguration: session.inferenceConfig
        }
      }
    };

    // Add turn detection if configured
    if (session.turnDetectionConfig?.endpointingSensitivity) {
      sessionStartEvent.event.sessionStart.turnDetectionConfiguration = {
        endpointingSensitivity: session.turnDetectionConfig.endpointingSensitivity
      };
    }

    console.log('Session start event for %s:', sessionId, JSON.stringify(sessionStartEvent, null, 2));
    this.addEventToSessionQueue(sessionId, sessionStartEvent);
  }

  public setupPromptStartEvent(sessionId: string, voiceId?: string, outputSampleRate: number = 24000): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    // Build audio output config - include all required fields for Nova Sonic
    const audioOutputConfig = {
      mediaType: "audio/lpcm" as const,
      sampleRateHertz: outputSampleRate,
      sampleSizeBits: 16,
      channelCount: 1,
      voiceId: voiceId || DefaultAudioOutputConfiguration.voiceId
    };

    // Get tool specs, filtered by enabledTools if specified
    let toolSpecs = this.toolRegistry.getToolSpecs();
    if (session.enabledTools && session.enabledTools.length > 0) {
      toolSpecs = toolSpecs.filter(t => session.enabledTools!.includes(t.toolSpec.name));
      console.log(`Filtered tools for session ${sessionId}: ${session.enabledTools.join(', ')}`);
    }

    // Build tool configuration per AWS docs
    // toolChoice: "auto" (default) - model decides when to use tools
    // toolChoice: "any" - model must use at least one tool
    // toolChoice: { tool: { name: "toolName" } } - force specific tool
    const toolConfiguration: any = {
      tools: toolSpecs,
      toolChoice: session.toolChoice || { auto: {} }
    };

    const promptStartEvent = {
      event: {
        promptStart: {
          promptName: session.promptName,
          textOutputConfiguration: {
            mediaType: "text/plain"
          },
          audioOutputConfiguration: audioOutputConfig,
          toolUseOutputConfiguration: {
            mediaType: "application/json"
          },
          toolConfiguration
        }
      }
    };

    this.addEventToSessionQueue(sessionId, promptStartEvent);
    session.isPromptStartSent = true;
    console.log(`Prompt start completed for ${sessionId} with sample rate ${outputSampleRate}, ${toolSpecs.length} tools configured`);
  }

  public setupSystemPromptEvent(sessionId: string,
    textConfig: typeof DefaultTextConfiguration = DefaultTextConfiguration,
    systemPromptContent: string
  ): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;
    
    // Ensure system prompt is never empty
    const promptContent = systemPromptContent?.trim();
    if (!promptContent) {
      throw new Error('System prompt content is required');
    }
    
    const textPromptID = randomUUID();
    
    // Text content start
    const contentStartEvent = {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: textPromptID,
          type: "TEXT",
          interactive: false,
          role: "SYSTEM",
          textInputConfiguration: {
            mediaType: "text/plain"
          }
        }
      }
    };
    console.log('System prompt contentStart for %s:', sessionId, JSON.stringify(contentStartEvent, null, 2));
    this.addEventToSessionQueue(sessionId, contentStartEvent);

    // Text input content
    const textInputEvent = {
      event: {
        textInput: {
          promptName: session.promptName,
          contentName: textPromptID,
          content: promptContent
        }
      }
    };
    console.log(`System prompt textInput for ${sessionId}: content length=${promptContent.length}`);
    this.addEventToSessionQueue(sessionId, textInputEvent);

    // Text content end
    const contentEndEvent = {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: textPromptID
        }
      }
    };
    console.log('System prompt contentEnd for %s:', sessionId, JSON.stringify(contentEndEvent, null, 2));
    this.addEventToSessionQueue(sessionId, contentEndEvent);
  }

  public setupStartAudioEvent(
    sessionId: string,
    audioConfig: typeof DefaultAudioInputConfiguration = DefaultAudioInputConfiguration
  ): void {
    console.log(`Setting up startAudioContent event for session ${sessionId}...`);
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    console.log(`Using audio content ID: ${session.audioContentId}`);
    // Audio content start
    const audioContentStartEvent = {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: session.audioContentId,
          type: "AUDIO",
          interactive: true,
          role: "USER",
          audioInputConfiguration: audioConfig,
        },
      }
    };
    console.log('Audio contentStart for %s:', sessionId, JSON.stringify(audioContentStartEvent, null, 2));
    this.addEventToSessionQueue(sessionId, audioContentStartEvent);
    session.isAudioContentStartSent = true;
    console.log(`Initial events setup complete for session ${sessionId}`);
  }

  // Send text input event (for typing mode)
  public sendTextInputEvent(sessionId: string, textContent: string): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      console.error(`No session found for ${sessionId}`);
      return;
    }

    const textContentId = randomUUID();

    // Text content start
    const contentStartEvent = {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: textContentId,
          type: "TEXT",
          interactive: true,
          role: "USER",
          textInputConfiguration: {
            mediaType: "text/plain"
          }
        }
      }
    };
    this.addEventToSessionQueue(sessionId, contentStartEvent);

    // Text input content
    const textInputEvent = {
      event: {
        textInput: {
          promptName: session.promptName,
          contentName: textContentId,
          content: textContent
        }
      }
    };
    this.addEventToSessionQueue(sessionId, textInputEvent);

    // Text content end
    const contentEndEvent = {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: textContentId
        }
      }
    };
    this.addEventToSessionQueue(sessionId, contentEndEvent);
    
    console.log(`Text input sent for session ${sessionId}: "${textContent.substring(0, 50)}${textContent.length > 50 ? '...' : ''}"`);
  }

  // Stream an audio chunk for a session
  public async streamAudioChunk(sessionId: string, audioData: Buffer): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isActive || !session.audioContentId) {
      throw new Error(`Invalid session ${sessionId} for audio streaming`);
    }
    // Convert audio to base64
    const base64Data = audioData.toString('base64');

    this.addEventToSessionQueue(sessionId, {
      event: {
        audioInput: {
          promptName: session.promptName,
          contentName: session.audioContentId,
          content: base64Data,
        },
      }
    });
  }


  // Send tool result back to the model
  private async sendToolResult(sessionId: string, toolUseId: string, result: any): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isActive) {
      console.log(`[ToolResult] ⚠ Cannot send - session ${sessionId.substring(0, 8)}... inactive`);
      return;
    }

    const contentId = randomUUID();

    // Tool content start
    this.addEventToSessionQueue(sessionId, {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: contentId,
          interactive: false,
          type: "TOOL",
          role: "TOOL",
          toolResultInputConfiguration: {
            toolUseId: toolUseId,
            type: "TEXT",
            textInputConfiguration: {
              mediaType: "text/plain"
            }
          }
        }
      }
    });

    // Small delay to ensure contentStart is processed first
    await new Promise(resolve => setTimeout(resolve, 50));

    // Tool content input - sanitize and truncate content
    const resultContent = typeof result === 'string' ? result : JSON.stringify(result);
    // Remove control characters and limit length to prevent issues
    let sanitizedContent = resultContent.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, '');
    
    // Truncate if too long
    if (sanitizedContent.length > DefaultToolConfiguration.maxResultLength) {
      console.log(`Tool result truncated from ${sanitizedContent.length} to ${DefaultToolConfiguration.maxResultLength} chars`);
      sanitizedContent = sanitizedContent.substring(0, DefaultToolConfiguration.maxResultLength) + '... (truncated)';
    }
    
    console.log(`[ToolResult] Sending ${sanitizedContent.length} chars to model`);
    
    this.addEventToSessionQueue(sessionId, {
      event: {
        toolResult: {
          promptName: session.promptName,
          contentName: contentId,
          content: sanitizedContent
        }
      }
    });

    // Small delay before contentEnd
    await new Promise(resolve => setTimeout(resolve, 50));

    // Tool content end
    this.addEventToSessionQueue(sessionId, {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: contentId
        }
      }
    });

    // Wait for events to be processed
    await new Promise(resolve => setTimeout(resolve, 100));
    console.log(`[ToolResult] ✓ Sent to model for session ${sessionId.substring(0, 8)}...`);
  }

  public async sendContentEnd(sessionId: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isAudioContentStartSent) return;

    await this.addEventToSessionQueue(sessionId, {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: session.audioContentId,
        }
      }
    });

    // Wait to ensure it's processed
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  public async sendPromptEnd(sessionId: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isPromptStartSent) return;

    await this.addEventToSessionQueue(sessionId, {
      event: {
        promptEnd: {
          promptName: session.promptName
        }
      }
    });

    // Wait to ensure it's processed
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  public async sendSessionEnd(sessionId: string): Promise<void> {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    await this.addEventToSessionQueue(sessionId, {
      event: {
        sessionEnd: {}
      }
    });

    // Wait to ensure it's processed
    await new Promise(resolve => setTimeout(resolve, 300));

    // Now it's safe to clean up
    session.isActive = false;
    session.closeSignal.next();
    session.closeSignal.complete();
    this.activeSessions.delete(sessionId);
    this.sessionLastActivity.delete(sessionId);
    console.log(`Session ${sessionId} closed and removed from active sessions`);
  }

  // Register an event handler for a session
  public registerEventHandler(sessionId: string, eventType: string, handler: (data: any) => void): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.responseHandlers.set(eventType, handler);
  }

  // Dispatch an event to registered handlers
  private dispatchEvent(sessionId: string, eventType: string, data: any): void {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const handler = session.responseHandlers.get(eventType);
    if (handler) {
      try {
        handler(data);
      } catch (e) {
        console.error('Error in %s handler for session %s:', eventType, sessionId, e);
      }
    }

    // Also dispatch to "any" handlers
    const anyHandler = session.responseHandlers.get('any');
    if (anyHandler) {
      try {
        anyHandler({ type: eventType, data });
      } catch (e) {
        console.error("Error in 'any' handler for session %s:", sessionId, e);
      }
    }
  }

  public async closeSession(sessionId: string): Promise<void> {
    if (this.sessionCleanupInProgress.has(sessionId)) {
      console.log(`Cleanup already in progress for session ${sessionId}, skipping`);
      return;
    }
    this.sessionCleanupInProgress.add(sessionId);
    try {
      console.log(`Starting close process for session ${sessionId}`);
      await this.sendContentEnd(sessionId);
      await this.sendPromptEnd(sessionId);
      await this.sendSessionEnd(sessionId);
      console.log(`Session ${sessionId} cleanup complete`);
    } catch (error) {
      console.error('Error during closing sequence for session %s:', sessionId, error);

      // Ensure cleanup happens even if there's an error
      const session = this.activeSessions.get(sessionId);
      if (session) {
        session.isActive = false;
        this.activeSessions.delete(sessionId);
        this.sessionLastActivity.delete(sessionId);
      }
    } finally {
      // Always clean up the tracking set
      this.sessionCleanupInProgress.delete(sessionId);
    }
  }

  // Same for forceCloseSession:
  public forceCloseSession(sessionId: string): void {
    if (this.sessionCleanupInProgress.has(sessionId) || !this.activeSessions.has(sessionId)) {
      console.log(`Session ${sessionId} already being cleaned up or not active`);
      return;
    }

    this.sessionCleanupInProgress.add(sessionId);
    try {
      const session = this.activeSessions.get(sessionId);
      if (!session) return;

      console.log(`Force closing session ${sessionId}`);

      // Immediately mark as inactive and clean up resources
      session.isActive = false;
      session.closeSignal.next();
      session.closeSignal.complete();
      this.activeSessions.delete(sessionId);
      this.sessionLastActivity.delete(sessionId);

      console.log(`Session ${sessionId} force closed`);
    } finally {
      this.sessionCleanupInProgress.delete(sessionId);
    }
  }

}