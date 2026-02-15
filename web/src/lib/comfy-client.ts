/**
 * ComfyUI WebSocket/REST Client
 *
 * A TypeScript client for interacting with the ComfyUI API.
 *
 * Example usage:
 *   const client = new ComfyClient({ baseUrl: 'http://localhost:8188' });
 *   await client.connect();
 *
 *   client.on('progress', (data) => console.log(`Progress: ${data.value}/${data.max}`));
 *   client.on('executed', (data) => console.log('Generated:', data.output));
 *
 *   const { prompt_id } = await client.queuePrompt(workflow);
 *   // Listen for events...
 */

import type {
  ComfyClientConfig,
  ComfyEventHandler,
  ComfyEventMap,
  ComfyMessage,
  HistoryEntry,
  HistoryResponse,
  ModelsResponse,
  ObjectInfo,
  PromptRequest,
  PromptResponse,
  QueueStatus,
  SystemStats,
  UploadImageResponse,
  ViewImageParams,
  WorkflowPrompt,
} from '../types/comfy';

type Listener<K extends keyof ComfyEventMap> = {
  event: K;
  handler: ComfyEventHandler<K>;
};

export class ComfyClient {
  private config: ComfyClientConfig;
  private ws: WebSocket | null = null;
  private listeners: Listener<keyof ComfyEventMap>[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private shouldReconnect = true;

  constructor(config: ComfyClientConfig) {
    this.config = {
      ...config,
      clientId: config.clientId ?? crypto.randomUUID(),
    };
  }

  // ============================================================
  // Connection Management
  // ============================================================

  get clientId(): string {
    return this.config.clientId!;
  }

  get baseUrl(): string {
    return this.config.baseUrl;
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Connect to ComfyUI WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnected) {
        resolve();
        return;
      }

      const wsUrl = this.config.baseUrl
        .replace(/^http/, 'ws')
        .replace(/\/$/, '');

      this.ws = new WebSocket(`${wsUrl}/ws?clientId=${this.clientId}`);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.emit('connected', undefined);
        resolve();
      };

      this.ws.onclose = () => {
        this.emit('disconnected', undefined);
        this.handleReconnect();
      };

      this.ws.onerror = (event) => {
        const error = new Error('WebSocket error');
        this.emit('error', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  /**
   * Disconnect from ComfyUI WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private handleReconnect(): void {
    if (!this.shouldReconnect) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.emit('error', new Error('Max reconnection attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    setTimeout(() => {
      this.connect().catch(() => {
        // Reconnect will be attempted again via onclose
      });
    }, delay);
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data) as ComfyMessage;
      this.emit(message.type, message.data as ComfyEventMap[typeof message.type]);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  // ============================================================
  // Event Handling
  // ============================================================

  on<K extends keyof ComfyEventMap>(
    event: K,
    handler: ComfyEventHandler<K>
  ): () => void {
    const listener = { event, handler } as Listener<keyof ComfyEventMap>;
    this.listeners.push(listener);

    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index !== -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  off<K extends keyof ComfyEventMap>(
    event: K,
    handler?: ComfyEventHandler<K>
  ): void {
    this.listeners = this.listeners.filter((listener) => {
      if (listener.event !== event) return true;
      if (handler && listener.handler !== handler) return true;
      return false;
    });
  }

  private emit<K extends keyof ComfyEventMap>(
    event: K,
    data: ComfyEventMap[K]
  ): void {
    for (const listener of this.listeners) {
      if (listener.event === event) {
        (listener.handler as ComfyEventHandler<K>)(data);
      }
    }
  }

  // ============================================================
  // HTTP Helpers
  // ============================================================

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text}`);
    }

    return response.json();
  }

  // ============================================================
  // System API
  // ============================================================

  /**
   * Get system statistics (RAM, VRAM, versions)
   */
  async getSystemStats(): Promise<SystemStats> {
    return this.fetch<SystemStats>('/system_stats');
  }

  /**
   * Get node definitions (object_info)
   */
  async getObjectInfo(): Promise<ObjectInfo> {
    return this.fetch<ObjectInfo>('/object_info');
  }

  /**
   * Get specific node definition
   */
  async getNodeInfo(nodeType: string): Promise<ObjectInfo> {
    return this.fetch<ObjectInfo>(`/object_info/${encodeURIComponent(nodeType)}`);
  }

  // ============================================================
  // Queue API
  // ============================================================

  /**
   * Get current queue status
   */
  async getQueue(): Promise<QueueStatus> {
    return this.fetch<QueueStatus>('/queue');
  }

  /**
   * Queue a prompt/workflow for execution
   */
  async queuePrompt(
    prompt: WorkflowPrompt,
    options: Partial<Omit<PromptRequest, 'prompt'>> = {}
  ): Promise<PromptResponse> {
    const request: PromptRequest = {
      prompt,
      client_id: options.client_id ?? this.clientId,
      extra_data: options.extra_data,
    };

    return this.fetch<PromptResponse>('/prompt', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Delete a queued item
   */
  async deleteQueueItem(deleteType: 'queue' | 'history', promptId: string): Promise<void> {
    await this.fetch(`/${deleteType}`, {
      method: 'POST',
      body: JSON.stringify({ delete: [promptId] }),
    });
  }

  /**
   * Clear the queue
   */
  async clearQueue(): Promise<void> {
    await this.fetch('/queue', {
      method: 'POST',
      body: JSON.stringify({ clear: true }),
    });
  }

  /**
   * Interrupt the current generation
   */
  async interrupt(): Promise<void> {
    await this.fetch('/interrupt', {
      method: 'POST',
    });
  }

  // ============================================================
  // History API
  // ============================================================

  /**
   * Get generation history
   */
  async getHistory(maxItems?: number): Promise<HistoryResponse> {
    const endpoint = maxItems ? `/history?max_items=${maxItems}` : '/history';
    return this.fetch<HistoryResponse>(endpoint);
  }

  /**
   * Get a specific history entry
   */
  async getHistoryEntry(promptId: string): Promise<HistoryEntry | undefined> {
    const history = await this.fetch<HistoryResponse>(
      `/history/${encodeURIComponent(promptId)}`
    );
    return history[promptId];
  }

  /**
   * Clear history
   */
  async clearHistory(): Promise<void> {
    await this.fetch('/history', {
      method: 'POST',
      body: JSON.stringify({ clear: true }),
    });
  }

  // ============================================================
  // Image API
  // ============================================================

  /**
   * Get URL for viewing an image
   */
  getImageUrl(params: ViewImageParams): string {
    const searchParams = new URLSearchParams();
    searchParams.set('filename', params.filename);
    if (params.subfolder) searchParams.set('subfolder', params.subfolder);
    if (params.type) searchParams.set('type', params.type);
    if (params.preview) searchParams.set('preview', params.preview);
    if (params.channel) searchParams.set('channel', params.channel);

    return `${this.config.baseUrl}/view?${searchParams.toString()}`;
  }

  /**
   * Upload an image
   */
  async uploadImage(
    file: File | Blob,
    filename: string,
    options: {
      overwrite?: boolean;
      subfolder?: string;
      type?: 'input' | 'temp';
    } = {}
  ): Promise<UploadImageResponse> {
    const formData = new FormData();
    formData.append('image', file, filename);
    if (options.overwrite) formData.append('overwrite', 'true');
    if (options.subfolder) formData.append('subfolder', options.subfolder);
    if (options.type) formData.append('type', options.type);

    const response = await fetch(`${this.config.baseUrl}/upload/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    return response.json();
  }

  // ============================================================
  // Models API (custom endpoints - may need ComfyUI-Manager)
  // ============================================================

  /**
   * Get list of available models by category
   * Note: Uses internal folder structure, not a standard endpoint
   */
  async getModels(): Promise<ModelsResponse> {
    // ComfyUI doesn't have a direct /models endpoint.
    // We get model lists from object_info for specific nodes
    const objectInfo = await this.getObjectInfo();
    const models: ModelsResponse = {};

    // Extract checkpoint list from CheckpointLoaderSimple
    const checkpointLoader = objectInfo['CheckpointLoaderSimple'];
    if (checkpointLoader?.input?.required?.['ckpt_name']) {
      const input = checkpointLoader.input.required['ckpt_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        models.checkpoints = input[0] as string[];
      }
    }

    // Extract LoRA list from LoraLoader
    const loraLoader = objectInfo['LoraLoader'];
    if (loraLoader?.input?.required?.['lora_name']) {
      const input = loraLoader.input.required['lora_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        models.loras = input[0] as string[];
      }
    }

    // Extract VAE list from VAELoader
    const vaeLoader = objectInfo['VAELoader'];
    if (vaeLoader?.input?.required?.['vae_name']) {
      const input = vaeLoader.input.required['vae_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        models.vae = input[0] as string[];
      }
    }

    // Extract upscaler list from UpscaleModelLoader
    const upscaleLoader = objectInfo['UpscaleModelLoader'];
    if (upscaleLoader?.input?.required?.['model_name']) {
      const input = upscaleLoader.input.required['model_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        models.upscale_models = input[0] as string[];
      }
    }

    // Extract ControlNet list from ControlNetLoader
    const controlnetLoader = objectInfo['ControlNetLoader'];
    if (controlnetLoader?.input?.required?.['control_net_name']) {
      const input = controlnetLoader.input.required['control_net_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        models.controlnet = input[0] as string[];
      }
    }

    // Extract embeddings from object_info
    const clipLoader = objectInfo['CLIPTextEncode'];
    if (clipLoader) {
      // Embeddings are typically shown in the text field tooltips or separate endpoint
      // This is a simplification - full embedding list requires different approach
    }

    return models;
  }

  /**
   * Get list of samplers
   */
  async getSamplers(): Promise<string[]> {
    const objectInfo = await this.getObjectInfo();
    const ksampler = objectInfo['KSampler'];
    if (ksampler?.input?.required?.['sampler_name']) {
      const input = ksampler.input.required['sampler_name'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        return input[0] as string[];
      }
    }
    return [];
  }

  /**
   * Get list of schedulers
   */
  async getSchedulers(): Promise<string[]> {
    const objectInfo = await this.getObjectInfo();
    const ksampler = objectInfo['KSampler'];
    if (ksampler?.input?.required?.['scheduler']) {
      const input = ksampler.input.required['scheduler'];
      if (Array.isArray(input) && Array.isArray(input[0])) {
        return input[0] as string[];
      }
    }
    return [];
  }

  // ============================================================
  // Utility Methods
  // ============================================================

  /**
   * Wait for a prompt to complete
   * Returns the history entry when done
   */
  async waitForPrompt(
    promptId: string,
    options: {
      onProgress?: (value: number, max: number) => void;
      onExecuting?: (node: string | null) => void;
      timeout?: number;
    } = {}
  ): Promise<HistoryEntry> {
    return new Promise((resolve, reject) => {
      const timeout = options.timeout ?? 300000; // 5 minutes default
      let timeoutId: ReturnType<typeof setTimeout>;

      const cleanup = () => {
        clearTimeout(timeoutId);
        unsubProgress();
        unsubExecuting();
        unsubExecuted();
        unsubError();
      };

      timeoutId = setTimeout(() => {
        cleanup();
        reject(new Error(`Prompt ${promptId} timed out after ${timeout}ms`));
      }, timeout);

      const unsubProgress = this.on('progress', (data) => {
        if (data.prompt_id === promptId) {
          options.onProgress?.(data.value, data.max);
        }
      });

      const unsubExecuting = this.on('executing', (data) => {
        if (data.prompt_id === promptId) {
          options.onExecuting?.(data.node);

          // When node is null, execution is complete
          if (data.node === null) {
            cleanup();
            this.getHistoryEntry(promptId)
              .then((entry) => {
                if (entry) {
                  resolve(entry);
                } else {
                  reject(new Error(`No history entry for prompt ${promptId}`));
                }
              })
              .catch(reject);
          }
        }
      });

      const unsubExecuted = this.on('executed', (data) => {
        // Individual node executed - could be useful for streaming results
        // but main completion is signaled by executing with node=null
      });

      const unsubError = this.on('execution_error', (data) => {
        if (data.prompt_id === promptId) {
          cleanup();
          reject(
            new Error(
              `Execution error in ${data.node_type}: ${data.exception_message}`
            )
          );
        }
      });
    });
  }

  /**
   * Queue a prompt and wait for completion
   */
  async generate(
    prompt: WorkflowPrompt,
    options: {
      onProgress?: (value: number, max: number) => void;
      onExecuting?: (node: string | null) => void;
      timeout?: number;
    } = {}
  ): Promise<HistoryEntry> {
    const { prompt_id } = await this.queuePrompt(prompt);
    return this.waitForPrompt(prompt_id, options);
  }
}

// ============================================================
// Singleton / Factory
// ============================================================

let defaultClient: ComfyClient | null = null;

/**
 * Get or create the default ComfyUI client
 */
export function getComfyClient(config?: ComfyClientConfig): ComfyClient {
  if (!defaultClient && !config) {
    throw new Error('ComfyClient not initialized. Provide config on first call.');
  }

  if (config) {
    defaultClient = new ComfyClient(config);
  }

  return defaultClient!;
}

export default ComfyClient;
