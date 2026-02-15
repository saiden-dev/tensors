/**
 * ComfyUI API Type Definitions
 */

// ============================================================
// System Types
// ============================================================

export interface SystemStats {
  system: {
    os: string;
    ram_total: number;
    ram_free: number;
    comfyui_version: string;
    required_frontend_version: string;
    installed_templates_version: string;
    required_templates_version: string;
    python_version: string;
    pytorch_version: string;
    embedded_python: boolean;
    argv: string[];
  };
  devices: DeviceInfo[];
}

export interface DeviceInfo {
  name: string;
  type: string;
  index: number;
  vram_total: number;
  vram_free: number;
  torch_vram_total: number;
  torch_vram_free: number;
}

// ============================================================
// Queue Types
// ============================================================

export interface QueueStatus {
  queue_running: QueueItem[];
  queue_pending: QueueItem[];
}

export interface QueueItem {
  prompt_id: string;
  number: number;
  prompt: WorkflowPrompt;
  extra_data: Record<string, unknown>;
  outputs_to_execute: string[];
}

// ============================================================
// Workflow / Prompt Types
// ============================================================

export type WorkflowPrompt = Record<string, WorkflowNode>;

export interface WorkflowNode {
  class_type: string;
  inputs: Record<string, unknown>;
  _meta?: {
    title?: string;
  };
}

export interface PromptRequest {
  prompt: WorkflowPrompt;
  client_id?: string;
  extra_data?: {
    extra_pnginfo?: Record<string, unknown>;
  };
}

export interface PromptResponse {
  prompt_id: string;
  number: number;
  node_errors?: Record<string, NodeError>;
}

export interface NodeError {
  type: string;
  message: string;
  details: string;
  extra_info: Record<string, unknown>;
}

// ============================================================
// History Types
// ============================================================

export type HistoryResponse = Record<string, HistoryEntry>;

export interface HistoryEntry {
  prompt: [number, string, WorkflowPrompt, Record<string, unknown>, string[]];
  outputs: Record<string, NodeOutput>;
  status: {
    status_str: string;
    completed: boolean;
    messages: Array<[string, Record<string, unknown>]>;
  };
}

export interface NodeOutput {
  images?: ImageOutput[];
  [key: string]: unknown;
}

export interface ImageOutput {
  filename: string;
  subfolder: string;
  type: string;
}

// ============================================================
// Object Info Types (Node Definitions)
// ============================================================

export type ObjectInfo = Record<string, NodeDefinition>;

export interface NodeDefinition {
  input: {
    required?: Record<string, InputDefinition>;
    optional?: Record<string, InputDefinition>;
    hidden?: Record<string, InputDefinition>;
  };
  input_order?: {
    required?: string[];
    optional?: string[];
  };
  output: string[];
  output_is_list: boolean[];
  output_name: string[];
  name: string;
  display_name: string;
  description: string;
  python_module: string;
  category: string;
  output_node: boolean;
  deprecated: boolean;
  experimental: boolean;
}

export type InputDefinition =
  | [string, InputOptions?]      // Type reference with options
  | [string[], InputOptions?];   // Enum choices with options

export interface InputOptions {
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  round?: number;
  tooltip?: string;
  multiline?: boolean;
  dynamicPrompts?: boolean;
  control_after_generate?: boolean;
  forceInput?: boolean;
}

// ============================================================
// WebSocket Message Types
// ============================================================

export type ComfyMessage =
  | StatusMessage
  | ProgressMessage
  | ExecutingMessage
  | ExecutedMessage
  | ExecutionStartMessage
  | ExecutionCachedMessage
  | ExecutionErrorMessage;

export interface StatusMessage {
  type: 'status';
  data: {
    status: {
      exec_info: {
        queue_remaining: number;
      };
    };
    sid?: string;
  };
}

export interface ProgressMessage {
  type: 'progress';
  data: {
    value: number;
    max: number;
    prompt_id: string;
    node: string;
  };
}

export interface ExecutingMessage {
  type: 'executing';
  data: {
    node: string | null;
    prompt_id: string;
    display_node?: string;
  };
}

export interface ExecutedMessage {
  type: 'executed';
  data: {
    node: string;
    display_node: string;
    output: NodeOutput;
    prompt_id: string;
  };
}

export interface ExecutionStartMessage {
  type: 'execution_start';
  data: {
    prompt_id: string;
    timestamp: number;
  };
}

export interface ExecutionCachedMessage {
  type: 'execution_cached';
  data: {
    nodes: string[];
    prompt_id: string;
    timestamp: number;
  };
}

export interface ExecutionErrorMessage {
  type: 'execution_error';
  data: {
    prompt_id: string;
    node_id: string;
    node_type: string;
    exception_message: string;
    exception_type: string;
    traceback: string[];
    current_inputs?: Record<string, unknown>;
    current_outputs?: Record<string, unknown>[];
  };
}

// ============================================================
// Image View Types
// ============================================================

export interface ViewImageParams {
  filename: string;
  subfolder?: string;
  type?: 'output' | 'input' | 'temp';
  preview?: string;
  channel?: string;
}

// ============================================================
// Models Types
// ============================================================

export interface ModelsResponse {
  checkpoints?: string[];
  loras?: string[];
  vae?: string[];
  controlnet?: string[];
  upscale_models?: string[];
  embeddings?: string[];
  hypernetworks?: string[];
  clip?: string[];
  clip_vision?: string[];
  style_models?: string[];
  diffusers?: string[];
  gligen?: string[];
  diffusion_models?: string[];
  unet?: string[];
  photomaker?: string[];
}

// ============================================================
// Upload Types
// ============================================================

export interface UploadImageResponse {
  name: string;
  subfolder: string;
  type: string;
}

// ============================================================
// Client Configuration
// ============================================================

export interface ComfyClientConfig {
  baseUrl: string;
  clientId?: string;
}

// ============================================================
// Event Types
// ============================================================

export interface ComfyEventMap {
  status: StatusMessage['data'];
  progress: ProgressMessage['data'];
  executing: ExecutingMessage['data'];
  executed: ExecutedMessage['data'];
  execution_start: ExecutionStartMessage['data'];
  execution_cached: ExecutionCachedMessage['data'];
  execution_error: ExecutionErrorMessage['data'];
  connected: undefined;
  disconnected: undefined;
  error: Error;
}

export type ComfyEventHandler<K extends keyof ComfyEventMap> = (
  data: ComfyEventMap[K]
) => void;
