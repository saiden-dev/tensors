import type { Model, LoRA, GeneratedImage, GalleryImage, CivitaiModel } from '@/types'

const BASE_URL = ''

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }))
    throw new Error(error.error || response.statusText)
  }
  return response.json()
}

// Models
export async function getModels(): Promise<{ models: Model[]; total: number }> {
  return fetchJson('/api/models')
}

export async function getActiveModel(): Promise<{ loaded: boolean; model: string | null }> {
  return fetchJson('/api/models/active')
}

export async function switchModel(model: string): Promise<{ ok: boolean }> {
  return fetchJson('/api/models/switch', {
    method: 'POST',
    body: JSON.stringify({ model }),
  })
}

export async function getLoras(): Promise<{ loras: LoRA[]; total: number }> {
  return fetchJson('/api/models/loras')
}

// Generation
export interface LoraConfig {
  path: string
  multiplier: number
}

export interface GenerateParams {
  prompt: string
  negative_prompt?: string
  width: number
  height: number
  steps: number
  cfg_scale?: number
  seed?: number
  save_to_gallery?: boolean
  lora?: LoraConfig
}

export async function generate(params: GenerateParams): Promise<{ images: GeneratedImage[] }> {
  return fetchJson('/api/generate', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

// Gallery
export async function getImages(limit = 100): Promise<{ images: GalleryImage[] }> {
  return fetchJson(`/api/images?limit=${limit}`)
}

export async function deleteImage(id: string): Promise<void> {
  await fetchJson(`/api/images/${id}`, { method: 'DELETE' })
}

export function getImageUrl(id: string): string {
  return `${BASE_URL}/api/images/${id}`
}

// CivitAI Search
export interface SearchParams {
  query?: string
  types?: string
  baseModels?: string
  sort?: string
  limit?: number
}

export async function searchCivitai(params: SearchParams): Promise<{ items: CivitaiModel[] }> {
  const searchParams = new URLSearchParams()
  if (params.query) searchParams.set('query', params.query)
  if (params.types) searchParams.set('types', params.types)
  if (params.baseModels) searchParams.set('baseModels', params.baseModels)
  if (params.sort) searchParams.set('sort', params.sort)
  if (params.limit) searchParams.set('limit', String(params.limit))

  return fetchJson(`/api/civitai/search?${searchParams}`)
}

export async function getCivitaiModel(id: number): Promise<CivitaiModel> {
  return fetchJson(`/api/civitai/model/${id}`)
}

// Download
export interface DownloadResponse {
  download_id: string
  status: string
  version_id: number
  destination: string
  model_name: string
  version_name: string
}

export interface DownloadStatus {
  id: string
  version_id: number
  status: 'queued' | 'downloading' | 'completed' | 'failed'
  path: string
  filename: string
  model_name: string
  version_name: string
  downloaded?: number
  total?: number
  progress?: number
  speed?: number
  downloaded_str?: string
  total_str?: string
  speed_str?: string
  error?: string
}

export async function downloadModel(modelId?: number, versionId?: number): Promise<DownloadResponse> {
  return fetchJson('/api/download', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, version_id: versionId }),
  })
}

export async function getDownloadStatus(downloadId: string): Promise<DownloadStatus> {
  return fetchJson(`/api/download/status/${downloadId}`)
}

export async function getActiveDownloads(): Promise<{ downloads: DownloadStatus[]; total: number }> {
  return fetchJson('/api/download/active')
}
