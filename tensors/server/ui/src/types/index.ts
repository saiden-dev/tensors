export interface Model {
  name: string
  path: string
  filename: string
  size_mb: number
  modified: number
  category: 'sd15' | 'large'
}

export interface LoRA {
  name: string
  path: string
  filename: string
  size_mb: number
  modified: number
  category: 'sd15' | 'large'
}

export interface GeneratedImage {
  id: string
  path: string
  seed: number
}

export interface GalleryImage {
  id: string
  filename: string
  created: string
  metadata?: ImageMetadata
}

export interface ImageMetadata {
  prompt?: string
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  seed?: number
  sampler?: string
}

export interface CivitaiModel {
  id: number
  name: string
  description?: string
  type: string
  nsfw: boolean
  creator?: {
    username: string
    image?: string
  }
  stats?: {
    downloadCount: number
    thumbsUpCount: number
  }
  modelVersions?: CivitaiVersion[]
}

export interface CivitaiVersion {
  id: number
  name: string
  baseModel?: string
  trainedWords?: string[]
  images?: CivitaiImage[]
  downloadUrl?: string
}

export interface CivitaiImage {
  url: string
  width: number
  height: number
  nsfwLevel: number
}

export interface Resolution {
  width: number
  height: number
  label: string
}

export interface ResolutionPreset {
  id: string
  ratio: string
  width: number
  height: number
  label: string
}
