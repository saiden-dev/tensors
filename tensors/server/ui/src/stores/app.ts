import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Model, LoRA, ResolutionPreset } from '@/types'
import * as api from '@/api/client'

export const useAppStore = defineStore('app', () => {
  // Navigation
  const currentView = ref<'generate' | 'search' | 'gallery'>('generate')

  // Models
  const models = ref<Model[]>([])
  const loras = ref<LoRA[]>([])
  const activeModel = ref<string | null>(null)
  const selectedModel = ref<string>('')
  const selectedLora = ref<string>('')
  const loraWeight = ref(0.8)

  // Generation settings
  const selectedPreset = ref('768-3:4')
  const steps = ref(20)
  const batchSize = ref(1)

  // Default quality prompts (applied automatically)
  const defaultQualityTags = 'masterpiece, best quality, absurdres, highres'
  const defaultNegativePrompt = 'bad anatomy, bad hands, missing fingers, extra fingers, extra digit, fewer digits, extra limbs, missing limbs, fused fingers, too many fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad proportions, gross proportions, malformed limbs, long neck, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, text, error'

  // All resolution presets (base size × aspect ratio)
  const resolutionPresets: ResolutionPreset[] = [
    // 512 base
    { id: '512-3:4', ratio: '3:4', width: 384, height: 512, label: '384×512' },
    { id: '512-1:1', ratio: '1:1', width: 512, height: 512, label: '512×512' },
    { id: '512-4:3', ratio: '4:3', width: 512, height: 384, label: '512×384' },
    // 768 base
    { id: '768-3:4', ratio: '3:4', width: 576, height: 768, label: '576×768' },
    { id: '768-1:1', ratio: '1:1', width: 768, height: 768, label: '768×768' },
    { id: '768-4:3', ratio: '4:3', width: 768, height: 576, label: '768×576' },
    // 1024 base
    { id: '1024-3:4', ratio: '3:4', width: 768, height: 1024, label: '768×1024' },
    { id: '1024-1:1', ratio: '1:1', width: 1024, height: 1024, label: '1024×1024' },
    { id: '1024-4:3', ratio: '4:3', width: 1024, height: 768, label: '1024×768' },
  ]

  const resolution = computed(() => {
    const preset = resolutionPresets.find(p => p.id === selectedPreset.value)
    return preset ? { width: preset.width, height: preset.height } : { width: 768, height: 1024 }
  })

  // Presets grouped by base size for 3-row display
  const presetGroups = [
    { label: 'low', presets: resolutionPresets.filter(p => p.id.startsWith('512-')) },
    { label: 'mid', presets: resolutionPresets.filter(p => p.id.startsWith('768-')) },
    { label: 'high', presets: resolutionPresets.filter(p => p.id.startsWith('1024-')) },
  ]

  // Computed: selected model's category
  const selectedModelCategory = computed(() => {
    const model = models.value.find(m => m.path === selectedModel.value || m.name === selectedModel.value)
    return model?.category || 'large'
  })

  // Computed: LoRAs filtered by selected model category
  const filteredLoras = computed(() => {
    return loras.value.filter(l => l.category === selectedModelCategory.value)
  })

  // Loading states
  const loadingModels = ref(false)
  const switchingModel = ref(false)

  // Actions
  async function loadModels() {
    loadingModels.value = true
    try {
      const [modelsRes, lorasRes, activeRes] = await Promise.all([
        api.getModels(),
        api.getLoras(),
        api.getActiveModel(),
      ])
      models.value = modelsRes.models
      loras.value = lorasRes.loras
      activeModel.value = activeRes.model
      if (activeRes.model) {
        selectedModel.value = activeRes.model
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    } finally {
      loadingModels.value = false
    }
  }

  async function switchModel(modelPath: string) {
    if (modelPath === activeModel.value) return

    switchingModel.value = true
    try {
      await api.switchModel(modelPath)
      activeModel.value = modelPath
      selectedModel.value = modelPath
    } catch (error) {
      console.error('Failed to switch model:', error)
      throw error
    } finally {
      switchingModel.value = false
    }
  }

  return {
    // Navigation
    currentView,

    // Models
    models,
    loras,
    filteredLoras,
    activeModel,
    selectedModel,
    selectedModelCategory,
    selectedLora,
    loraWeight,

    // Generation settings
    selectedPreset,
    resolutionPresets,
    presetGroups,
    steps,
    batchSize,
    resolution,
    defaultQualityTags,
    defaultNegativePrompt,

    // Loading states
    loadingModels,
    switchingModel,

    // Actions
    loadModels,
    switchModel,
  }
})
