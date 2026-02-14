import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Model, LoRA, AspectRatio, BaseSize } from '@/types'
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
  const baseSize = ref<BaseSize>(768)
  const aspectRatio = ref<AspectRatio>('1:1')
  const steps = ref(20)
  const batchSize = ref(1)

  // Resolution computation
  const resolutions: Record<BaseSize, Record<AspectRatio, [number, number]>> = {
    512: { '1:1': [512, 512], '4:3': [512, 384], '3:4': [384, 512] },
    768: { '1:1': [768, 768], '4:3': [768, 576], '3:4': [576, 768] },
    1024: { '1:1': [1024, 1024], '4:3': [1024, 768], '3:4': [768, 1024] },
  }

  const resolution = computed(() => {
    const [width, height] = resolutions[baseSize.value][aspectRatio.value]
    return { width, height }
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
    activeModel,
    selectedModel,
    selectedLora,
    loraWeight,

    // Generation settings
    baseSize,
    aspectRatio,
    steps,
    batchSize,
    resolution,

    // Loading states
    loadingModels,
    switchingModel,

    // Actions
    loadModels,
    switchModel,
  }
})
