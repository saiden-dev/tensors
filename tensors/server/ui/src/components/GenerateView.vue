<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppStore } from '@/stores/app'
import * as api from '@/api/client'
import type { GeneratedImage } from '@/types'

const store = useAppStore()

const prompt = ref('')
const generating = ref(false)

interface ChatMessage {
  prompt: string
  params: string
  images: GeneratedImage[]
  error?: string
  loading: boolean
}

const messages = ref<ChatMessage[]>([])

const modelItems = computed(() =>
  store.models.map(m => ({ title: m.name, value: m.path }))
)

const loraItems = computed(() => [
  { title: 'None', value: '' },
  ...store.loras.map(l => ({ title: l.name, value: l.path }))
])

const baseSizes = [
  { title: '512', value: 512 },
  { title: '768', value: 768 },
  { title: '1024', value: 1024 },
] as const

const aspectRatios = [
  { title: '3:4', value: '3:4' as const },
  { title: '1:1', value: '1:1' as const },
  { title: '4:3', value: '4:3' as const },
]

async function handleModelChange(model: string) {
  if (model && model !== store.activeModel) {
    try {
      await store.switchModel(model)
    } catch (e) {
      console.error(e)
    }
  }
}

async function generate() {
  if (!prompt.value.trim() || generating.value) return

  const currentPrompt = prompt.value.trim()
  prompt.value = ''
  generating.value = true

  // Build final prompt with LoRA
  let finalPrompt = currentPrompt
  if (store.selectedLora) {
    const loraName = store.loras.find(l => l.path === store.selectedLora)?.name
    if (loraName) {
      finalPrompt = `<lora:${loraName}:${store.loraWeight}> ${currentPrompt}`
    }
  }

  const { width, height } = store.resolution
  const paramsStr = `${width}×${height}, ${store.steps} steps${store.batchSize > 1 ? `, batch ${store.batchSize}` : ''}${store.selectedLora ? ', +LoRA' : ''}`

  const message: ChatMessage = {
    prompt: currentPrompt,
    params: paramsStr,
    images: [],
    loading: true,
  }
  messages.value.push(message)

  try {
    for (let i = 0; i < store.batchSize; i++) {
      const result = await api.generate({
        prompt: finalPrompt,
        width,
        height,
        steps: store.steps,
        seed: -1,
        save_to_gallery: true,
      })
      message.images.push(...result.images)
    }
  } catch (e: any) {
    message.error = e.message || 'Generation failed'
  } finally {
    message.loading = false
    generating.value = false
  }
}
</script>

<template>
  <v-container fluid class="fill-height pa-0 d-flex flex-column">
    <!-- Chat area -->
    <v-container fluid class="flex-grow-1 overflow-y-auto pa-4">
      <div v-if="messages.length === 0" class="text-center text-grey mt-16">
        <v-icon size="64" color="grey-darken-1">mdi-auto-fix</v-icon>
        <p class="mt-4">Enter a prompt to generate images</p>
      </div>

      <div v-for="(msg, idx) in messages" :key="idx" class="mb-6">
        <v-chip color="primary" variant="tonal" class="mb-3">
          <span class="font-weight-medium">{{ msg.prompt }}</span>
          <span class="text-grey ml-2 text-caption">[{{ msg.params }}]</span>
        </v-chip>

        <div class="d-flex flex-wrap ga-3">
          <template v-if="msg.loading">
            <v-card
              v-for="i in store.batchSize - msg.images.length"
              :key="'loading-' + i"
              width="200"
              height="200"
              class="d-flex align-center justify-center"
            >
              <v-progress-circular indeterminate color="primary" />
            </v-card>
          </template>

          <v-card
            v-for="img in msg.images"
            :key="img.id"
            width="200"
            height="200"
            class="overflow-hidden"
          >
            <v-img
              :src="api.getImageUrl(img.id)"
              cover
              height="200"
            />
          </v-card>

          <v-alert v-if="msg.error" type="error" density="compact">
            {{ msg.error }}
          </v-alert>
        </div>
      </div>
    </v-container>

    <!-- Controls -->
    <v-sheet class="border-t px-4 py-3">
      <div class="d-flex flex-wrap align-center justify-center ga-4 mb-3">
        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">Model</span>
          <v-select
            v-model="store.selectedModel"
            :items="modelItems"
            :loading="store.switchingModel"
            :disabled="store.switchingModel || generating"
            density="compact"
            hide-details
            style="min-width: 180px"
            @update:model-value="handleModelChange"
          />
        </div>

        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">LoRA</span>
          <v-select
            v-model="store.selectedLora"
            :items="loraItems"
            :disabled="generating"
            density="compact"
            hide-details
            style="min-width: 150px"
          />
          <v-text-field
            v-model.number="store.loraWeight"
            type="number"
            min="0"
            max="2"
            step="0.1"
            density="compact"
            hide-details
            style="width: 70px"
            :disabled="!store.selectedLora || generating"
          />
        </div>

        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">Size</span>
          <v-btn-toggle v-model="store.baseSize" mandatory density="compact" :disabled="generating">
            <v-btn v-for="s in baseSizes" :key="s.value" :value="s.value" size="small">
              {{ s.title }}
            </v-btn>
          </v-btn-toggle>
        </div>

        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">Ratio</span>
          <v-btn-toggle v-model="store.aspectRatio" mandatory density="compact" :disabled="generating">
            <v-btn v-for="r in aspectRatios" :key="r.value" :value="r.value" size="small">
              {{ r.title }}
            </v-btn>
          </v-btn-toggle>
        </div>

        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">Steps</span>
          <v-text-field
            v-model.number="store.steps"
            type="number"
            min="1"
            max="50"
            density="compact"
            hide-details
            style="width: 70px"
            :disabled="generating"
          />
        </div>

        <div class="d-flex align-center ga-2">
          <span class="text-caption text-grey text-uppercase">Batch</span>
          <v-text-field
            v-model.number="store.batchSize"
            type="number"
            min="1"
            max="8"
            density="compact"
            hide-details
            style="width: 70px"
            :disabled="generating"
          />
        </div>
      </div>

      <!-- Prompt input -->
      <div class="d-flex ga-3 mx-auto" style="max-width: 800px">
        <v-text-field
          v-model="prompt"
          placeholder="Describe what you want to generate..."
          density="comfortable"
          hide-details
          :disabled="generating"
          @keydown.enter="generate"
        />
        <v-btn
          color="secondary"
          size="large"
          :loading="generating"
          :disabled="!prompt.trim()"
          @click="generate"
        >
          Generate
        </v-btn>
      </div>
    </v-sheet>
  </v-container>
</template>

<style scoped>
.border-t {
  border-top: 1px solid rgba(255, 255, 255, 0.12);
}
</style>
