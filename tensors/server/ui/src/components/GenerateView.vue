<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
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
  ...store.filteredLoras.map(l => ({ title: l.name, value: l.path }))
])


async function handleModelChange(model: string) {
  if (model && model !== store.activeModel) {
    try {
      await store.switchModel(model)
      // Reset LoRA if it's not compatible with the new model
      const loraStillValid = store.filteredLoras.some(l => l.path === store.selectedLora)
      if (!loraStillValid) {
        store.selectedLora = ''
      }
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

  // Build final prompt with quality tags
  const finalPrompt = `${store.defaultQualityTags}, ${currentPrompt}`

  // Get LoRA config if selected (sd-server expects LoRA as separate param with filename, not in prompt)
  const selectedLoraModel = store.selectedLora ? store.loras.find(l => l.path === store.selectedLora) : null
  const loraConfig = selectedLoraModel ? { path: selectedLoraModel.filename, multiplier: store.loraWeight } : undefined

  const { width, height } = store.resolution
  const paramsStr = `${width}×${height}, ${store.steps} steps${store.batchSize > 1 ? `, batch ${store.batchSize}` : ''}${selectedLoraModel ? `, +${selectedLoraModel.name}` : ''}`

  const message = reactive<ChatMessage>({
    prompt: currentPrompt,
    params: paramsStr,
    images: [],
    loading: true,
  })
  messages.value.push(message)

  try {
    for (let i = 0; i < store.batchSize; i++) {
      const result = await api.generate({
        prompt: finalPrompt,
        negative_prompt: store.defaultNegativePrompt,
        width,
        height,
        steps: store.steps,
        seed: -1,
        save_to_gallery: true,
        lora: loraConfig,
      })
      message.images.push(...result.images)
    }
  } catch (e: any) {
    console.error('Generate error:', e)
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
            style="width: 200px"
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
            style="width: 150px"
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

        <div class="resolution-grid rounded border pa-2">
          <div
            v-for="group in store.presetGroups"
            :key="group.label"
            class="d-flex align-center ga-2"
          >
            <span class="text-caption text-grey text-uppercase resolution-label">{{ group.label }}</span>
            <v-btn-toggle
              v-model="store.selectedPreset"
              mandatory
              density="compact"
              :disabled="generating"
              class="resolution-row"
            >
              <v-btn v-for="p in group.presets" :key="p.id" :value="p.id" size="small" class="resolution-btn">
                {{ p.label }}
              </v-btn>
            </v-btn-toggle>
          </div>
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

.resolution-grid {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-color: rgba(255, 255, 255, 0.12) !important;
}

.resolution-label {
  width: 36px;
  text-align: left;
}

.resolution-row {
  display: grid !important;
  grid-template-columns: repeat(3, 100px);
  width: 300px;
}

.resolution-btn {
  width: 100px !important;
  min-width: 100px !important;
  max-width: 100px !important;
  justify-content: center;
}
</style>
