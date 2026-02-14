<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as api from '@/api/client'
import type { CivitaiModel } from '@/types'
import ModelCard from './ModelCard.vue'

const STORAGE_KEY = 'civitai-search-state'

const query = ref('')
const modelType = ref('')
const baseModel = ref('')
const sortOrder = ref('Most Downloaded')
const loading = ref(false)
const results = ref<CivitaiModel[]>([])
const searched = ref(false)

// Restore search state from sessionStorage
onMounted(() => {
  const saved = sessionStorage.getItem(STORAGE_KEY)
  if (saved) {
    try {
      const state = JSON.parse(saved)
      query.value = state.query || ''
      modelType.value = state.modelType || ''
      baseModel.value = state.baseModel || ''
      sortOrder.value = state.sortOrder || 'Most Downloaded'
      results.value = state.results || []
      searched.value = state.searched || false
    } catch (e) {
      console.error('Failed to restore search state:', e)
    }
  }
})

function saveState() {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
    query: query.value,
    modelType: modelType.value,
    baseModel: baseModel.value,
    sortOrder: sortOrder.value,
    results: results.value,
    searched: searched.value,
  }))
}

const modelTypes = [
  { title: 'All Types', value: '' },
  { title: 'Checkpoint', value: 'Checkpoint' },
  { title: 'LoRA', value: 'LORA' },
  { title: 'LoCon', value: 'LoCon' },
  { title: 'Embedding', value: 'TextualInversion' },
  { title: 'VAE', value: 'VAE' },
  { title: 'ControlNet', value: 'Controlnet' },
]

const baseModels = [
  { title: 'All Base Models', value: '' },
  { title: 'SD 1.5', value: 'SD 1.5' },
  { title: 'SDXL', value: 'SDXL 1.0' },
  { title: 'Pony', value: 'Pony' },
  { title: 'Illustrious', value: 'Illustrious' },
  { title: 'Flux', value: 'Flux.1 D' },
]

const sortOptions = [
  { title: 'Most Downloaded', value: 'Most Downloaded' },
  { title: 'Highest Rated', value: 'Highest Rated' },
  { title: 'Newest', value: 'Newest' },
]

async function search() {
  loading.value = true
  searched.value = true
  try {
    const data = await api.searchCivitai({
      query: query.value || undefined,
      types: modelType.value || undefined,
      baseModels: baseModel.value || undefined,
      sort: sortOrder.value,
      limit: 24,
    })
    results.value = data.items || []
  } catch (e: any) {
    console.error('Search failed:', e)
    results.value = []
  } finally {
    loading.value = false
    saveState()
  }
}
</script>

<template>
  <v-container fluid class="fill-height pa-0 d-flex flex-column">
    <!-- Search header -->
    <v-sheet class="border-b pa-4">
      <div class="d-flex flex-wrap align-center justify-center ga-3 mx-auto" style="max-width: 1000px">
        <v-text-field
          v-model="query"
          placeholder="Search CivitAI models..."
          prepend-inner-icon="mdi-magnify"
          density="compact"
          hide-details
          clearable
          style="min-width: 250px; flex: 1"
          @keydown.enter="search"
        />

        <v-select
          v-model="modelType"
          :items="modelTypes"
          density="compact"
          hide-details
          style="min-width: 140px"
        />

        <v-select
          v-model="baseModel"
          :items="baseModels"
          density="compact"
          hide-details
          style="min-width: 150px"
        />

        <v-select
          v-model="sortOrder"
          :items="sortOptions"
          density="compact"
          hide-details
          style="min-width: 160px"
        />

        <v-btn color="primary" :loading="loading" @click="search">
          Search
        </v-btn>
      </div>
    </v-sheet>

    <!-- Results -->
    <v-container fluid class="flex-grow-1 overflow-y-auto pa-4">
      <div v-if="!searched" class="text-center text-grey mt-16">
        <v-icon size="64" color="grey-darken-1">mdi-magnify</v-icon>
        <p class="mt-4">Search for models on CivitAI</p>
      </div>

      <div v-else-if="loading" class="text-center mt-16">
        <v-progress-circular indeterminate color="primary" size="64" />
        <p class="mt-4 text-grey">Searching...</p>
      </div>

      <div v-else-if="results.length === 0" class="text-center text-grey mt-16">
        <v-icon size="64" color="grey-darken-1">mdi-magnify-close</v-icon>
        <p class="mt-4">No models found</p>
      </div>

      <v-row v-else>
        <v-col
          v-for="model in results"
          :key="model.id"
          cols="12"
          sm="6"
          md="4"
          lg="3"
        >
          <ModelCard :model="model" />
        </v-col>
      </v-row>
    </v-container>
  </v-container>
</template>

<style scoped>
.border-b {
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
</style>
