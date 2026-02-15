<script setup lang="ts">
import { ref, computed } from 'vue'
import type { CivitaiModel } from '@/types'
import * as api from '@/api/client'
import { useAppStore } from '@/stores/app'

const props = defineProps<{
  model: CivitaiModel
}>()

const store = useAppStore()

const showDialog = ref(false)
const loadingDetails = ref(false)
const modelDetails = ref<CivitaiModel | null>(null)

const previewImage = computed(() => {
  const version = props.model.modelVersions?.[0]
  return version?.images?.[0]?.url || ''
})

const baseModel = computed(() => {
  return props.model.modelVersions?.[0]?.baseModel || ''
})

const trainedWords = computed(() => {
  return modelDetails.value?.modelVersions?.[0]?.trainedWords || []
})

function formatNumber(n: number): string {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K'
  return String(n)
}

async function openDetails() {
  showDialog.value = true
  loadingDetails.value = true
  try {
    modelDetails.value = await api.getCivitaiModel(props.model.id)
  } catch (e) {
    console.error('Failed to load model details:', e)
  } finally {
    loadingDetails.value = false
  }
}

async function downloadVersion(versionId: number) {
  if (store.isDownloading(versionId)) return // Already downloading
  if (!confirm(`Download "${props.model.name}" to the server?`)) return

  const downloadId = await store.startDownload(versionId)
  if (!downloadId) {
    alert('Failed to start download')
  }
}
</script>

<template>
  <v-card class="model-card" @click="openDetails">
    <v-img
      :src="previewImage"
      height="180"
      cover
      class="bg-grey-darken-3"
    >
      <template #placeholder>
        <div class="d-flex align-center justify-center fill-height">
          <v-icon size="48" color="grey">mdi-image</v-icon>
        </div>
      </template>
    </v-img>

    <v-card-text class="pb-2">
      <h4 class="text-body-1 font-weight-medium mb-2 text-truncate-2">
        {{ model.name }}
      </h4>

      <div class="d-flex flex-wrap ga-1 mb-2">
        <v-chip size="x-small" color="primary" variant="flat">
          {{ model.type }}
        </v-chip>
        <v-chip v-if="baseModel" size="x-small" variant="outlined">
          {{ baseModel }}
        </v-chip>
        <v-chip v-if="model.nsfw" size="x-small" color="error" variant="flat">
          NSFW
        </v-chip>
      </div>

      <div class="d-flex ga-3 text-caption text-grey">
        <span class="d-flex align-center ga-1">
          <v-icon size="14">mdi-download</v-icon>
          {{ formatNumber(model.stats?.downloadCount || 0) }}
        </span>
        <span class="d-flex align-center ga-1">
          <v-icon size="14">mdi-thumb-up</v-icon>
          {{ formatNumber(model.stats?.thumbsUpCount || 0) }}
        </span>
      </div>
    </v-card-text>

    <v-divider />

    <v-card-actions class="px-3">
      <span class="text-caption text-grey">
        by {{ model.creator?.username || 'Unknown' }}
      </span>
      <v-spacer />
      <!-- Download progress or button -->
      <template v-if="store.isDownloading(model.modelVersions?.[0]?.id || 0)">
        <div class="download-progress">
          <v-progress-linear
            :model-value="store.getDownloadProgress(model.modelVersions?.[0]?.id || 0)?.progress || 0"
            :indeterminate="store.getDownloadProgress(model.modelVersions?.[0]?.id || 0)?.status === 'queued'"
            color="primary"
            height="6"
            rounded
          />
          <span class="text-caption text-grey">
            {{ store.getDownloadProgress(model.modelVersions?.[0]?.id || 0)?.progress?.toFixed(0) || 0 }}%
          </span>
        </div>
      </template>
      <v-btn
        v-else
        size="small"
        color="primary"
        variant="flat"
        @click.stop="downloadVersion(model.modelVersions?.[0]?.id || 0)"
      >
        Download
      </v-btn>
    </v-card-actions>
  </v-card>

  <!-- Detail Dialog -->
  <v-dialog v-model="showDialog" max-width="600">
    <v-card>
      <v-card-title class="d-flex align-center">
        {{ model.name }}
        <v-spacer />
        <v-btn icon="mdi-close" variant="text" @click="showDialog = false" />
      </v-card-title>

      <v-card-text>
        <div v-if="loadingDetails" class="text-center py-8">
          <v-progress-circular indeterminate color="primary" />
        </div>

        <template v-else-if="modelDetails">
          <!-- Preview images -->
          <div v-if="modelDetails.modelVersions?.[0]?.images?.length" class="mb-4">
            <div class="d-flex ga-2 overflow-x-auto pb-2">
              <v-img
                v-for="(img, idx) in modelDetails.modelVersions[0].images.slice(0, 6)"
                :key="idx"
                :src="img.url"
                width="120"
                height="120"
                cover
                class="rounded flex-shrink-0"
              />
            </div>
          </div>

          <!-- Info -->
          <v-list density="compact">
            <v-list-item>
              <template #prepend><span class="text-grey mr-4">Type</span></template>
              {{ modelDetails.type }}
            </v-list-item>
            <v-list-item>
              <template #prepend><span class="text-grey mr-4">Creator</span></template>
              {{ modelDetails.creator?.username || 'Unknown' }}
            </v-list-item>
            <v-list-item>
              <template #prepend><span class="text-grey mr-4">Downloads</span></template>
              {{ formatNumber(modelDetails.stats?.downloadCount || 0) }}
            </v-list-item>
            <v-list-item>
              <template #prepend><span class="text-grey mr-4">Rating</span></template>
              {{ formatNumber(modelDetails.stats?.thumbsUpCount || 0) }} likes
            </v-list-item>
          </v-list>

          <!-- Trigger words -->
          <div v-if="trainedWords.length" class="mt-4">
            <h4 class="text-caption text-grey mb-2">TRIGGER WORDS</h4>
            <v-chip
              v-for="word in trainedWords"
              :key="word"
              size="small"
              class="mr-1 mb-1"
            >
              {{ word }}
            </v-chip>
          </div>

          <!-- Versions -->
          <div v-if="modelDetails.modelVersions?.length" class="mt-4">
            <h4 class="text-caption text-grey mb-2">VERSIONS</h4>
            <v-list density="compact" class="bg-transparent">
              <v-list-item
                v-for="version in modelDetails.modelVersions.slice(0, 5)"
                :key="version.id"
                class="px-0"
              >
                <v-list-item-title>{{ version.name }}</v-list-item-title>
                <v-list-item-subtitle>{{ version.baseModel }}</v-list-item-subtitle>
                <template #append>
                  <template v-if="store.isDownloading(version.id)">
                    <div class="download-progress-dialog">
                      <v-progress-linear
                        :model-value="store.getDownloadProgress(version.id)?.progress || 0"
                        :indeterminate="store.getDownloadProgress(version.id)?.status === 'queued'"
                        color="primary"
                        height="6"
                        rounded
                      />
                      <span class="text-caption text-grey">
                        {{ store.getDownloadProgress(version.id)?.progress?.toFixed(0) || 0 }}%
                      </span>
                    </div>
                  </template>
                  <v-btn
                    v-else
                    size="small"
                    color="primary"
                    variant="flat"
                    @click="downloadVersion(version.id)"
                  >
                    Download
                  </v-btn>
                </template>
              </v-list-item>
            </v-list>
          </div>
        </template>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.model-card {
  cursor: pointer;
  transition: transform 0.2s, border-color 0.2s;
}
.model-card:hover {
  transform: translateY(-2px);
  border-color: rgb(var(--v-theme-primary));
}
.text-truncate-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.download-progress {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 120px;
}
.download-progress-dialog {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100px;
}
</style>
