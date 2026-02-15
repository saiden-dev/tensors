<script setup lang="ts">
import { onMounted, onUnmounted, watch } from 'vue'
import { useAppStore } from '@/stores/app'

const store = useAppStore()

// Start polling when panel opens or when there are active downloads
watch(() => store.showDownloadsPanel, (show) => {
  if (show) {
    store.startDownloadPolling()
  }
})

// Also poll on mount if there might be active downloads
onMounted(() => {
  store.pollDownloads()
})

onUnmounted(() => {
  // Don't stop polling - let it continue for background tracking
})

// Auto-stop polling when no active downloads for a while
watch(() => store.hasActiveDownloads, (hasActive) => {
  if (!hasActive && !store.showDownloadsPanel) {
    // Give a grace period before stopping
    setTimeout(() => {
      if (!store.hasActiveDownloads && !store.showDownloadsPanel) {
        store.stopDownloadPolling()
      }
    }, 5000)
  }
})

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB'
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return bytes + ' B'
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'completed': return 'success'
    case 'failed': return 'error'
    case 'downloading': return 'primary'
    default: return 'grey'
  }
}

function getStatusIcon(status: string): string {
  switch (status) {
    case 'completed': return 'mdi-check-circle'
    case 'failed': return 'mdi-alert-circle'
    case 'downloading': return 'mdi-download'
    default: return 'mdi-clock-outline'
  }
}
</script>

<template>
  <v-navigation-drawer
    v-model="store.showDownloadsPanel"
    location="right"
    temporary
    width="360"
  >
    <v-toolbar density="compact" color="surface">
      <v-toolbar-title class="text-body-1">Downloads</v-toolbar-title>
      <v-spacer />
      <v-btn icon="mdi-close" variant="text" size="small" @click="store.showDownloadsPanel = false" />
    </v-toolbar>

    <v-list v-if="store.downloads.length > 0" density="compact">
      <v-list-item
        v-for="dl in store.downloads"
        :key="dl.id"
        :class="{ 'bg-surface-light': dl.status === 'downloading' }"
      >
        <template #prepend>
          <v-icon :color="getStatusColor(dl.status)" class="mr-3">
            {{ getStatusIcon(dl.status) }}
          </v-icon>
        </template>

        <v-list-item-title class="text-body-2 font-weight-medium">
          {{ dl.model_name || dl.filename }}
        </v-list-item-title>
        <v-list-item-subtitle class="text-caption">
          {{ dl.version_name || '' }}
        </v-list-item-subtitle>

        <!-- Progress bar for active downloads -->
        <div v-if="dl.status === 'downloading'" class="mt-2">
          <v-progress-linear
            :model-value="dl.progress || 0"
            color="primary"
            height="6"
            rounded
          />
          <div class="d-flex justify-space-between text-caption text-grey mt-1">
            <span>{{ dl.downloaded_str || formatBytes(dl.downloaded || 0) }} / {{ dl.total_str || formatBytes(dl.total || 0) }}</span>
            <span class="text-primary">{{ dl.speed_str || '' }}</span>
          </div>
        </div>

        <!-- Queued status -->
        <div v-else-if="dl.status === 'queued'" class="mt-2">
          <v-progress-linear indeterminate color="grey" height="4" rounded />
          <div class="text-caption text-grey mt-1">Queued...</div>
        </div>

        <!-- Completed status -->
        <div v-else-if="dl.status === 'completed'" class="text-caption text-success mt-1">
          Downloaded to {{ dl.filename }}
        </div>

        <!-- Failed status -->
        <div v-else-if="dl.status === 'failed'" class="text-caption text-error mt-1">
          {{ dl.error || 'Download failed' }}
        </div>
      </v-list-item>
    </v-list>

    <div v-else class="text-center text-grey pa-8">
      <v-icon size="48" color="grey-darken-1">mdi-download-off</v-icon>
      <p class="mt-4">No downloads</p>
    </div>
  </v-navigation-drawer>
</template>
