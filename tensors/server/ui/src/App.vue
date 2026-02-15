<script setup lang="ts">
import { onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import GenerateView from '@/components/GenerateView.vue'
import SearchView from '@/components/SearchView.vue'
import GalleryView from '@/components/GalleryView.vue'
import DownloadsPanel from '@/components/DownloadsPanel.vue'

const store = useAppStore()

onMounted(() => {
  store.loadModels()
  store.pollDownloads() // Check for any active downloads on load
})
</script>

<template>
  <v-app>
    <v-navigation-drawer permanent rail>
      <v-list density="compact" nav>
        <v-list-item
          :active="store.currentView === 'generate'"
          @click="store.currentView = 'generate'"
          prepend-icon="mdi-auto-fix"
          title="Generate"
        />
        <v-list-item
          :active="store.currentView === 'search'"
          @click="store.currentView = 'search'"
          prepend-icon="mdi-magnify"
          title="Search"
        />
        <v-list-item
          :active="store.currentView === 'gallery'"
          @click="store.currentView = 'gallery'"
          prepend-icon="mdi-image-multiple"
          title="Gallery"
        />

        <v-divider class="my-2" />

        <v-list-item
          @click="store.showDownloadsPanel = true"
          title="Downloads"
        >
          <template #prepend>
            <v-badge
              v-if="store.hasActiveDownloads"
              :content="store.activeDownloads.length"
              color="primary"
              offset-x="-2"
              offset-y="-2"
            >
              <v-icon>mdi-download</v-icon>
            </v-badge>
            <v-icon v-else>mdi-download</v-icon>
          </template>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <DownloadsPanel />

    <v-main>
      <GenerateView v-if="store.currentView === 'generate'" />
      <SearchView v-else-if="store.currentView === 'search'" />
      <GalleryView v-else-if="store.currentView === 'gallery'" />
    </v-main>
  </v-app>
</template>

<style>
html, body {
  overflow: hidden;
}
</style>
