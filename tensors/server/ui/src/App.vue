<script setup lang="ts">
import { onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import GenerateView from '@/components/GenerateView.vue'
import SearchView from '@/components/SearchView.vue'
import GalleryView from '@/components/GalleryView.vue'

const store = useAppStore()

onMounted(() => {
  store.loadModels()
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
      </v-list>
    </v-navigation-drawer>

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
