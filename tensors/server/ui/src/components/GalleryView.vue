<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as api from '@/api/client'
import type { GalleryImage } from '@/types'

const images = ref<GalleryImage[]>([])
const loading = ref(true)
const selectedImage = ref<string | null>(null)
const showLightbox = ref(false)

async function loadImages() {
  loading.value = true
  try {
    const data = await api.getImages(100)
    images.value = data.images || []
  } catch (e) {
    console.error('Failed to load gallery:', e)
  } finally {
    loading.value = false
  }
}

function openLightbox(id: string) {
  selectedImage.value = id
  showLightbox.value = true
}

function closeLightbox() {
  showLightbox.value = false
  selectedImage.value = null
}

function navigate(direction: 'prev' | 'next') {
  if (!selectedImage.value) return
  const currentIndex = images.value.findIndex(img => img.id === selectedImage.value)
  if (currentIndex === -1) return

  let newIndex: number
  if (direction === 'prev') {
    newIndex = currentIndex > 0 ? currentIndex - 1 : images.value.length - 1
  } else {
    newIndex = currentIndex < images.value.length - 1 ? currentIndex + 1 : 0
  }
  selectedImage.value = images.value[newIndex]?.id ?? null
}

async function deleteImage(id: string) {
  if (!confirm('Delete this image?')) return

  try {
    await api.deleteImage(id)
    images.value = images.value.filter(img => img.id !== id)
    if (selectedImage.value === id) {
      closeLightbox()
    }
  } catch (e: any) {
    alert('Failed to delete: ' + e.message)
  }
}

onMounted(loadImages)
</script>

<template>
  <v-container fluid class="fill-height pa-0">
    <div v-if="loading" class="d-flex align-center justify-center fill-height w-100">
      <v-progress-circular indeterminate color="primary" size="64" />
    </div>

    <div v-else-if="images.length === 0" class="d-flex flex-column align-center justify-center fill-height w-100 text-grey">
      <v-icon size="64" color="grey-darken-1">mdi-image-multiple</v-icon>
      <p class="mt-4">No images yet</p>
    </div>

    <v-container v-else fluid class="pa-4 overflow-y-auto">
      <v-row>
        <v-col
          v-for="img in images"
          :key="img.id"
          cols="6"
          sm="4"
          md="3"
          lg="2"
        >
          <v-card
            class="gallery-card"
            @click="openLightbox(img.id)"
          >
            <v-img
              :src="api.getImageUrl(img.id)"
              aspect-ratio="1"
              cover
              class="bg-grey-darken-3"
            >
              <template #placeholder>
                <div class="d-flex align-center justify-center fill-height">
                  <v-progress-circular indeterminate color="primary" size="24" />
                </div>
              </template>
            </v-img>
            <div class="delete-overlay">
              <v-btn
                icon="mdi-delete"
                size="small"
                color="error"
                variant="flat"
                @click.stop="deleteImage(img.id)"
              />
            </div>
          </v-card>
        </v-col>
      </v-row>
    </v-container>

    <!-- Lightbox -->
    <v-dialog
      v-model="showLightbox"
      fullscreen
      transition="fade-transition"
    >
      <v-card class="bg-black d-flex flex-column">
        <v-toolbar color="transparent" density="compact">
          <v-spacer />
          <v-btn
            icon="mdi-delete"
            variant="text"
            @click="selectedImage && deleteImage(selectedImage)"
          />
          <v-btn
            icon="mdi-close"
            variant="text"
            @click="closeLightbox"
          />
        </v-toolbar>

        <div class="flex-grow-1 d-flex align-center justify-center position-relative">
          <v-btn
            icon="mdi-chevron-left"
            variant="text"
            size="x-large"
            class="position-absolute"
            style="left: 16px"
            @click="navigate('prev')"
          />

          <v-img
            v-if="selectedImage"
            :src="api.getImageUrl(selectedImage)"
            max-height="90vh"
            max-width="90vw"
            contain
          />

          <v-btn
            icon="mdi-chevron-right"
            variant="text"
            size="x-large"
            class="position-absolute"
            style="right: 16px"
            @click="navigate('next')"
          />
        </div>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<style scoped>
.gallery-card {
  cursor: pointer;
  transition: transform 0.2s, border-color 0.2s;
  border: 1px solid transparent;
  position: relative;
}
.gallery-card:hover {
  transform: scale(1.03);
  border-color: rgb(var(--v-theme-primary));
}
.delete-overlay {
  position: absolute;
  top: 4px;
  right: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}
.gallery-card:hover .delete-overlay {
  opacity: 1;
}
</style>
