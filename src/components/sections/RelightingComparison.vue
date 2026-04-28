<script setup>
import { computed, ref } from 'vue';

import VideoCompare from '../VideoCompare.vue';

const envmaps = [
  {
    key: 'museum',
    title: 'Museum',
    preview: './envmaps/museum.png',
    description: 'Warm indoor lighting with balanced highlights.',
    owl: {
      gir: './relighting_videos/gir/gir_owl_museum.mp4',
      phygap: './relighting_videos/phygap/phy_owl_museum.mp4',
    },
    squirrel: {
      gir: './relighting_videos/gir/gir_squirrel_museum.mp4',
      phygap: './relighting_videos/phygap/phy_squirrel_museum.mp4',
    },
  },
  {
    key: 'wharf',
    title: 'Wharf',
    preview: './envmaps/wharf.png',
    description: 'Outdoor waterfront reflections with stronger contrast.',
    owl: {
      gir: './relighting_videos/gir/gir_owl_wharf.mp4',
      phygap: './relighting_videos/phygap/phy_owl_wharf.mp4',
    },
    squirrel: {
      gir: './relighting_videos/gir/gir_squirrel_wharf.mp4',
      phygap: './relighting_videos/phygap/phy_squirrel_wharf.mp4',
    },
  },
  {
    key: 'dikhololo_night',
    title: 'Dikhololo Night',
    preview: './envmaps/dikhololo_night.png',
    description: 'Night illumination with sparse but sharp specular cues.',
    owl: {
      gir: './relighting_videos/gir/gir_owl_dik.mp4',
      phygap: './relighting_videos/phygap/phy_owl_dik.mp4',
    },
    squirrel: {
      gir: './relighting_videos/gir/gir_squirrel_dik.mp4',
      phygap: './relighting_videos/phygap/phy_squirrel_dik.mp4',
    },
  },
];

const selectedKey = ref(envmaps[0].key);

const selectedEnvmap = computed(() => {
  return envmaps.find((envmap) => envmap.key === selectedKey.value) ?? envmaps[0];
});
</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <h1 class="section-title">Relighting Comparisons</h1>
        <p class="section-copy">
          Choose an environment map to update both object comparisons. Each panel keeps the
          split-view interaction and a consistent presentation across the two results.
        </p>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <div class="selector-grid">
          <button
            v-for="envmap in envmaps"
            :key="envmap.key"
            type="button"
            class="selector-card"
            :class="{ 'selector-card-active': selectedKey === envmap.key }"
            @click="selectedKey = envmap.key"
          >
            <img :src="envmap.preview" :alt="envmap.title" class="selector-image">
            <span class="selector-title">{{ envmap.title }}</span>
            <span class="selector-description">{{ envmap.description }}</span>
          </button>
        </div>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <div class="selected-banner">
          <span class="selected-pill">Current envmap</span>
          <span class="selected-name">{{ selectedEnvmap.title }}</span>
        </div>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <div class="compare-grid">
          <VideoCompare
            title="Owl"
            :left-src="selectedEnvmap.owl.gir"
            :right-src="selectedEnvmap.owl.phygap"
          />
          <VideoCompare
            title="Squirrel"
            :left-src="selectedEnvmap.squirrel.gir"
            :right-src="selectedEnvmap.squirrel.phygap"
          />
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.section-copy {
  margin: 12px 0 0;
}

.selector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-top: 18px;
}

.selector-card {
  width: 100%;
  min-width: 0;
  border: 1px solid #dddddd;
  border-radius: 20px;
  padding: 12px;
  text-align: left;
  background: linear-gradient(180deg, #ffffff, #f7f8fa);
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.selector-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
}

.selector-card-active {
  border-color: #1f2937;
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.14);
}

.selector-image {
  width: 100%;
  aspect-ratio: 2 / 1;
  object-fit: cover;
  border-radius: 14px;
  display: block;
  margin-bottom: 10px;
  background: #e5e7eb;
}

.selector-title {
  display: block;
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 4px;
}

.selector-description {
  display: block;
  font-size: 13px;
  line-height: 1.5;
  color: #6b7280;
}

.selected-banner {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin: 18px 0 16px;
}

.selected-pill {
  border-radius: 999px;
  padding: 4px 10px;
  background: #111827;
  color: #ffffff;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.selected-name {
  font-size: 18px;
  font-weight: 600;
}

.compare-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
}

.compare-grid > * {
  min-width: 0;
}

@media (max-width: 768px) {
  .selector-grid {
    grid-template-columns: 1fr;
  }

  .compare-grid {
    grid-template-columns: 1fr;
  }
}
</style>
