<script setup>
import { computed, onMounted, ref, watch } from 'vue';

const resultItems = [
  {
    title: 'Captured Scenes',
    kicker: 'Partial Polarization',
    input: './results/input/capture_zoom.jpg',
    output: './results/output/capture_zoom.bak.jpg',
    summary: 'Real World Scenes were captured under 2 linear polarizers and reconstructed using our method to recover spatially-varying reflectance and illumination.',
    shortSummary: '2LP & reconstruction',
  },
  {
    title: 'Reconstructed Envmaps',
    kicker: 'Lighting recovery',
    input: './results/input/envmap_gridv3.3.drawio.jpg',
    output: './results/output/envmap_gridv3.3.drawio.bak.jpg',
    summary: 'Recovered illumination preserves directional structure and overall color composition across scenes.',
    shortSummary: 'Comparison on envmaps',
  },
  {
    title: 'Relighting',
    kicker: 'Rendered response',
    input: './results/input/relightingv3.2.drawio.jpg',
    output: './results/output/relightingv3.2.drawio.bak.jpg',
    summary: 'Relighting results on David sculpture, compared with Gound Truth.',
    shortSummary: 'Comparison on relighting',
  },
  {
    title: 'Decomposition',
    kicker: 'More objects',
    input: './results/input/supp_resultsv1.jpg',
    output: './results/output/supp_resultsv1.bak.jpg',
    summary: 'More reconstruction results on different objects, along with the decomposed diffuse and specular components.',
    shortSummary: 'Material & Shape & Color',
  },
];

const selectedIndex = ref(0);
const isLoading = ref(true);
const displayedOutput = ref('');
let latestLoadToken = 0;

const selectedResult = computed(() => resultItems[selectedIndex.value] ?? resultItems[0]);

function handleChange(index) {
  if (selectedIndex.value === index) {
    return;
  }
  selectedIndex.value = index;
}

function preloadOutput(src) {
  const loadToken = ++latestLoadToken;
  isLoading.value = true;

  const image = new Image();
  image.onload = () => {
    if (loadToken !== latestLoadToken) {
      return;
    }
    displayedOutput.value = src;
    isLoading.value = false;
  };
  image.onerror = () => {
    if (loadToken !== latestLoadToken) {
      return;
    }
    displayedOutput.value = src;
    isLoading.value = false;
  };
  image.src = src;
}

watch(
  () => selectedResult.value.output,
  (nextOutput) => {
    preloadOutput(nextOutput);
  },
  { immediate: true }
);

onMounted(() => {
  preloadOutput(selectedResult.value.output);
});
</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <h1 class="section-title">Selected Results</h1>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <div class="result-selector-grid">
          <button
            v-for="(item, index) in resultItems"
            :key="item.title"
            type="button"
            class="result-selector-card"
            :class="{ 'result-selector-card-active': selectedIndex === index }"
            @click="handleChange(index)"
          >
            <div class="result-selector-thumb">
              <img :src="item.input" :alt="item.title" class="result-selector-image" loading="lazy">
            </div>
            <span class="result-selector-kicker">{{ item.kicker }}</span>
            <span class="result-selector-title">{{ item.title }}</span>
            <span class="result-selector-summary">{{ item.shortSummary }}</span>
          </button>
        </div>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="22" :md="20" :lg="18" :xl="18">
        <div class="result-spotlight">
          <div class="result-info-panel">
            <span class="result-info-kicker">{{ selectedResult.kicker }}</span>
            <h3 class="result-info-title">{{ selectedResult.title }}</h3>
            <p class="result-info-copy">{{ selectedResult.summary }}</p>

            <div class="reference-card">
              <!-- <span class="reference-label">Reference input</span> -->
              <div class="reference-thumb">
                <img
                  :src="selectedResult.input"
                  :alt="`${selectedResult.title} input`"
                  class="reference-image"
                  loading="lazy"
                >
              </div>
            </div>
          </div>

          <div class="result-output-panel">
            <div class="result-output-frame">
              <img
                v-if="displayedOutput"
                :src="displayedOutput"
                :alt="selectedResult.title"
                class="result-output-image"
                :class="{ 'result-output-image-loading': isLoading }"
                loading="lazy"
              >

              <div v-if="isLoading" class="result-output-loading">
                <el-skeleton-item variant="image" class="result-output-placeholder" />
              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.section-copy {
  margin: 12px 0 0;
}

.result-selector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  margin-top: 18px;
}

.result-selector-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
  min-width: 0;
  padding: 14px;
  border: 1px solid #d9dde5;
  border-radius: 22px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(245, 247, 250, 0.98));
  text-align: left;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.result-selector-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
}

.result-selector-card-active {
  border-color: #111827;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.14);
}

.result-selector-thumb {
  aspect-ratio: 5 / 4;
  border-radius: 16px;
  overflow: hidden;
  background:
    linear-gradient(135deg, rgba(235, 239, 244, 0.95), rgba(246, 248, 251, 0.95));
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-selector-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.result-selector-kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6b7280;
}

.result-selector-title {
  font-size: 17px;
  font-weight: 600;
  color: #111827;
}

.result-selector-summary {
  font-size: 13px;
  line-height: 1.55;
  color: #4b5563;
}

.result-spotlight {
  display: grid;
  grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
  gap: 22px;
  margin-top: 22px;
  align-items: stretch;
}

.result-info-panel,
.result-output-panel {
  min-width: 0;
}

.result-info-panel {
  padding: 22px;
  border: 1px solid #d8dde6;
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgba(234, 241, 255, 0.85), transparent 36%),
    linear-gradient(180deg, #ffffff, #f7f9fc);
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

.result-info-kicker {
  display: inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  background: #111827;
  color: #ffffff;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.result-info-title {
  margin: 16px 0 10px;
  font-size: 24px;
  line-height: 1.15;
}

.result-info-copy {
  margin: 0;
  color: #4b5563;
}

.reference-card {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid rgba(17, 24, 39, 0.08);
}

.reference-label {
  display: block;
  margin-bottom: 10px;
  font-size: 12px;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.reference-thumb {
  width: 100%;
  aspect-ratio: 4 / 3;
  border-radius: 18px;
  overflow: hidden;
  background: #eef2f7;
  display: flex;
  align-items: center;
  justify-content: center;
}

.reference-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.result-output-frame {
  position: relative;
  width: 100%;
  min-height: 420px;
  max-height: min(78vh, 920px);
  padding: 18px;
  border: 1px solid #d8dde6;
  border-radius: 24px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 249, 252, 0.98));
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.result-output-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  transition: opacity 0.2s ease;
}

.result-output-image-loading {
  opacity: 0.3;
}

.result-output-loading {
  position: absolute;
  inset: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.result-output-placeholder {
  width: 100%;
  height: 100%;
}

@media (max-width: 1024px) {
  .result-spotlight {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .result-selector-grid {
    grid-template-columns: 1fr;
  }

  .result-output-frame {
    min-height: 300px;
    padding: 14px;
  }

  .result-output-loading {
    inset: 14px;
  }

  .result-info-panel {
    padding: 18px;
  }
}
</style>
