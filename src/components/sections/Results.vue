<script setup>
import { ref, onMounted } from 'vue';

const imageSeletorPaths = [
  "./results/input/capture_zoom.jpg",
  "./results/input/envmap_gridv3.3.drawio.jpg",
  "./results/input/relightingv3.2.drawio.jpg",
  "./results/input/supp_resultsv1.jpg"
];

const outputImagesPaths = [
  "./results/output/capture_zoom.bak.jpg",
  "./results/output/envmap_gridv3.3.drawio.bak.jpg",
  "./results/output/relightingv3.2.drawio.bak.jpg",
  "./results/output/supp_resultsv1.bak.jpg"
];

const imageSelectorDescriptions = [
  "Captured Scenes",
  "Reconstructed Envmaps",
  "Relighting",
  "Decomposition"
];

let outputImagePath = ref("");
let indexSelected = ref(0);
let isLoading = ref(true);

onMounted(() => {
    handleChange(0);
});

const handleChange = (value) => {
  indexSelected.value = value;
  isLoading.value = true;
  outputImagePath.value = outputImagesPaths[value];
};

const handleOutputLoaded = () => {
  isLoading.value = false;
};

</script>

<template>
  <div>
    <el-divider />

    <el-row justify="center">
      <el-col :xs="24" :sm="20" :md="18" :lg="14" :xl="14">
        <h1 class="section-title">Selected Results</h1>
      </el-col>
    </el-row>

    <el-row justify="center">
      <el-col :xs="24" :sm="20" :md="16" :lg="12" :xl="12">
        <el-row justify="space-evenly" style="margin-top: 20px;">
          <el-col :span="4" v-for="(imageSeletorPath, index) in imageSeletorPaths" :key="index">
            <el-image 
              class="image" 
              :src="imageSeletorPath" style="aspect-ratio: 1;" 
              fit="scale-down" 
              :lazy="true"
              @click="handleChange(index)"
              :class="{ 'selected-image': indexSelected === index, 'unselected-image': indexSelected !== index }"
            />
            <div class="image-desc">{{ imageSelectorDescriptions[index] }}</div>
          </el-col>
        </el-row>
        <el-row justify="center" style="margin-top: 10px;">
          <el-col :span="24" class="output-image-col">
            <el-skeleton
              style="width: 100%"
              :loading="isLoading"
              animated
              :throttle="1000">
              <template #template>
                <div class="output-image-frame">
                  <el-skeleton-item variant="image" class="output-image-skeleton" />
                </div>
              </template>
              <template #default>
                <div class="output-image-frame">
                  <img :src="outputImagePath" class="output-image" loading="lazy" @load="handleOutputLoaded">
                </div>
              </template>
            </el-skeleton>
          </el-col>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>

.image:hover{
  transition: none;
  box-shadow: 0px 0px 6px 0px #aaaaaa;
}

.selected-image{
  transition: 0.5s ease;
  box-shadow: 0px 0px 6px 0px #aaaaaa;
}


/* 未选中图像的样式，颜色变灰 */
.unselected-image {
  transition: 0.5s ease;
  opacity: 0.4;
}

.image-desc {
  margin-top: 6px;
  font-size: 12px;
  text-align: center;
  color: #666666;
}

.output-image-col {
  width: 100%;
  display: flex;
  justify-content: center;
}

.output-image {
  height: 100%;
  width: auto;
  max-width: none;
  object-fit: contain;
  display: block;
  margin: 0;
}

.output-image-skeleton {
  width: 100%;
  height: 100%;
}

.output-image-frame {
  width: 100%;
  max-width: 680px;
  height: 680px;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  overflow-x: auto;
  overflow-y: hidden;
}

</style>