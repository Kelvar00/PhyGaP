<script setup>
import teaserImg from '/teaser/Polar_Fig0_v3.png'
</script>

<template>
  <div>
    <el-divider />
    <el-row justify="center">
      <el-col :xs="24" :sm="20" :md="18" :lg="14" :xl="14">
       <h1 class="section-title teaser-title">Overview</h1>
        <img :src="teaserImg" alt="Teaser" class="teaser-img" />
         
        <p class="teaser-text">
          We propose PhyGaP, a physically-grounded 3DGS method that (a) takes full or partial
          polarization information as input, (b) accurately reconstructs the shape and physical
          attributes of glossy object, (c) achieves decomposed rendering of object appearance (top),
          diffuse reflection (mid), and specular reflection (bottom), as well as (d) enables robust
          and realistic relighting with natural reflection. Results in this visualization are from
          our captured buddha scene.
        </p>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.teaser-img {
  width: 96%;
  max-width: 960px;
  display: block;
  margin: 0px auto 12px;
}

.teaser-text {
  width: 100%;
  margin: 0px auto;
}

.teaser-title{
    margin: 0;
    margin-bottom: 12px;
}
</style>
