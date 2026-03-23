<script setup>
import page from '../mds/pipeline.mdx'
import pipelineImg from '/pipeline/pipeline_fig_v3.png'
</script>

<template>
  <div>
    <el-divider />
    <el-row justify="center">
      <el-col :xs="24" :sm="20" :md="18" :lg="14" :xl="14">
        <img :src="pipelineImg" alt="Pipeline" class="pipeline-img" />
        <page class= "page" />
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.pipeline-img {
  width: 100%;
  max-width: 1080px;
  display: block;
  margin: 0px auto 12px;
}

.page{
  width:96%;
  margin: 0px auto;
}
</style>