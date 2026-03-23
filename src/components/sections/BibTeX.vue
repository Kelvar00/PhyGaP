<script>
export default {
  data() {
    return {
      bibtex: [
        "@article{wu2026phygap,",
        "    title   = {PhyGaP: Physically-Grounded Gaussians with Polarization Cues},",
        "    author  = {Wu, Jiale and Bai, Xiaoyang and He, Zongqi and Xu, Weiwei and Peng, Yifan},",
        "    journal = {arXiv preprint arXiv:2603.14001},",
        "    year    = {2026}",
        "  }",
      ],
    }
  },
  methods: {
    copyVal() {
      let oInput = document.createElement('textarea');
      let text = document.getElementById('bibtex').innerText;
      oInput.value = text;
      document.body.appendChild(oInput);
      oInput.select();
      document.execCommand('Copy');
      this.$message.success('Copy Successfully');
      oInput.remove();
    }
  }
}

</script>

<template>
  <div>
    <el-divider />
      
    <el-row justify="center">
      <el-col :xs="24" :sm="20" :md="18" :lg="14" :xl="14">
        <el-row justify="space-between" align="middle" class="bibtex-header">
          <h1 class="section-title bibtex-title">BibTeX</h1>
          <el-button size="small" type="primary" plain class="bibtex-copy" @click="copyVal()">Copy BibTeX</el-button>
        </el-row>

        <el-col class='bibtex' :xs="24" :sm="24" :md="24" :lg="24" :xl="24">
          <el-row>
            <el-scrollbar style="margin: 0px 20px 5px 20px;">
              <pre id="bibtex"><code v-for="b in bibtex">{{ b }}<br/></code></pre>
            </el-scrollbar>
          </el-row>
        </el-col>
      </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.scrollbar-flex-content {
  display: flex;
}

.bibtex-header {
  margin-top: 10px;
}

.bibtex-title {
  margin: 0;
}

.bibtex-copy {
  margin-left: auto;
}

/* 卡片属性 */
.bibtex {
  margin: 20px 0px;
  padding-top: 5px;
  box-shadow: var(--el-box-shadow-light); 
  border-radius: 10px;
}

/* 卡片悬浮 */
.bibtex:hover {
  box-shadow: var(--el-box-shadow); 
}

/* 卡片点击 */
.bibtex:active{
  box-shadow: var(--el-box-shadow-lighter); 
}

pre {
  border: none;
  border-radius: 0px;
  padding: 10px;
  background: none;
}

pre code {
  font-size: 12px;
  background: #ffffff;
}



</style>