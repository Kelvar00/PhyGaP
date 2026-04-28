<script lang="ts" setup>
import { ElIcon } from 'element-plus'
import { Document, Files, Film } from '@element-plus/icons-vue'

const logo = ''

const title = 'PhyGaP: Physically-Grounded Gaussians with Polarization Cues'
const titleColor = '#000000'

const conference = 'IEEE/CVF CVPR 2026'
const conferenceHighlight = 'Oral'
const conferenceColor = '#303132'

const titleSupp = ''
const titleSuppColor = '#42B883'

const btnColor = '#444444'

const authors = [
  {
    name: 'Jiale Wu',
    homepage: 'https://kelvar00.github.io/',
    addressFlag: '1,2',
  },
  {
    name: 'Xiaoyang Bai',
    homepage: 'https://andrewbxy.github.io/',
    addressFlag: '2,\u2709',
  },
  {
    name: 'Zongqi He',
    homepage: 'https://wuyou012.github.io/',
    addressFlag: '2',
  },
  {
    name: 'Weiwei Xu',
    homepage: 'http://www.cad.zju.edu.cn/home/weiweixu/weiweixu_en.htm',
    addressFlag: '1',
  },
  {
    name: 'Yifan "Evan" Peng',
    homepage: 'https://www.eee.hku.hk/~evanpeng/',
    addressFlag: '2,\u2709',
  },
]

const addresses = [
  {
    addressFlag: '1',
    name: 'Zhejiang University',
    icon: './icon/zju.png',
    homepage: 'http://www.cad.zju.edu.cn/',
  },
  {
    addressFlag: '2',
    name: 'The University of Hong Kong',
    icon: './icon/hku.png',
    homepage: 'https://www.hku.hk/',
  },
]

const conAndCorrespondingAuthor = '*: Corresponding Author.'

const buttons = [
  {
    disabled: false,
    name: 'Paper',
    link: 'https://arxiv.org/abs/2603.14001',
    component: Document,
  },
  {
    disabled: false,
    name: 'Code',
    link: 'https://github.com/Kelvar00/PhyGaP/blob/main/README.md',
    component: Files,
  },
  {
    disabled: true,
    name: 'Video (TBD)',
    link: 'https://www.bilibili.com',
    component: Film,
  },
]
</script>

<template>
  <div>
    <el-row v-if="logo" justify="center">
      <el-image :src="logo" class="logo" fit="cover" />
    </el-row>

    <el-row justify="center">
      <el-col :xs="23" :sm="22" :md="20" :lg="20" :xl="18">
        <h1 class="paper-title">
          <span v-if="title" :style="{ color: titleColor }">{{ title }}</span>
          <span v-if="titleSupp" :style="{ color: titleSuppColor }">{{ titleSupp }}</span>
        </h1>
      </el-col>
    </el-row>

    

    <el-row justify="center" class="author-row">
      <a
        v-for="author in authors"
        :key="author.name"
        :href="author.homepage"
      >
        <el-button class="title-button" type="primary" text>
          <span class="author">
            {{ author.name }}<sup v-if="author.addressFlag" class="name-sup">{{ author.addressFlag }}</sup>
          </span>
        </el-button>
      </a>
    </el-row>

    <el-row justify="center" class="address-row">
      <a
        v-for="address in addresses"
        :key="address.name"
        :href="address.homepage"
      >
        <el-button class="title-button" type="primary" text>
          <el-avatar v-if="address.icon" :size="36" :src="address.icon" class="school-avatar" />
          <span class="address">
            <sup v-if="address.addressFlag" class="address-sup">{{ address.addressFlag }}</sup>{{ address.name }}
          </span>
        </el-button>
      </a>
    </el-row>

    <el-row justify="center">
      <el-col :xs="22" :sm="18" :md="14" :lg="12" :xl="10">
        <div class="conference-title">
          <span class="conference-line" :style="{ color: conferenceColor }">
            <span class="conference-emoji">&#127881;</span>
            <span class="conference-prefix">{{ conference }}</span>
            <strong class="conference-highlight">{{ conferenceHighlight }}</strong>
            <span class="conference-emoji">&#127881;</span>
          </span>
        </div>
      </el-col>
    </el-row>

    <!-- <el-row justify="center" class="con-cor">
      {{ conAndCorrespondingAuthor }}
    </el-row> -->

    <el-row justify="center" class="button-row">
      <el-col :xs="23" :sm="22" :md="20" :lg="20" :xl="18">
        <el-row justify="center">
          <a
            v-for="button in buttons"
            :key="button.name"
            :href="button.link"
          >
            <el-button class="guidance-button" size="default" :color="btnColor" :disabled="button.disabled" round>
              <el-icon :size="18">
                <component :is="button.component" />
              </el-icon>
              <span class="btn-text">{{ button.name }}</span>
            </el-button>
          </a>
        </el-row>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.paper-title {
  font-family: "IBM Plex Sans", "Manrope", "Segoe UI", "Arial", sans-serif;
  letter-spacing: 1.2px;
  font-size: 42px;
  line-height: 1.14;
  margin-top: 12px;
  margin-bottom: 18px;
  text-align: center;
}

.conference-title {
  font-family: "IBM Plex Sans", "Manrope", "Segoe UI", "Arial", sans-serif;
  display: flex;
  justify-content: center;
  margin-top: 12px;
  margin-bottom: 12px;
  text-align: center;
}

.conference-line {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  line-height: 1.15;
  flex-wrap: wrap;
}

.conference-prefix {
  font-size: 22px;
  letter-spacing: 0.02em;
  font-weight: 600;
}

.conference-highlight {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.conference-emoji {
  font-size: 20px;
  line-height: 1;
}

.author-row,
.address-row {
  row-gap: 0;
}

.title-button {
  margin: 3px 2px;
}

.title-button:hover {
  margin: 3px 5px;
}

.guidance-button {
  margin: 8px 5px;
  box-shadow: #d8d8d8 1px 1px 1px 1px;
  border-radius: 10px !important;
}

.author {
  font-family: "IBM Plex Sans", "Manrope", "Segoe UI", "Arial", sans-serif;
  font-size: 18px;
  margin-left: 3px;
}

.name-sup {
  color: #606266;
  margin-left: 3px;
}

.address {
  font-family: "IBM Plex Sans", "Manrope", "Segoe UI", "Arial", sans-serif;
  font-size: 18px;
}

.address-sup {
  color: #606266;
  margin-right: 1px;
}

.school-avatar {
  margin-right: 6px;
  box-shadow: #b7b7b7 0 0 3px 1px;
}

.con-cor {
  font-family: "IBM Plex Sans", "Manrope", "Segoe UI", "Arial", sans-serif;
  font-size: 13px;
  margin: 8px 0 14px;
  text-align: center;
}

.button-row {
  margin-bottom: 20px;
}

.btn-text {
  font-size: 18px;
  color: #ffffff;
}

.logo {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  box-shadow: #ced3dc 0 0 3px 2px;
  margin-top: 40px;
}

a:-webkit-any-link {
  text-decoration: none;
}

a:hover {
  color: inherit;
  border-bottom: none;
}

a {
  text-decoration: none;
  color: inherit;
}

@media (max-width: 768px) {
  .paper-title {
    font-size: 30px;
    line-height: 1.18;
    letter-spacing: 0.4px;
    margin-top: 8px;
    margin-bottom: 14px;
  }

  .conference-title {
    margin-top: 2px;
    margin-bottom: 2px;
  }

  .conference-line {
    gap: 6px;
  }

  .conference-prefix {
    font-size: 18px;
    letter-spacing: 0.04em;
  }

  .conference-highlight {
    font-size: 18px;
  }

  .conference-emoji {
    font-size: 16px;
  }

  .title-button {
    margin: 2px 1px;
  }

  .title-button:hover {
    margin: 2px 1px;
  }

  .author {
    font-size: 14px;
  }

  .address {
    font-size: 14px;
  }

  .school-avatar {
    width: 28px !important;
    height: 28px !important;
    font-size: 12px !important;
    margin-right: 5px;
  }

  .btn-text {
    font-size: 15px;
  }

  .con-cor {
    font-size: 11px;
    margin: 6px 0 12px;
    padding: 0 16px;
  }
}
</style>
