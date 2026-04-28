<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';

const props = defineProps({
  title: {
    type: String,
    default: '',
  },
  leftSrc: {
    type: String,
    required: true,
  },
  rightSrc: {
    type: String,
    required: true,
  },
  leftLabel: {
    type: String,
    default: 'GIR',
  },
  rightLabel: {
    type: String,
    default: 'PhyGaP',
  },
});

const stageRef = ref(null);
const leftVideoRef = ref(null);
const rightVideoRef = ref(null);
const split = ref(0.5);
const currentTime = ref(0);
const duration = ref(0);
const isPlaying = ref(false);
const isDragging = ref(false);

const leftClipStyle = computed(() => ({
  clipPath: `inset(0 ${100 - split.value * 100}% 0 0)`,
}));

const dividerStyle = computed(() => ({
  left: `${split.value * 100}%`,
}));

const progress = computed(() => {
  if (!duration.value) {
    return 0;
  }
  return currentTime.value / duration.value;
});

function syncSecondaryVideo() {
  const leftVideo = leftVideoRef.value;
  const rightVideo = rightVideoRef.value;

  if (!leftVideo || !rightVideo) {
    return;
  }

  if (Math.abs(rightVideo.currentTime - leftVideo.currentTime) > 0.08) {
    rightVideo.currentTime = leftVideo.currentTime;
  }

  if (rightVideo.playbackRate !== leftVideo.playbackRate) {
    rightVideo.playbackRate = leftVideo.playbackRate;
  }
}

async function playBoth() {
  const videos = [leftVideoRef.value, rightVideoRef.value].filter(Boolean);
  if (!videos.length) {
    return;
  }

  try {
    await Promise.all(videos.map((video) => video.play()));
    isPlaying.value = true;
  } catch {
    isPlaying.value = false;
  }
}

function pauseBoth() {
  const videos = [leftVideoRef.value, rightVideoRef.value].filter(Boolean);
  videos.forEach((video) => video.pause());
  isPlaying.value = false;
}

function togglePlayback() {
  if (isPlaying.value) {
    pauseBoth();
    return;
  }
  playBoth();
}

function handleLoadedMetadata() {
  const leftVideo = leftVideoRef.value;
  const rightVideo = rightVideoRef.value;
  duration.value = leftVideo?.duration || rightVideo?.duration || 0;
}

function handleTimeUpdate() {
  const leftVideo = leftVideoRef.value;
  if (!leftVideo) {
    return;
  }

  currentTime.value = leftVideo.currentTime;
  duration.value = leftVideo.duration || duration.value;
  syncSecondaryVideo();
}

function handleSeek(event) {
  const nextProgress = Number(event.target.value);
  const nextTime = (duration.value || 0) * nextProgress;

  [leftVideoRef.value, rightVideoRef.value].filter(Boolean).forEach((video) => {
    video.currentTime = nextTime;
  });
  currentTime.value = nextTime;
}

function updateSplit(clientX) {
  if (!stageRef.value) {
    return;
  }

  const rect = stageRef.value.getBoundingClientRect();
  const nextSplit = (clientX - rect.left) / rect.width;
  split.value = clamp(nextSplit, 0.05, 0.95);
}

function handlePointerDown(event) {
  isDragging.value = true;
  updateSplit(event.clientX);
  window.addEventListener('pointermove', handlePointerMove);
  window.addEventListener('pointerup', stopDragging);
}

function handlePointerMove(event) {
  if (!isDragging.value) {
    return;
  }
  updateSplit(event.clientX);
}

function stopDragging() {
  isDragging.value = false;
  window.removeEventListener('pointermove', handlePointerMove);
  window.removeEventListener('pointerup', stopDragging);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

watch(
  () => [props.leftSrc, props.rightSrc],
  async () => {
    const wasPlaying = isPlaying.value;
    pauseBoth();
    currentTime.value = 0;
    duration.value = 0;

    await nextTick();

    [leftVideoRef.value, rightVideoRef.value].filter(Boolean).forEach((video) => {
      video.load();
      video.currentTime = 0;
    });

    if (wasPlaying) {
      playBoth();
    }
  }
);

onMounted(() => {
  playBoth();
});

onBeforeUnmount(() => {
  stopDragging();
});
</script>

<template>
  <div class="compare-shell">
    <div class="compare-topline">
      <h3 class="compare-title">{{ title }}</h3>
      <button class="play-button" type="button" @click="togglePlayback">
        {{ isPlaying ? 'Pause' : 'Play' }}
      </button>
    </div>

    <div
      ref="stageRef"
      class="compare-stage"
      @pointerdown="handlePointerDown"
    >
      <video
        ref="rightVideoRef"
        class="compare-video"
        :src="rightSrc"
        muted
        loop
        playsinline
        preload="metadata"
        @loadedmetadata="handleLoadedMetadata"
      />

      <video
        ref="leftVideoRef"
        class="compare-video compare-video-overlay"
        :src="leftSrc"
        muted
        loop
        playsinline
        preload="metadata"
        :style="leftClipStyle"
        @loadedmetadata="handleLoadedMetadata"
        @timeupdate="handleTimeUpdate"
        @play="isPlaying = true"
        @pause="isPlaying = false"
      />

      <div class="compare-label compare-label-left">{{ leftLabel }}</div>
      <div class="compare-label compare-label-right">{{ rightLabel }}</div>

      <div class="compare-divider" :style="dividerStyle">
        <div class="compare-divider-line"></div>
        <div class="compare-divider-handle">&lt;&gt;</div>
      </div>
    </div>

    <div class="progress-row">
      <span class="progress-text">Drag center bar to compare</span>
      <input
        class="progress-slider"
        type="range"
        min="0"
        max="1"
        step="0.001"
        :value="progress"
        @input="handleSeek"
      >
    </div>
  </div>
</template>

<style scoped>
.compare-shell {
  border: 1px solid #dddddd;
  border-radius: 20px;
  background: #ffffff;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  padding: 16px;
}

.compare-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.compare-title {
  margin: 0;
  font-size: 18px;
  letter-spacing: 0.04em;
}

.play-button {
  border: none;
  border-radius: 999px;
  padding: 8px 14px;
  background: #1f2937;
  color: #ffffff;
  font: inherit;
  cursor: pointer;
}

.compare-stage {
  position: relative;
  overflow: hidden;
  border-radius: 16px;
  background:
    radial-gradient(circle at top, rgba(255, 255, 255, 0.18), transparent 42%),
    linear-gradient(135deg, #0f172a, #243b53);
  aspect-ratio: 1 / 1;
  touch-action: none;
  cursor: ew-resize;
}

.compare-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.compare-video-overlay {
  z-index: 2;
}

.compare-label {
  position: absolute;
  top: 14px;
  z-index: 4;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.72);
  color: #ffffff;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.compare-label-left {
  left: 14px;
}

.compare-label-right {
  right: 14px;
}

.compare-divider {
  position: absolute;
  top: 0;
  bottom: 0;
  z-index: 5;
  transform: translateX(-50%);
  pointer-events: none;
}

.compare-divider-line {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 0 12px rgba(0, 0, 0, 0.25);
}

.compare-divider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 40px;
  height: 40px;
  transform: translate(-50%, -50%);
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.96);
  color: #111827;
  font-size: 18px;
  font-weight: 700;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.25);
}

.progress-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 14px;
}

.progress-text {
  min-width: 150px;
  font-size: 12px;
  color: #6b7280;
}

.progress-slider {
  width: 100%;
}

@media (max-width: 768px) {
  .compare-shell {
    padding: 14px;
  }

  .compare-title {
    font-size: 16px;
  }

  .progress-row {
    flex-direction: column;
    align-items: stretch;
  }

  .progress-text {
    min-width: 0;
  }
}
</style>
