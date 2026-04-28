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
const leftReady = ref(false);
const rightReady = ref(false);
const shouldAutoPlayWhenReady = ref(true);

const leftClipStyle = computed(() => ({
  clipPath: `inset(0 ${100 - split.value * 100}% 0 0)`,
}));

const dividerStyle = computed(() => ({
  left: `${split.value * 100}%`,
}));

const isReady = computed(() => leftReady.value && rightReady.value);

const progress = computed(() => {
  if (!duration.value) {
    return 0;
  }
  return currentTime.value / duration.value;
});

const playButtonLabel = computed(() => {
  if (!isReady.value) {
    return 'Loading...';
  }
  return isPlaying.value ? 'Pause' : 'Play';
});

function resetReadyState() {
  leftReady.value = false;
  rightReady.value = false;
}

function syncLeftVideoToRight() {
  const leftVideo = leftVideoRef.value;
  const rightVideo = rightVideoRef.value;

  if (!leftVideo || !rightVideo) {
    return;
  }

  if (Math.abs(leftVideo.currentTime - rightVideo.currentTime) > 0.08) {
    leftVideo.currentTime = rightVideo.currentTime;
  }

  if (leftVideo.playbackRate !== rightVideo.playbackRate) {
    leftVideo.playbackRate = rightVideo.playbackRate;
  }
}

async function playBoth() {
  const videos = [leftVideoRef.value, rightVideoRef.value].filter(Boolean);
  if (!videos.length) {
    return;
  }

  shouldAutoPlayWhenReady.value = true;

  if (!isReady.value) {
    isPlaying.value = false;
    return;
  }

  try {
    await Promise.all(videos.map((video) => video.play()));
    isPlaying.value = true;
  } catch {
    isPlaying.value = false;
  }
}

function pauseBoth(updateAutoplayPreference = true) {
  const videos = [leftVideoRef.value, rightVideoRef.value].filter(Boolean);
  videos.forEach((video) => video.pause());
  isPlaying.value = false;

  if (updateAutoplayPreference) {
    shouldAutoPlayWhenReady.value = false;
  }
}

function togglePlayback() {
  if (!isReady.value) {
    return;
  }

  if (isPlaying.value) {
    pauseBoth();
    return;
  }
  playBoth();
}

function handleLoadedMetadata() {
  const leftVideo = leftVideoRef.value;
  const rightVideo = rightVideoRef.value;
  duration.value = rightVideo?.duration || leftVideo?.duration || 0;
}

function handleVideoReady(side) {
  if (side === 'left') {
    leftReady.value = true;
  } else {
    rightReady.value = true;
  }

  handleLoadedMetadata();

  if (!isReady.value) {
    return;
  }

  syncLeftVideoToRight();

  if (shouldAutoPlayWhenReady.value) {
    playBoth();
  }
}

function handleTimeUpdate() {
  const rightVideo = rightVideoRef.value;
  if (!rightVideo) {
    return;
  }

  currentTime.value = rightVideo.currentTime;
  duration.value = rightVideo.duration || duration.value;
  syncLeftVideoToRight();
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
    const shouldResumePlayback = isPlaying.value || shouldAutoPlayWhenReady.value;
    pauseBoth(false);
    shouldAutoPlayWhenReady.value = shouldResumePlayback;
    currentTime.value = 0;
    duration.value = 0;
    resetReadyState();

    await nextTick();

    [leftVideoRef.value, rightVideoRef.value].filter(Boolean).forEach((video) => {
      video.load();
      video.currentTime = 0;
    });
  }
);

onMounted(() => {
  resetReadyState();
});

onBeforeUnmount(() => {
  stopDragging();
});
</script>

<template>
  <div class="compare-shell">
    <div class="compare-topline">
      <h3 class="compare-title">{{ title }}</h3>
      <button class="play-button" type="button" :disabled="!isReady" @click="togglePlayback">
        {{ playButtonLabel }}
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
        :class="{ 'compare-video-hidden': !isReady }"
        @loadedmetadata="handleLoadedMetadata"
        @loadeddata="handleVideoReady('right')"
        @timeupdate="handleTimeUpdate"
        @play="isPlaying = true"
        @pause="isPlaying = false"
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
        :class="{ 'compare-video-hidden': !isReady }"
        @loadedmetadata="handleLoadedMetadata"
        @loadeddata="handleVideoReady('left')"
      />

      <div v-if="!isReady" class="compare-loading">
        <span class="compare-loading-text">Loading both videos...</span>
      </div>

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
        :disabled="!isReady"
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

.play-button:disabled {
  opacity: 0.65;
  cursor: wait;
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
  transition: opacity 0.2s ease;
}

.compare-video-overlay {
  z-index: 2;
}

.compare-video-hidden {
  opacity: 0;
}

.compare-loading {
  position: absolute;
  inset: 0;
  z-index: 3;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.6), rgba(36, 59, 83, 0.35));
}

.compare-loading-text {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.92);
  color: #111827;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
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
