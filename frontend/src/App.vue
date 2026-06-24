<template>
  <div class="min-h-screen app-bg text-slate-900">
    <header class="sticky top-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div>
          <h1 class="text-2xl font-bold tracking-tight">ISR Segmentation</h1>
          <p class="mt-1 text-sm text-slate-600">Batch satellite image segmentation with queued inference</p>
        </div>
        <div class="text-right text-xs">
          <p :class="modelStatus.loaded ? 'text-emerald-700' : 'text-amber-700'">
            {{ modelStatus.loaded ? 'Model ready' : 'Checking model' }}
          </p>
          <p class="max-w-xs truncate text-slate-500">{{ modelStatus.model }}</p>
        </div>
      </div>
    </header>

    <main class="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-6 xl:grid-cols-[430px_1fr]">
      <section class="space-y-4">
        <div
          @click="fileInput?.click()"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="handleDrop"
          :class="[
            'panel cursor-pointer border-2 border-dashed p-6 transition',
            isDragging ? 'border-cyan-500 bg-cyan-50' : 'border-slate-300 hover:border-cyan-500'
          ]"
        >
          <input
            ref="fileInput"
            type="file"
            accept="image/*"
            multiple
            class="hidden"
            @change="handleFileSelect"
          />
          <div class="text-center">
            <p class="text-lg font-semibold">Add Images</p>
            <p class="mt-1 text-sm text-slate-600">Select multiple images or drop a batch here</p>
            <p class="mt-3 text-xs text-slate-500">PNG, JPG, TIFF, or other browser-supported image files up to 25 MB each</p>
          </div>
        </div>

        <div class="panel overflow-hidden">
          <button
            type="button"
            class="flex w-full items-center justify-between gap-3 px-5 py-4 text-left transition hover:bg-slate-50"
            @click="configPanelOpen = !configPanelOpen"
          >
            <div>
              <h2 class="text-lg font-semibold">Prompt & Settings</h2>
              <p class="text-sm text-slate-500">
                {{ runMode === 'aether' ? 'AETHER handoff configuration' : 'Prompt and SAM mask tuning' }}
              </p>
            </div>
            <span class="shrink-0 rounded-md border border-slate-300 px-2 py-1 text-xs font-semibold text-slate-600">
              {{ configPanelOpen ? 'Collapse' : 'Expand' }}
            </span>
          </button>

          <div v-if="configPanelOpen" class="border-t border-slate-200 p-5">
            <div class="grid grid-cols-2 gap-2 rounded-lg bg-slate-100 p-1">
              <button
                type="button"
                :class="[
                  'rounded-md px-3 py-2 text-xs font-semibold transition',
                  runMode === 'prompt' ? 'bg-white text-cyan-800 shadow-sm' : 'text-slate-600'
                ]"
                @click="runMode = 'prompt'"
              >
                Prompt Segmentation
              </button>
              <button
                type="button"
                :class="[
                  'rounded-md px-3 py-2 text-xs font-semibold transition',
                  runMode === 'aether' ? 'bg-white text-cyan-800 shadow-sm' : 'text-slate-600'
                ]"
                @click="runMode = 'aether'"
              >
                AETHER Handoff
              </button>
            </div>

            <label v-if="runMode === 'prompt'" class="mt-4 block text-sm font-semibold text-slate-800" for="prompt">
              Segmentation Prompt
            </label>
            <textarea
              v-if="runMode === 'prompt'"
              id="prompt"
              v-model="prompt"
              rows="4"
              placeholder="water bodies, roads, buildings, forest coverage..."
              class="mt-3 w-full resize-none rounded-lg border border-slate-300 bg-white p-3 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
            ></textarea>
            <div v-if="runMode === 'prompt'" class="mt-3 flex flex-wrap gap-2">
              <button
                v-for="suggestion in suggestedPrompts"
                :key="suggestion"
                type="button"
                class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 transition hover:border-cyan-500 hover:text-cyan-700"
                @click="prompt = suggestion"
              >
                {{ suggestion }}
              </button>
            </div>

            <div v-if="runMode === 'prompt'" class="mt-5 border-t border-slate-200 pt-5">
              <div class="flex items-center justify-between gap-3">
                <div>
                  <h3 class="text-sm font-semibold text-slate-800">Segmentation Parameters</h3>
                  <p class="text-xs text-slate-500">Adjust mask strictness, cleanup, and iterative refinement.</p>
                </div>
                <button
                  type="button"
                  class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-cyan-500 hover:text-cyan-700"
                  @click="resetSegmentationSettings"
                >
                  Reset
                </button>
              </div>

              <div class="mt-4 space-y-5">
                <div>
                  <div class="flex items-center justify-between gap-3">
                    <label class="text-sm font-semibold text-slate-800" for="mask-threshold">Mask Threshold</label>
                    <span class="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700">
                      {{ Number(segmentationSettings.samMaskThreshold).toFixed(1) }}
                    </span>
                  </div>
                  <input
                    id="mask-threshold"
                    v-model.number="segmentationSettings.samMaskThreshold"
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    class="mt-2 w-full accent-cyan-700"
                  />
                  <div class="mt-1 flex justify-between text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    <span>Min -2.0 · More fuzzy</span>
                    <span>Max 2.0 · Less fuzzy</span>
                  </div>
                  <p class="mt-1 text-xs leading-5 text-slate-500">
                    Default 0.0. Lower values include softer, uncertain mask pixels. Higher values keep only stronger SAM pixels.
                  </p>
                </div>

                <div>
                  <div class="flex items-center justify-between gap-3">
                    <label class="text-sm font-semibold text-slate-800" for="mask-expand">Mask Expand</label>
                    <span class="rounded-md bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700">
                      {{ segmentationSettings.maskExpandPx }} px
                    </span>
                  </div>
                  <input
                    id="mask-expand"
                    v-model.number="segmentationSettings.maskExpandPx"
                    type="range"
                    min="-31"
                    max="31"
                    step="1"
                    class="mt-2 w-full accent-cyan-700"
                  />
                  <div class="mt-1 flex justify-between text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    <span>Min -31 px · Erode</span>
                    <span>Max 31 px · Dilate</span>
                  </div>
                  <p class="mt-1 text-xs leading-5 text-slate-500">
                    Default 0 px. Negative values shrink mask edges to remove bleed. Positive values grow edges to fill missed borders.
                  </p>
                </div>

                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <label class="block">
                    <span class="text-sm font-semibold text-slate-800">Cleanup Pixels</span>
                    <input
                      v-model.number="segmentationSettings.maskCleanupPx"
                      type="number"
                      min="0"
                      max="31"
                      step="1"
                      class="mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    />
                    <span class="mt-1 block text-xs leading-5 text-slate-500">
                      Range 0-31 px, default 0. 1-3 px removes small speckles; 4-7 px closes small gaps;
                      8+ px can erase thin roads or shorelines. Pixels refer to the processed image, capped at 1024 px
                      on the longest side.
                    </span>
                  </label>

                  <label class="block">
                    <span class="text-sm font-semibold text-slate-800">Minimum Area</span>
                    <input
                      v-model.number="segmentationSettings.maskMinArea"
                      type="number"
                      min="0"
                      max="2000000"
                      step="50"
                      class="mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    />
                    <span class="mt-1 block text-xs leading-5 text-slate-500">
                      Range 0-2,000,000 px, default 0. Drops connected mask islands smaller than this many pixels.
                      Try 50-250 for tiny noise, 500-2,000 for small clutter, and higher values only for large targets.
                    </span>
                  </label>
                </div>

                <div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <label class="block">
                    <span class="text-sm font-semibold text-slate-800">Refinement Passes</span>
                    <input
                      v-model.number="segmentationSettings.refinementPasses"
                      type="number"
                      min="0"
                      max="3"
                      step="1"
                      class="mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    />
                    <span class="mt-1 block text-xs leading-5 text-slate-500">
                      Range 0-3 extra passes, default 0. Re-runs prompt generation and SAM on the masked image.
                      More passes are slower and may over-prune with Intersection mode.
                    </span>
                  </label>

                  <label class="block">
                    <span class="text-sm font-semibold text-slate-800">Refinement Mode</span>
                    <select
                      v-model="segmentationSettings.refinementMode"
                      class="mt-2 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    >
                      <option value="intersection">Intersection</option>
                      <option value="union">Union</option>
                      <option value="replace">Replace</option>
                    </select>
                    <span class="mt-1 block text-xs leading-5 text-slate-500">
                      Options: Intersection, Union, Replace. Default Intersection.
                      {{ refinementModeDescription }}
                    </span>
                  </label>
                </div>

                <label class="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <input
                    v-model="segmentationSettings.samMultimaskOutput"
                    type="checkbox"
                    class="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-700 focus:ring-cyan-600"
                  />
                  <span>
                    <span class="block text-sm font-semibold text-slate-800">Use SAM Multimask Best Score</span>
                    <span class="mt-1 block text-xs leading-5 text-slate-500">
                      Asks SAM for multiple candidate masks and keeps the highest scoring one for ambiguous targets.
                    </span>
                  </span>
                </label>
              </div>
            </div>

            <div v-else class="mt-4">
              <label class="block text-sm font-semibold text-slate-800" for="crop-id">
                Ultra-Sim Crop ID
              </label>
              <input
                id="crop-id"
                v-model="cropId"
                type="text"
                placeholder="Existing crop folder on the laptop"
                class="mt-3 w-full rounded-lg border border-slate-300 bg-white p-3 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
              />
              <p class="mt-2 text-xs leading-5 text-slate-500">
                Runs all 12 broad classes, pushes masks and probability vectors to Ultra-Sim,
                and starts local topology, blueprint, Blender, and Unreal handoff preparation.
              </p>
            </div>
          </div>
        </div>

        <div class="panel p-5">
          <div class="grid grid-cols-3 gap-3 text-center">
            <div>
              <p class="text-2xl font-bold text-slate-900">{{ queue.length }}</p>
              <p class="text-xs text-slate-500">Total</p>
            </div>
            <div>
              <p class="text-2xl font-bold text-emerald-700">{{ completedCount }}</p>
              <p class="text-xs text-slate-500">Done</p>
            </div>
            <div>
              <p class="text-2xl font-bold text-rose-700">{{ failedCount }}</p>
              <p class="text-xs text-slate-500">Failed</p>
            </div>
          </div>

          <div class="mt-5 grid grid-cols-2 gap-3">
            <button
              type="button"
              class="rounded-lg bg-cyan-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              :disabled="!canStartQueue"
              @click="startQueue"
            >
              {{ isQueueRunning ? 'Queue Running' : runMode === 'aether' ? 'Run AETHER Handoff' : `Start ${runnableCount || ''} Queue` }}
            </button>
            <button
              type="button"
              class="rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-50"
              :disabled="!finishedCount || isQueueRunning"
              @click="clearFinished"
            >
              Clear Finished
            </button>
          </div>
          <button
            v-if="failedCount"
            type="button"
            class="mt-3 w-full rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
            :disabled="isQueueRunning"
            @click="retryFailed"
          >
            Retry Failed Jobs
          </button>
        </div>

        <div v-if="currentJob" class="panel p-5">
          <div class="flex items-start justify-between gap-4">
            <div>
              <p class="text-xs font-semibold uppercase tracking-wide text-cyan-700">Live Progress</p>
              <h2 class="mt-1 truncate text-lg font-semibold">{{ currentJob.name }}</h2>
            </div>
            <span class="rounded-full bg-cyan-100 px-3 py-1 text-xs font-semibold text-cyan-800">
              {{ currentJob.progress }}%
            </span>
          </div>
          <div class="mt-4 h-2 overflow-hidden rounded-full bg-slate-200">
            <div
              class="h-full rounded-full bg-cyan-600 transition-all duration-500"
              :style="{ width: `${currentJob.progress}%` }"
            ></div>
          </div>
          <div class="mt-4 space-y-1 text-sm">
            <p class="font-medium text-slate-800">{{ currentJob.stage }}</p>
            <p class="text-slate-600">{{ currentJob.message }}</p>
            <p class="text-xs text-slate-500">
              Elapsed {{ formatDuration(currentJob.elapsedMs) }}
              <span v-if="queuePosition(currentJob)"> · {{ queuePosition(currentJob) }}</span>
            </p>
          </div>
        </div>

        <div v-if="error" class="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
          <div class="flex items-start justify-between gap-3">
            <p>{{ error }}</p>
            <button type="button" class="text-xs font-semibold underline" @click="error = null">Dismiss</button>
          </div>
        </div>
      </section>

      <section class="grid min-h-[620px] grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div class="panel overflow-hidden">
          <div class="border-b border-slate-200 px-5 py-4">
            <h2 class="text-lg font-semibold">Selected Image</h2>
            <p class="text-sm text-slate-500">{{ selectedJob ? selectedJob.name : 'No image selected' }}</p>
          </div>

          <div v-if="selectedJob" class="p-5">
            <div class="mb-4 flex flex-wrap gap-2">
              <button
                v-for="view in resultViews"
                :key="view.id"
                type="button"
                :disabled="!resultViewAvailable(selectedJob, view.id)"
                :class="[
                  'rounded-lg border px-3 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-40',
                  selectedView === view.id
                    ? 'border-cyan-700 bg-cyan-700 text-white'
                    : 'border-slate-300 bg-white text-slate-700 hover:border-cyan-600'
                ]"
                @click="selectedView = view.id"
              >
                {{ view.label }}
              </button>
            </div>

            <div class="overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
              <img
                v-if="selectedViewImage(selectedJob)"
                :src="selectedViewImage(selectedJob)"
                :alt="selectedViewAlt"
                class="max-h-[560px] w-full object-contain"
              />
              <div v-else class="flex h-80 items-center justify-center px-6 text-center text-sm text-slate-300">
                This view will be available when segmentation finishes.
              </div>
            </div>

            <div class="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div class="rounded-lg bg-slate-100 p-3">
                <p class="text-xs text-slate-500">Status</p>
                <p class="mt-1 text-sm font-semibold">{{ statusText(selectedJob) }}</p>
              </div>
              <div class="rounded-lg bg-slate-100 p-3">
                <p class="text-xs text-slate-500">Elapsed</p>
                <p class="mt-1 text-sm font-semibold">{{ formatDuration(selectedJob.elapsedMs) }}</p>
              </div>
              <div class="rounded-lg bg-slate-100 p-3">
                <p class="text-xs text-slate-500">Image</p>
                <p class="mt-1 truncate text-sm font-semibold">{{ selectedJob.dimensions }}</p>
              </div>
              <div class="rounded-lg bg-slate-100 p-3">
                <p class="text-xs text-slate-500">Targets</p>
                <p class="mt-1 text-sm font-semibold">{{ targetCount(selectedJob) }}</p>
              </div>
            </div>

            <div class="mt-4 rounded-lg border border-slate-200 bg-white p-4">
              <p class="text-xs font-semibold uppercase tracking-wide text-slate-500">Prompt</p>
              <p class="mt-1 text-sm text-slate-700">{{ selectedJob.prompt || prompt || 'No prompt assigned yet' }}</p>
              <p v-if="selectedJob.error" class="mt-3 text-sm text-rose-700">{{ selectedJob.error }}</p>
            </div>

            <div v-if="selectedJob.jobType === 'aether'" class="mt-4 rounded-lg border border-cyan-200 bg-cyan-50 p-4">
              <p class="text-xs font-semibold uppercase tracking-wide text-cyan-800">Ultra-Sim Handoff</p>
              <p class="mt-1 text-sm text-cyan-950">
                Crop {{ selectedJob.cropId }}:
                {{ selectedJob.handoff?.status === 'accepted' ? 'accepted by laptop' : selectedJob.handoff?.message || 'waiting' }}
              </p>
              <p v-if="selectedJob.handoff?.local_job_id" class="mt-1 text-xs text-cyan-800">
                Local job {{ selectedJob.handoff.local_job_id }}
              </p>
            </div>

            <div class="mt-4 rounded-lg border border-slate-800 bg-slate-950 text-slate-100">
              <div class="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                <div>
                  <p class="text-sm font-semibold">Resource Log</p>
                  <p class="text-xs text-slate-400">RAM and GPU memory snapshots while the job runs</p>
                </div>
                <span class="rounded-full bg-slate-800 px-2 py-1 text-xs text-slate-300">
                  {{ latestResourceLog(selectedJob).length }} latest
                  <span v-if="selectedJob.resourceLogTotal">
                    of {{ selectedJob.resourceLogTotal }}
                  </span>
                </span>
              </div>
              <div v-if="latestResourceLog(selectedJob).length" class="max-h-72 overflow-auto p-3 font-mono text-xs">
                <div
                  v-for="entry in latestResourceLog(selectedJob)"
                  :key="`${entry.time}-${entry.stage}`"
                  class="border-b border-slate-800 py-2 last:border-0"
                >
                  <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-400">
                    <span>{{ formatLogTime(entry.time) }}</span>
                    <span class="text-cyan-300">{{ entry.stage }}</span>
                  </div>
                  <p class="mt-1 text-slate-200">{{ formatResourceSnapshot(entry.snapshot) }}</p>
                </div>
              </div>
              <div v-else class="px-4 py-5 text-sm text-slate-400">
                Resource snapshots will appear once this job starts.
              </div>
            </div>

            <div class="mt-4 flex flex-wrap gap-3">
              <button
                type="button"
                class="rounded-lg bg-emerald-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                :disabled="!selectedViewImage(selectedJob)"
                @click="downloadJob(selectedJob)"
              >
                Download Current View
              </button>
              <button
                v-if="selectedJob.status === 'failed'"
                type="button"
                class="rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100"
                @click="retryJob(selectedJob)"
              >
                Retry
              </button>
            </div>
          </div>

          <div v-else class="flex h-full min-h-[520px] items-center justify-center p-8 text-center">
            <div>
              <p class="text-lg font-semibold text-slate-800">No image in the queue yet</p>
              <p class="mt-2 text-sm text-slate-500">Add one or more images to begin a queued segmentation run.</p>
            </div>
          </div>
        </div>

        <aside class="panel overflow-hidden">
          <button
            type="button"
            class="flex w-full items-center justify-between gap-3 border-b border-slate-200 px-4 py-4 text-left transition hover:bg-slate-50"
            @click="queuePanelOpen = !queuePanelOpen"
          >
            <div>
              <h2 class="text-lg font-semibold">Queue</h2>
              <p class="text-sm text-slate-500">{{ activeCount }} active, {{ completedCount }} complete</p>
            </div>
            <span class="shrink-0 rounded-md border border-slate-300 px-2 py-1 text-xs font-semibold text-slate-600">
              {{ queuePanelOpen ? 'Collapse' : 'Expand' }}
            </span>
          </button>

          <div v-if="queuePanelOpen">
            <div v-if="queue.length" class="max-h-[720px] divide-y divide-slate-200 overflow-auto">
              <article
                v-for="job in queue"
                :key="job.id"
                :class="[
                  'cursor-pointer p-4 transition hover:bg-slate-50',
                  selectedJobId === job.id ? 'bg-cyan-50' : 'bg-white'
                ]"
                @click="selectedJobId = job.id"
              >
                <div class="flex gap-3">
                  <img
                    v-if="job.previewAvailable"
                    :src="job.previewUrl"
                    alt=""
                    class="h-16 w-20 rounded-md border border-slate-200 object-cover"
                  />
                  <div v-else class="flex h-16 w-20 items-center justify-center rounded-md border border-slate-200 bg-slate-100 text-xs text-slate-500">
                    Image
                  </div>

                  <div class="min-w-0 flex-1">
                    <div class="flex items-start justify-between gap-2">
                      <p class="truncate text-sm font-semibold text-slate-800">{{ job.name }}</p>
                      <span :class="statusBadgeClass(job)">{{ statusText(job) }}</span>
                    </div>
                    <p class="mt-1 truncate text-xs text-slate-500">{{ job.dimensions }} · {{ formatBytes(job.size) }}</p>
                    <p class="mt-2 truncate text-xs text-slate-600">{{ job.stage }}</p>
                  </div>
                </div>

                <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-200">
                  <div
                    :class="progressBarClass(job)"
                    :style="{ width: `${job.progress}%` }"
                  ></div>
                </div>

                <div class="mt-3 flex items-center justify-between gap-2 text-xs text-slate-500">
                  <span>{{ formatDuration(job.elapsedMs) }}</span>
                  <div class="flex gap-2">
                    <button
                      v-if="job.status === 'failed'"
                      type="button"
                      class="font-semibold text-rose-700 hover:underline"
                      @click.stop="retryJob(job)"
                    >
                      Retry
                    </button>
                    <button
                      type="button"
                      class="font-semibold text-slate-600 hover:underline disabled:cursor-not-allowed disabled:text-slate-300"
                      :disabled="isJobBusy(job)"
                      @click.stop="removeJob(job)"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </article>
            </div>

            <div v-else class="p-8 text-center text-sm text-slate-500">
              The queue is empty.
            </div>
          </div>
        </aside>
      </section>
    </main>
  </div>
</template>

<script>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import axios from 'axios'

const MAX_FILE_SIZE = 25 * 1024 * 1024
const POLL_INTERVAL_MS = 2000
const resultViews = [
  { id: 'before', label: 'Before' },
  { id: 'overlay', label: 'Mask Overlay' },
  { id: 'mask', label: 'Mask Only' },
  { id: 'segmented', label: 'Segmented Image' }
]
const defaultSegmentationSettings = {
  samMaskThreshold: 0,
  samMultimaskOutput: false,
  maskMinArea: 0,
  maskCleanupPx: 0,
  maskExpandPx: 0,
  refinementPasses: 0,
  refinementMode: 'intersection'
}
const refinementModeDescriptions = {
  intersection: 'Keeps only pixels that survive every pass, which is best for reducing noise.',
  union: 'Keeps pixels found by any pass, which is best when the mask is missing pieces.',
  replace: 'Uses the newest pass only, which is useful for comparing whether iteration improves the mask.'
}
const RESOURCE_LOG_DISPLAY_LIMIT = 8

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

export default {
  name: 'App',
  setup() {
    const fileInput = ref(null)
    const prompt = ref('')
    const cropId = ref('')
    const runMode = ref('prompt')
    const queue = ref([])
    const selectedJobId = ref(null)
    const selectedView = ref('before')
    const currentJobId = ref(null)
    const isQueueRunning = ref(false)
    const isDragging = ref(false)
    const error = ref(null)
    const modelStatus = ref({ loaded: false, model: 'Think2Seg-RS-7B' })
    const configPanelOpen = ref(true)
    const queuePanelOpen = ref(true)
    const segmentationSettings = ref({ ...defaultSegmentationSettings })
    let nextLocalId = 1
    let elapsedTimer = null

    const suggestedPrompts = [
      'buildings',
      'roads and highways',
      'vegetation and forests',
      'water bodies',
      'urban development',
      'agricultural land'
    ]

    const selectedJob = computed(() => {
      return queue.value.find((job) => job.id === selectedJobId.value) || null
    })

    const currentJob = computed(() => {
      return queue.value.find((job) => job.id === currentJobId.value) || null
    })

    const selectedViewAlt = computed(() => {
      const labels = {
        before: 'Original image before segmentation',
        overlay: 'Image with segmentation mask overlay',
        mask: 'Segmentation mask only',
        segmented: 'Segmented image cutout'
      }
      return labels[selectedView.value] || 'Segmentation view'
    })

    const refinementModeDescription = computed(() => {
      return refinementModeDescriptions[segmentationSettings.value.refinementMode] ||
        refinementModeDescriptions.intersection
    })

    const completedCount = computed(() => {
      return queue.value.filter((job) => job.status === 'succeeded').length
    })

    const failedCount = computed(() => {
      return queue.value.filter((job) => job.status === 'failed').length
    })

    const finishedCount = computed(() => completedCount.value + failedCount.value)

    const activeCount = computed(() => {
      return queue.value.filter((job) => !['succeeded', 'failed'].includes(job.status)).length
    })

    const runnableCount = computed(() => {
      return queue.value.filter((job) => job.status === 'queued' && !job.serverJobId).length
    })

    const canStartQueue = computed(() => {
      const inputReady = runMode.value === 'aether'
        ? Boolean(cropId.value.trim())
        : Boolean(prompt.value.trim())
      return !isQueueRunning.value && runnableCount.value > 0 && inputReady
    })

    const checkModelStatus = async () => {
      try {
        const response = await axios.get('/api/health', { timeout: 5000 })
        modelStatus.value = {
          loaded: Boolean(response.data.model_loaded),
          model: response.data.model || 'Think2Seg-RS-7B'
        }
      } catch (err) {
        modelStatus.value = {
          loaded: false,
          model: 'Backend unavailable'
        }
      }
    }

    const handleFileSelect = (event) => {
      addFiles(event.target.files)
      event.target.value = ''
    }

    const handleDrop = (event) => {
      isDragging.value = false
      addFiles(event.dataTransfer.files)
    }

    const addFiles = (fileList) => {
      const files = Array.from(fileList || [])
      const rejected = []

      files.forEach((file) => {
        if (!file.type.startsWith('image/')) {
          rejected.push(`${file.name} is not an image`)
          return
        }
        if (file.size > MAX_FILE_SIZE) {
          rejected.push(`${file.name} is larger than 25 MB`)
          return
        }

        const job = createLocalJob(file)
        queue.value.push(job)
        loadImageDetails(job)
        if (!selectedJobId.value) {
          selectedJobId.value = job.id
        }
      })

      if (rejected.length) {
        error.value = rejected.join('; ')
      }
    }

    const createLocalJob = (file) => {
      return {
        id: `local-${Date.now()}-${nextLocalId++}`,
        serverJobId: null,
        file,
        name: file.name,
        size: file.size,
        previewUrl: URL.createObjectURL(file),
        previewAvailable: true,
        dimensions: 'Reading image',
        prompt: '',
        cropId: '',
        jobType: 'prompt',
        status: 'queued',
        progress: 0,
        uploadProgress: 0,
        stage: 'Queued',
        message: 'Ready to process',
        elapsedMs: 0,
        startedAt: null,
        completedAt: null,
        resourceLog: [],
        resourceLogTotal: 0,
        resourceSnapshot: null,
        settings: null,
        result: null,
        error: null
      }
    }

    const loadImageDetails = (job) => {
      const img = new Image()
      img.onload = () => {
        job.dimensions = `${img.naturalWidth} x ${img.naturalHeight}px`
        job.previewAvailable = true
      }
      img.onerror = () => {
        job.dimensions = 'Preview unavailable'
        job.previewAvailable = false
      }
      img.src = job.previewUrl
    }

    const normalizeSegmentationSettings = (settings = {}) => {
      const source = settings || {}
      const normalized = {
        samMaskThreshold: Number(source.samMaskThreshold ?? defaultSegmentationSettings.samMaskThreshold),
        samMultimaskOutput: Boolean(source.samMultimaskOutput ?? defaultSegmentationSettings.samMultimaskOutput),
        maskMinArea: Number(source.maskMinArea ?? defaultSegmentationSettings.maskMinArea),
        maskCleanupPx: Number(source.maskCleanupPx ?? defaultSegmentationSettings.maskCleanupPx),
        maskExpandPx: Number(source.maskExpandPx ?? defaultSegmentationSettings.maskExpandPx),
        refinementPasses: Number(source.refinementPasses ?? defaultSegmentationSettings.refinementPasses),
        refinementMode: source.refinementMode || defaultSegmentationSettings.refinementMode
      }
      normalized.samMaskThreshold = Math.min(2, Math.max(-2, normalized.samMaskThreshold || 0))
      normalized.maskMinArea = Math.min(2000000, Math.max(0, Math.round(normalized.maskMinArea || 0)))
      normalized.maskCleanupPx = Math.min(31, Math.max(0, Math.round(normalized.maskCleanupPx || 0)))
      normalized.maskExpandPx = Math.min(31, Math.max(-31, Math.round(normalized.maskExpandPx || 0)))
      normalized.refinementPasses = Math.min(3, Math.max(0, Math.round(normalized.refinementPasses || 0)))
      if (!refinementModeDescriptions[normalized.refinementMode]) {
        normalized.refinementMode = defaultSegmentationSettings.refinementMode
      }
      return normalized
    }

    const optionsToSegmentationSettings = (options, fallback = {}) => {
      if (!options) return normalizeSegmentationSettings(fallback)
      return normalizeSegmentationSettings({
        ...fallback,
        samMaskThreshold: options.sam_mask_threshold,
        samMultimaskOutput: options.sam_multimask_output,
        maskMinArea: options.mask_min_area,
        maskCleanupPx: options.mask_cleanup_px,
        maskExpandPx: options.mask_expand_px,
        refinementPasses: options.refinement_passes,
        refinementMode: options.refinement_mode
      })
    }

    const appendSegmentationSettings = (formData, settings) => {
      const normalized = normalizeSegmentationSettings(settings)
      formData.append('sam_mask_threshold', String(normalized.samMaskThreshold))
      formData.append('sam_multimask_output', String(Boolean(normalized.samMultimaskOutput)))
      formData.append('mask_min_area', String(normalized.maskMinArea))
      formData.append('mask_cleanup_px', String(normalized.maskCleanupPx))
      formData.append('mask_expand_px', String(normalized.maskExpandPx))
      formData.append('refinement_passes', String(normalized.refinementPasses))
      formData.append('refinement_mode', normalized.refinementMode)
    }

    const resetSegmentationSettings = () => {
      segmentationSettings.value = { ...defaultSegmentationSettings }
    }

    const startQueue = async () => {
      if (isQueueRunning.value) return

      const queuePrompt = prompt.value.trim()
      const queueCropId = cropId.value.trim()
      const queueMode = runMode.value
      const queueSettings = normalizeSegmentationSettings(segmentationSettings.value)
      if (queueMode === 'prompt' && !queuePrompt) {
        error.value = 'Enter a segmentation prompt before starting the queue.'
        return
      }
      if (queueMode === 'aether' && !queueCropId) {
        error.value = 'Enter the existing Ultra-Sim crop ID before starting the handoff.'
        return
      }

      isQueueRunning.value = true
      error.value = null

      try {
        while (true) {
          const nextJob = queue.value.find((job) => job.status === 'queued' && !job.serverJobId)
          if (!nextJob) break
          await processJob(nextJob, queuePrompt, queueMode, queueCropId, queueSettings)
        }
      } finally {
        isQueueRunning.value = false
        currentJobId.value = null
      }
    }

    const processJob = async (job, queuePrompt, queueMode, queueCropId, queueSettings) => {
      currentJobId.value = job.id
      selectedJobId.value = job.id
      Object.assign(job, {
        prompt: queueMode === 'aether' ? '12-class AETHER inference' : queuePrompt,
        cropId: queueMode === 'aether' ? queueCropId : '',
        jobType: queueMode,
        settings: queueMode === 'prompt' ? { ...queueSettings } : null,
        status: 'uploading',
        progress: 1,
        uploadProgress: 0,
        stage: 'Uploading image',
        message: 'Sending image to backend',
        startedAt: Date.now(),
        completedAt: null,
        elapsedMs: 0,
        result: null,
        error: null,
        serverJobId: null
      })

      try {
        const formData = new FormData()
        formData.append('file', job.file, job.file.name)
        if (queueMode === 'aether') {
          formData.append('crop_id', queueCropId)
        } else {
          formData.append('prompt', queuePrompt)
          appendSegmentationSettings(formData, queueSettings)
        }

        const endpoint = queueMode === 'aether' ? '/api/aether/jobs' : '/api/segment/jobs'
        let response = null
        for (let attempt = 1; attempt <= 3; attempt += 1) {
          try {
            response = await axios.post(endpoint, formData, {
              headers: { 'Content-Type': 'multipart/form-data' },
              timeout: 0,
              onUploadProgress: (event) => {
                if (!event.total) return
                job.uploadProgress = Math.round((event.loaded / event.total) * 100)
                job.progress = Math.max(job.progress, Math.min(15, Math.round(job.uploadProgress * 0.15)))
                job.message = `Upload ${job.uploadProgress}% complete`
              }
            })
            break
          } catch (err) {
            if (!isTransientGatewayError(err) || attempt === 3) {
              throw err
            }
            job.stage = 'Backend warming up'
            job.message = `Gateway returned ${err.response?.status}; retrying upload ${attempt + 1} of 3`
            await wait(1500 * attempt)
          }
        }

        applyServerJob(job, response.data)
        await pollServerJob(job)
      } catch (err) {
        const message = getErrorMessage(err)
        Object.assign(job, {
          status: 'failed',
          progress: 100,
          stage: 'Failed',
          message,
          error: message,
          completedAt: Date.now()
        })
        job.elapsedMs = job.startedAt ? job.completedAt - job.startedAt : job.elapsedMs
        error.value = `${job.name}: ${message}`
      }
    }

    const pollServerJob = async (job) => {
      if (!job.serverJobId) {
        throw new Error('Backend did not return a job id')
      }

      while (true) {
        const jobPath = job.jobType === 'aether' ? 'aether/jobs' : 'segment/jobs'
        const response = await axios.get(`/api/${jobPath}/${job.serverJobId}`, {
          timeout: 10000
        })
        applyServerJob(job, response.data)

        if (response.data.status === 'succeeded') {
          job.completedAt = Date.now()
          job.elapsedMs = job.startedAt ? job.completedAt - job.startedAt : job.elapsedMs
          return
        }
        if (response.data.status === 'failed') {
          throw new Error(response.data.error || response.data.message || 'Segmentation failed')
        }

        await wait(POLL_INTERVAL_MS)
      }
    }

    const applyServerJob = (job, serverJob) => {
      job.serverJobId = serverJob.id || job.serverJobId
      job.status = serverJob.status || job.status
      job.progress = Math.max(job.progress || 0, Number(serverJob.progress || 0))
      job.stage = serverJob.stage || job.stage
      job.message = serverJob.message || job.message
      job.result = serverJob.result || job.result
      job.error = serverJob.error || job.error
      job.resourceLog = serverJob.resource_log || job.resourceLog || []
      job.resourceLogTotal = Number(serverJob.resource_log_total ?? job.resourceLog.length)
      job.resourceSnapshot = serverJob.resource_snapshot || job.resourceSnapshot || null
      job.handoff = serverJob.handoff || job.handoff || null
      const returnedOptions = serverJob.options || serverJob.result?.options
      if (returnedOptions) {
        job.settings = optionsToSegmentationSettings(returnedOptions, job.settings)
      }

      if (job.status === 'succeeded') {
        job.progress = 100
        job.completedAt = job.completedAt || Date.now()
        if (selectedJobId.value === job.id) {
          selectedView.value = 'overlay'
        }
      }
    }

    const retryJob = (job) => {
      if (isJobBusy(job)) return
      runMode.value = job.jobType || 'prompt'
      if (job.jobType === 'aether') {
        cropId.value = job.cropId || ''
      } else {
        prompt.value = job.prompt || ''
        if (job.settings) {
          segmentationSettings.value = normalizeSegmentationSettings(job.settings)
        }
      }
      Object.assign(job, {
        serverJobId: null,
        status: 'queued',
        progress: 0,
        uploadProgress: 0,
        stage: 'Queued',
        message: 'Ready to retry',
        resourceLog: [],
        resourceLogTotal: 0,
        resourceSnapshot: null,
        result: null,
        error: null,
        startedAt: null,
        completedAt: null,
        elapsedMs: 0
      })
      selectedJobId.value = job.id
      startQueue()
    }

    const retryFailed = () => {
      queue.value
        .filter((job) => job.status === 'failed')
        .forEach((job) => {
          Object.assign(job, {
            serverJobId: null,
            status: 'queued',
            progress: 0,
            uploadProgress: 0,
            stage: 'Queued',
            message: 'Ready to retry',
            resourceLog: [],
            resourceLogTotal: 0,
            resourceSnapshot: null,
            result: null,
            error: null,
            startedAt: null,
            completedAt: null,
            elapsedMs: 0
          })
        })
      startQueue()
    }

    const removeJob = (job) => {
      if (isJobBusy(job)) return
      URL.revokeObjectURL(job.previewUrl)
      const index = queue.value.findIndex((item) => item.id === job.id)
      queue.value = queue.value.filter((item) => item.id !== job.id)

      if (selectedJobId.value === job.id) {
        selectedJobId.value = queue.value[index]?.id || queue.value[index - 1]?.id || queue.value[0]?.id || null
      }
    }

    const clearFinished = () => {
      const removedIds = new Set(
        queue.value
          .filter((job) => ['succeeded', 'failed'].includes(job.status))
          .map((job) => job.id)
      )

      queue.value.forEach((job) => {
        if (removedIds.has(job.id)) {
          URL.revokeObjectURL(job.previewUrl)
        }
      })
      queue.value = queue.value.filter((job) => !removedIds.has(job.id))

      if (selectedJobId.value && removedIds.has(selectedJobId.value)) {
        selectedJobId.value = queue.value[0]?.id || null
      }
    }

    const downloadJob = (job) => {
      const imageSource = selectedViewImage(job)
      if (!imageSource) return
      const link = document.createElement('a')
      link.href = imageSource
      link.download = `isr-${selectedView.value}-${sanitizeFileName(job.name)}-${Date.now()}.png`
      link.click()
    }

    const resultViewAvailable = (job, viewId) => {
      if (!job) return false
      if (viewId === 'before') return Boolean(job.previewAvailable)
      if (viewId === 'overlay') return Boolean(job.result?.visualization_base64)
      if (viewId === 'mask') return Boolean(job.result?.mask_base64)
      if (viewId === 'segmented') return Boolean(job.result?.segmented_image_base64)
      return false
    }

    const selectedViewImage = (job) => {
      if (!job) return ''
      if (selectedView.value === 'before' && job.previewAvailable) {
        return job.previewUrl
      }
      if (selectedView.value === 'overlay' && job.result?.visualization_base64) {
        return `data:image/png;base64,${job.result.visualization_base64}`
      }
      if (selectedView.value === 'mask' && job.result?.mask_base64) {
        return `data:image/png;base64,${job.result.mask_base64}`
      }
      if (selectedView.value === 'segmented' && job.result?.segmented_image_base64) {
        return `data:image/png;base64,${job.result.segmented_image_base64}`
      }
      if (job.previewAvailable) {
        return job.previewUrl
      }
      return ''
    }

    const isJobBusy = (job) => {
      return ['uploading', 'running'].includes(job.status) || (job.status === 'queued' && Boolean(job.serverJobId))
    }

    const statusText = (job) => {
      const labels = {
        queued: job.serverJobId ? 'Waiting' : 'Queued',
        uploading: 'Uploading',
        running: 'Running',
        succeeded: 'Done',
        failed: 'Failed'
      }
      return labels[job.status] || job.status
    }

    const statusBadgeClass = (job) => {
      const base = 'shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold'
      if (job.status === 'succeeded') return `${base} bg-emerald-100 text-emerald-800`
      if (job.status === 'failed') return `${base} bg-rose-100 text-rose-800`
      if (isJobBusy(job)) return `${base} bg-cyan-100 text-cyan-800`
      return `${base} bg-slate-100 text-slate-700`
    }

    const progressBarClass = (job) => {
      const base = 'h-full rounded-full transition-all duration-500'
      if (job.status === 'succeeded') return `${base} bg-emerald-600`
      if (job.status === 'failed') return `${base} bg-rose-600`
      return `${base} bg-cyan-600`
    }

    const queuePosition = (job) => {
      const activeJobs = queue.value.filter((item) => !['succeeded', 'failed'].includes(item.status))
      const index = activeJobs.findIndex((item) => item.id === job.id)
      if (index < 0) return ''
      return `${index + 1} of ${activeJobs.length} active`
    }

    const targetCount = (job) => {
      const count = job?.result?.sam_prompts?.length
      if (typeof count === 'number') return count
      if (typeof job?.result?.segment_count === 'number') return job.result.segment_count
      if (job?.status === 'succeeded') return 0
      return 'Pending'
    }

    const latestResourceLog = (job) => {
      return (job?.resourceLog || []).slice(-RESOURCE_LOG_DISPLAY_LIMIT)
    }

    const formatDuration = (ms = 0) => {
      const totalSeconds = Math.max(0, Math.floor(ms / 1000))
      const hours = Math.floor(totalSeconds / 3600)
      const minutes = Math.floor((totalSeconds % 3600) / 60)
      const seconds = totalSeconds % 60
      if (hours) {
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
      }
      return `${minutes}:${String(seconds).padStart(2, '0')}`
    }

    const formatBytes = (bytes = 0) => {
      if (!bytes) return '0 B'
      const units = ['B', 'KB', 'MB', 'GB']
      const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
      const value = bytes / Math.pow(1024, index)
      return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
    }

    const formatLogTime = (timestamp) => {
      if (!timestamp) return '--:--:--'
      return new Date(timestamp * 1000).toLocaleTimeString()
    }

    const formatResourceSnapshot = (snapshot) => {
      if (!snapshot) return 'No resource snapshot available'

      const parts = []
      const ram = snapshot.ram || {}
      if (ram.total && ram.used) {
        parts.push(`RAM ${formatBytes(ram.used)} / ${formatBytes(ram.total)}`)
      }
      if (ram.process_rss) {
        parts.push(`Process ${formatBytes(ram.process_rss)}`)
      }

      ;(snapshot.gpus || []).forEach((gpu) => {
        parts.push(
          `GPU${gpu.index} ${formatBytes(gpu.used)} / ${formatBytes(gpu.total)} used, ` +
          `${formatBytes(gpu.allocated)} allocated, ${formatBytes(gpu.reserved)} reserved`
        )
      })

      return parts.length ? parts.join(' | ') : 'No CUDA/RAM data available'
    }

    const sanitizeFileName = (name) => {
      return name.replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-|-$/g, '') || 'image'
    }

    const isTransientGatewayError = (err) => {
      return [502, 503, 504].includes(Number(err.response?.status))
    }

    const getErrorMessage = (err) => {
      const status = Number(err.response?.status)
      if ([502, 503, 504].includes(status)) {
        return `Backend gateway error (${status}). The backend may still be starting or temporarily busy.`
      }
      return err.response?.data?.detail || err.message || 'Segmentation failed'
    }

    onMounted(() => {
      checkModelStatus()
      elapsedTimer = window.setInterval(() => {
        const now = Date.now()
        queue.value.forEach((job) => {
          if (job.startedAt && !job.completedAt && !['succeeded', 'failed'].includes(job.status)) {
            job.elapsedMs = now - job.startedAt
          }
        })
      }, 1000)
    })

    onUnmounted(() => {
      if (elapsedTimer) {
        window.clearInterval(elapsedTimer)
      }
      queue.value.forEach((job) => URL.revokeObjectURL(job.previewUrl))
    })

    return {
      fileInput,
      prompt,
      cropId,
      runMode,
      queue,
      selectedJobId,
      selectedView,
      selectedViewAlt,
      selectedJob,
      currentJob,
      isQueueRunning,
      isDragging,
      error,
      modelStatus,
      configPanelOpen,
      queuePanelOpen,
      segmentationSettings,
      refinementModeDescription,
      suggestedPrompts,
      resultViews,
      completedCount,
      failedCount,
      finishedCount,
      activeCount,
      runnableCount,
      canStartQueue,
      handleFileSelect,
      handleDrop,
      startQueue,
      resetSegmentationSettings,
      retryJob,
      retryFailed,
      removeJob,
      clearFinished,
      downloadJob,
      resultViewAvailable,
      selectedViewImage,
      isJobBusy,
      statusText,
      statusBadgeClass,
      progressBarClass,
      queuePosition,
      targetCount,
      latestResourceLog,
      formatDuration,
      formatBytes,
      formatLogTime,
      formatResourceSnapshot
    }
  }
}
</script>
