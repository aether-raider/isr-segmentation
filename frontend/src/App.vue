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
              Layer Prompts
            </label>
            <textarea
              v-if="runMode === 'prompt'"
              id="prompt"
              v-model="prompt"
              rows="4"
              placeholder="buildings&#10;trees&#10;water bodies&#10;shopping mall, basketball court and residential buildings"
              class="mt-3 w-full resize-none rounded-lg border border-slate-300 bg-white p-3 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
            ></textarea>
            <p v-if="runMode === 'prompt'" class="mt-2 text-xs leading-5 text-slate-500">
              Enter one layer per line. A line can still describe multiple objects that should belong to the same layer.
            </p>
            <div v-if="runMode === 'prompt'" class="mt-3 flex flex-wrap gap-2">
              <button
                v-for="suggestion in suggestedPrompts"
                :key="suggestion"
                type="button"
                class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 transition hover:border-cyan-500 hover:text-cyan-700"
                @click="appendPromptLayer(suggestion)"
              >
                {{ suggestion }}
              </button>
            </div>

            <div v-if="runMode === 'prompt'" class="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div class="flex items-center justify-between gap-3">
                <div>
                  <h3 class="text-sm font-semibold text-slate-800">Prompt Provider</h3>
                  <p class="text-xs leading-5 text-slate-500">Choose the local Think2Seg prompter or a BYOK cloud model through LiteLLM.</p>
                </div>
                <span class="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-700">
                  {{ promptProviderSettings.promptProvider === 'cloud' ? 'Cloud' : 'Local' }}
                </span>
              </div>
              <div class="mt-3 grid grid-cols-2 gap-2 rounded-lg bg-white p-1">
                <button
                  type="button"
                  :class="[
                    'rounded-md px-3 py-2 text-xs font-semibold transition',
                    promptProviderSettings.promptProvider === 'local' ? 'bg-cyan-700 text-white' : 'text-slate-600 hover:bg-slate-50'
                  ]"
                  @click="promptProviderSettings.promptProvider = 'local'"
                >
                  Local
                </button>
                <button
                  type="button"
                  :class="[
                    'rounded-md px-3 py-2 text-xs font-semibold transition',
                    promptProviderSettings.promptProvider === 'cloud' ? 'bg-cyan-700 text-white' : 'text-slate-600 hover:bg-slate-50'
                  ]"
                  @click="promptProviderSettings.promptProvider = 'cloud'"
                >
                  Cloud BYOK
                </button>
              </div>

              <div v-if="promptProviderSettings.promptProvider === 'cloud'" class="mt-4 space-y-3">
                <label class="block">
                  <span class="text-xs font-semibold text-slate-700">Provider Preset</span>
                  <select
                    v-model="promptProviderSettings.providerPreset"
                    class="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    @change="applyProviderPreset"
                  >
                    <option v-for="preset in cloudProviderPresets" :key="preset.id" :value="preset.id">
                      {{ preset.label }}
                    </option>
                  </select>
                </label>
                <label class="block">
                  <span class="text-xs font-semibold text-slate-700">LiteLLM Model ID</span>
                  <input
                    v-model="promptProviderSettings.litellmModel"
                    type="text"
                    placeholder="openai/gpt-4o, anthropic/claude-3-5-sonnet-latest..."
                    class="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                  />
                </label>
                <label class="block">
                  <span class="text-xs font-semibold text-slate-700">API Key</span>
                  <input
                    v-model="promptProviderSettings.litellmApiKey"
                    type="password"
                    :placeholder="promptProviderSettings.hasBackendApiKey ? 'Using saved backend key unless replaced' : 'Paste provider API key'"
                    autocomplete="off"
                    class="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                  />
                </label>
                <label class="block">
                  <span class="text-xs font-semibold text-slate-700">Custom API Base</span>
                  <input
                    v-model="promptProviderSettings.litellmApiBase"
                    type="text"
                    placeholder="Optional OpenAI-compatible base URL"
                    class="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                  />
                </label>
              </div>
              <div class="mt-3 flex items-center justify-between gap-3">
                <p class="text-xs text-slate-500">
                  {{ promptProviderSettings.message || 'Per-job settings are sent only when a segmentation starts.' }}
                </p>
                <button
                  type="button"
                  class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-cyan-500 hover:text-cyan-700"
                  @click="applyPromptProviderSettings"
                >
                  Apply
                </button>
              </div>
            </div>

            <div v-if="runMode === 'prompt'" class="mt-5 border-t border-slate-200 pt-5">
              <div class="flex items-center justify-between gap-3">
                <div>
                  <h3 class="text-sm font-semibold text-slate-800">Segmentation Parameters</h3>
                  <p class="text-xs text-slate-500">Starts from Think2Seg defaults; adjust only when a run needs tuning.</p>
                </div>
                <button
                  type="button"
                  class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-cyan-500 hover:text-cyan-700"
                  @click="resetSegmentationSettings"
                >
                  Think2Seg Defaults
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

                <div class="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div class="flex items-center justify-between gap-3">
                    <div>
                      <p class="text-sm font-semibold text-slate-800">GPU Memory Target</p>
                      <p class="text-xs leading-5 text-slate-500">
                        Uses this share of currently free VRAM when the Think2Seg model is loaded.
                      </p>
                    </div>
                    <span class="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-700">
                      {{ Math.round(runtimeSettings.modelGpuMemoryUtilization * 100) }}%
                    </span>
                  </div>
                  <input
                    v-model.number="runtimeSettings.modelGpuMemoryUtilization"
                    type="range"
                    min="0.5"
                    max="0.99"
                    step="0.01"
                    class="mt-3 w-full accent-cyan-700"
                  />
                  <label class="mt-3 block">
                    <span class="text-xs font-semibold text-slate-700">Optional hard cap per GPU in GB</span>
                    <input
                      v-model.number="runtimeSettings.modelMaxMemoryGb"
                      type="number"
                      min="0"
                      step="1"
                      placeholder="0 = no hard cap"
                      class="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 text-sm focus:border-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-100"
                    />
                  </label>
                  <div class="mt-3 flex items-center justify-between gap-3">
                    <p class="text-xs text-slate-500">
                      {{ runtimeSettings.message || (runtimeSettings.modelLoaded ? 'Model is loaded; changes apply next load.' : 'Applies before the first model load.') }}
                    </p>
                    <button
                      type="button"
                      class="rounded-md bg-cyan-700 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-cyan-800"
                      @click="applyRuntimeSettings"
                    >
                      Apply
                    </button>
                  </div>
                </div>
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

        <div class="panel overflow-hidden">
          <button
            type="button"
            class="flex w-full items-center justify-between gap-3 px-5 py-4 text-left transition hover:bg-slate-50"
            @click="historyPanelOpen = !historyPanelOpen"
          >
            <div>
              <h2 class="text-lg font-semibold">Past Outputs</h2>
              <p class="text-sm text-slate-500">{{ outputHistory.length }} saved runs</p>
            </div>
            <span class="shrink-0 rounded-md border border-slate-300 px-2 py-1 text-xs font-semibold text-slate-600">
              {{ historyPanelOpen ? 'Collapse' : 'Expand' }}
            </span>
          </button>

          <div v-if="historyPanelOpen" class="border-t border-slate-200">
            <div class="flex items-center justify-between gap-3 px-5 py-3">
              <p class="text-xs text-slate-500">Saved in the backend output folder.</p>
              <button
                type="button"
                class="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-cyan-500 hover:text-cyan-700"
                @click="loadOutputHistory"
              >
                {{ historyLoading ? 'Loading' : 'Refresh' }}
              </button>
            </div>
            <div v-if="outputHistory.length" class="max-h-80 divide-y divide-slate-200 overflow-auto">
              <button
                v-for="output in outputHistory"
                :key="output.id"
                type="button"
                class="flex w-full gap-3 p-4 text-left transition hover:bg-slate-50"
                @click="openSavedOutput(output)"
              >
                <img
                  v-if="output.segmented_image_url || output.visualization_url || output.original_image_url"
                  :src="output.segmented_image_url || output.visualization_url || output.original_image_url"
                  alt=""
                  class="h-14 w-16 rounded-md border border-slate-200 bg-slate-950 object-contain"
                />
                <div v-else class="flex h-14 w-16 items-center justify-center rounded-md border border-slate-200 bg-slate-100 text-xs text-slate-500">
                  Saved
                </div>
                <div class="min-w-0 flex-1">
                  <p class="truncate text-sm font-semibold text-slate-800">{{ output.filename || output.id }}</p>
                  <p class="mt-1 truncate text-xs text-slate-500">{{ output.prompt || 'No prompt' }}</p>
                  <p class="mt-1 text-[11px] text-slate-400">
                    {{ output.target_count || 0 }} targets · {{ output.pass_count || 0 }} passes
                  </p>
                </div>
              </button>
            </div>
            <div v-else class="px-5 py-5 text-sm text-slate-500">
              No saved outputs yet.
            </div>
          </div>
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
          <div class="mt-4 grid gap-2">
            <div
              v-for="item in progressChecklist(currentJob)"
              :key="item.id"
              :class="[
                'flex items-center gap-3 rounded-md border px-3 py-2 text-xs',
                checklistItemClass(item)
              ]"
            >
              <span :class="['h-2.5 w-2.5 rounded-full', checklistDotClass(item)]"></span>
              <span class="font-semibold">{{ item.label }}</span>
              <span class="ml-auto text-[11px] uppercase tracking-wide">{{ item.state }}</span>
            </div>
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

            <div
              v-if="selectedView === 'layers' && layerOutputs(selectedJob).length"
              class="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
              <div class="flex items-center justify-between gap-3">
                <div>
                  <p class="text-sm font-semibold text-slate-800">Prompt Layers</p>
                  <p class="text-xs text-slate-500">Each line from the prompt box is preserved as its own layer.</p>
                </div>
                <span class="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-600">
                  {{ selectedLayerIndex + 1 }} of {{ layerOutputs(selectedJob).length }}
                </span>
              </div>
              <div class="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <button
                  v-for="(layerOutput, index) in layerOutputs(selectedJob)"
                  :key="`layer-${layerOutput.layer_number || index}`"
                  type="button"
                  :class="[
                    'overflow-hidden rounded-lg border bg-white text-left transition hover:border-cyan-600',
                    selectedLayerIndex === index ? 'border-cyan-700 ring-2 ring-cyan-100' : 'border-slate-200'
                  ]"
                  @click="selectedLayerIndex = index"
                >
                  <img
                    v-if="layerSegmentedImage(layerOutput)"
                    :src="layerSegmentedImage(layerOutput)"
                    alt=""
                    class="h-20 w-full bg-slate-950 object-contain"
                  />
                  <div v-else class="flex h-20 items-center justify-center bg-slate-100 text-xs text-slate-500">
                    No image
                  </div>
                  <div class="p-2">
                    <p class="text-xs font-semibold text-slate-800">
                      Layer {{ layerOutput.layer_number || index + 1 }}
                    </p>
                    <p class="mt-0.5 truncate text-[11px] text-slate-500">
                      {{ layerOutput.layer_prompt || layerOutput.prompt }}
                    </p>
                  </div>
                </button>
              </div>
            </div>

            <div
              v-if="selectedView === 'targets' && targetOutputs(selectedJob).length"
              class="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
              <div class="flex items-center justify-between gap-3">
                <div>
                  <p class="text-sm font-semibold text-slate-800">Individual Target Masks</p>
                  <p class="text-xs text-slate-500">Select the masks that are relevant for the final review.</p>
                </div>
                <span class="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-600">
                  {{ selectedTargetCount(selectedJob) }} selected
                </span>
              </div>
              <div class="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <button
                  v-for="(targetOutput, index) in targetOutputs(selectedJob)"
                  :key="`target-${targetOutput.target_number || index}`"
                  type="button"
                  :class="[
                    'overflow-hidden rounded-lg border bg-white text-left transition hover:border-cyan-600',
                    selectedTargetIndex === index ? 'border-cyan-700 ring-2 ring-cyan-100' : 'border-slate-200'
                  ]"
                  @click="selectedTargetIndex = index"
                >
                  <img
                    v-if="targetSegmentedImage(targetOutput)"
                    :src="targetSegmentedImage(targetOutput)"
                    alt=""
                    class="h-20 w-full bg-slate-950 object-contain"
                  />
                  <div v-else class="flex h-20 items-center justify-center bg-slate-100 text-xs text-slate-500">
                    No image
                  </div>
                  <div class="space-y-2 p-2">
                    <div class="flex items-start justify-between gap-2">
                      <div>
                        <p class="text-xs font-semibold text-slate-800">
                          Target {{ targetOutput.target_number || index + 1 }}
                        </p>
                        <p class="mt-0.5 text-[11px] text-slate-500">
                          Pass {{ targetOutput.pass_number || 1 }}
                          <span v-if="targetOutput.score !== null && targetOutput.score !== undefined">
                            · {{ Number(targetOutput.score).toFixed(2) }}
                          </span>
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        class="mt-0.5 h-4 w-4 rounded border-slate-300 text-cyan-700 focus:ring-cyan-600"
                        :checked="targetSelected(selectedJob, targetOutput)"
                        @click.stop="toggleTargetSelection(selectedJob, targetOutput)"
                      />
                    </div>
                  </div>
                </button>
              </div>
            </div>

            <div
              v-if="selectedView === 'passes' && passOutputs(selectedJob).length"
              class="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
              <div class="flex items-center justify-between gap-3">
                <div>
                  <p class="text-sm font-semibold text-slate-800">Segmented Output By Pass</p>
                  <p class="text-xs text-slate-500">Each pass shows the segmented image after that pass has been combined and cleaned.</p>
                </div>
                <span class="rounded-md bg-white px-2 py-1 text-xs font-semibold text-slate-600">
                  {{ selectedPassIndex + 1 }} of {{ passOutputs(selectedJob).length }}
                </span>
              </div>
              <div class="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <button
                  v-for="(passOutput, index) in passOutputs(selectedJob)"
                  :key="`pass-${passOutput.pass_number || index}`"
                  type="button"
                  :class="[
                    'overflow-hidden rounded-lg border bg-white text-left transition hover:border-cyan-600',
                    selectedPassIndex === index ? 'border-cyan-700 ring-2 ring-cyan-100' : 'border-slate-200'
                  ]"
                  @click="selectedPassIndex = index"
                >
                  <img
                    v-if="passSegmentedImage(passOutput)"
                    :src="passSegmentedImage(passOutput)"
                    alt=""
                    class="h-20 w-full bg-slate-950 object-contain"
                  />
                  <div v-else class="flex h-20 items-center justify-center bg-slate-100 text-xs text-slate-500">
                    No image
                  </div>
                  <div class="p-2">
                    <p class="text-xs font-semibold text-slate-800">
                      Pass {{ passOutput.pass_number || index + 1 }}
                    </p>
                    <p class="mt-0.5 text-[11px] text-slate-500">
                      {{ passOutput.prompt_count ?? 0 }} prompt targets
                    </p>
                  </div>
                </button>
              </div>
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
                  <p class="text-sm font-semibold">Resource Usage</p>
                  <p class="text-xs text-slate-400">CPU, RAM, GPU, and VRAM usage while the job runs</p>
                </div>
                <span class="rounded-full bg-slate-800 px-2 py-1 text-xs text-slate-300">
                  {{ latestResourceLog(selectedJob).length }} samples
                  <span v-if="selectedJob.resourceLogTotal">
                    of {{ selectedJob.resourceLogTotal }}
                  </span>
                </span>
              </div>
              <div v-if="resourceGraphLines(selectedJob).length" class="p-4">
                <div class="rounded-lg border border-slate-800 bg-slate-900 p-3">
                  <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="h-44 w-full">
                    <line
                      v-for="tick in [25, 50, 75]"
                      :key="`grid-${tick}`"
                      x1="0"
                      x2="100"
                      :y1="tick"
                      :y2="tick"
                      stroke="rgba(148, 163, 184, 0.18)"
                      stroke-width="0.6"
                    />
                    <polyline
                      v-for="line in resourceGraphLines(selectedJob)"
                      :key="line.id"
                      :points="line.points"
                      fill="none"
                      :stroke="line.color"
                      stroke-width="2"
                      vector-effect="non-scaling-stroke"
                    />
                  </svg>
                </div>
                <div class="mt-3 grid gap-2 sm:grid-cols-2">
                  <div
                    v-for="line in resourceGraphLines(selectedJob)"
                    :key="`legend-${line.id}`"
                    class="flex items-center gap-2 rounded-md bg-slate-900 px-3 py-2 text-xs"
                  >
                    <span class="h-2.5 w-2.5 rounded-full" :style="{ backgroundColor: line.color }"></span>
                    <span class="font-semibold text-slate-200">{{ line.label }}</span>
                    <span class="ml-auto text-slate-400">{{ line.latestLabel }}</span>
                  </div>
                </div>
              </div>
              <div v-else class="px-4 py-5 text-sm text-slate-400">
                Resource graph will appear once this job starts.
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
                type="button"
                class="rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-2 text-sm font-semibold text-cyan-800 transition hover:bg-cyan-100 disabled:cursor-not-allowed disabled:opacity-50"
                :disabled="!selectedJob.result?.output_id || isQueueRunning"
                @click="refineSavedOutput(selectedJob)"
              >
                Refine This Output
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
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import axios from 'axios'

const MAX_FILE_SIZE = 25 * 1024 * 1024
const POLL_INTERVAL_MS = 2000
const resultViews = [
  { id: 'before', label: 'Before' },
  { id: 'overlay', label: 'Mask Overlay' },
  { id: 'mask', label: 'Mask Only' },
  { id: 'segmented', label: 'Segmented Image' },
  { id: 'layers', label: 'Prompt Layers' },
  { id: 'targets', label: 'Target Masks' },
  { id: 'passes', label: 'Pass Outputs' }
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
const cloudProviderPresets = [
  { id: 'openai', label: 'OpenAI', model: 'openai/gpt-4o' },
  { id: 'anthropic', label: 'Anthropic', model: 'anthropic/claude-3-5-sonnet-latest' },
  { id: 'google', label: 'Google Gemini', model: 'gemini/gemini-1.5-pro' },
  { id: 'openai-compatible', label: 'OpenAI-compatible', model: 'openai/model-name' },
  { id: 'custom', label: 'Custom LiteLLM ID', model: '' }
]

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
    const selectedPassIndex = ref(0)
    const selectedTargetIndex = ref(0)
    const selectedLayerIndex = ref(0)
    const currentJobId = ref(null)
    const isQueueRunning = ref(false)
    const isDragging = ref(false)
    const error = ref(null)
    const modelStatus = ref({ loaded: false, model: 'Think2Seg-RS-7B' })
    const configPanelOpen = ref(true)
    const queuePanelOpen = ref(true)
    const historyPanelOpen = ref(false)
    const outputHistory = ref([])
    const historyLoading = ref(false)
    const segmentationSettings = ref({ ...defaultSegmentationSettings })
    const promptProviderSettings = ref({
      promptProvider: 'local',
      providerPreset: 'openai',
      litellmModel: 'openai/gpt-4o',
      litellmApiKey: '',
      litellmApiBase: '',
      hasBackendApiKey: false,
      message: ''
    })
    const runtimeSettings = ref({
      modelGpuMemoryUtilization: 0.95,
      modelMaxMemoryGb: 0,
      modelLoaded: false,
      message: ''
    })
    let postprocessTimer = null
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
        segmented: 'Segmented image cutout',
        layers: 'Segmented image from the selected prompt layer',
        targets: 'Segmented image from the selected target mask',
        passes: 'Segmented image from the selected refinement pass'
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
      const providerReady = runMode.value !== 'prompt' ||
        promptProviderSettings.value.promptProvider !== 'cloud' ||
        Boolean(promptProviderSettings.value.litellmModel.trim())
      return !isQueueRunning.value && runnableCount.value > 0 && inputReady && providerReady
    })

    const checkModelStatus = async () => {
      try {
        const response = await axios.get('/api/health')
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

    const loadRuntimeSettings = async () => {
      try {
        const response = await axios.get('/api/runtime-settings')
        runtimeSettings.value = {
          modelGpuMemoryUtilization: Number(response.data.model_gpu_memory_utilization ?? 0.95),
          modelMaxMemoryGb: Number(response.data.model_max_memory_gb || 0),
          modelLoaded: Boolean(response.data.model_loaded),
          message: response.data.applies_to_loaded_model === false
            ? 'Model is loaded; changes apply next load.'
            : 'Applies before the first model load.'
        }
      } catch (err) {
        runtimeSettings.value.message = `Could not load runtime settings: ${getErrorMessage(err)}`
      }
    }

    const loadPromptProviderSettings = async () => {
      try {
        const response = await axios.get('/api/prompt-provider-settings')
        const data = response.data || {}
        promptProviderSettings.value = {
          ...promptProviderSettings.value,
          promptProvider: data.prompt_provider || 'local',
          litellmModel: data.litellm_model || promptProviderSettings.value.litellmModel,
          litellmApiBase: data.litellm_api_base || '',
          litellmApiKey: '',
          hasBackendApiKey: Boolean(data.has_litellm_api_key),
          message: data.has_litellm_api_key ? 'Backend BYOK key is configured.' : ''
        }
      } catch (err) {
        promptProviderSettings.value.message = `Could not load provider settings: ${getErrorMessage(err)}`
      }
    }

    const normalizePromptProviderSettings = (settings = {}) => {
      const provider = settings.promptProvider === 'cloud' ? 'cloud' : 'local'
      return {
        promptProvider: provider,
        providerPreset: settings.providerPreset || 'custom',
        litellmModel: String(settings.litellmModel || '').trim(),
        litellmApiKey: String(settings.litellmApiKey || '').trim(),
        litellmApiBase: String(settings.litellmApiBase || '').trim(),
        hasBackendApiKey: Boolean(settings.hasBackendApiKey),
        message: settings.message || ''
      }
    }

    const promptProviderPayload = (settings = promptProviderSettings.value) => {
      const normalized = normalizePromptProviderSettings(settings)
      const payload = {
        prompt_provider: normalized.promptProvider,
        litellm_model: normalized.litellmModel,
        litellm_api_base: normalized.litellmApiBase
      }
      if (normalized.litellmApiKey) {
        payload.litellm_api_key = normalized.litellmApiKey
      }
      return payload
    }

    const appendPromptProviderSettings = (formData, settings = promptProviderSettings.value) => {
      const payload = promptProviderPayload(settings)
      formData.append('prompt_provider', payload.prompt_provider)
      if (payload.prompt_provider === 'cloud') {
        formData.append('litellm_model', payload.litellm_model)
        if (payload.litellm_api_key) {
          formData.append('litellm_api_key', payload.litellm_api_key)
        }
        if (payload.litellm_api_base) {
          formData.append('litellm_api_base', payload.litellm_api_base)
        }
      }
    }

    const applyProviderPreset = () => {
      const preset = cloudProviderPresets.find((item) => item.id === promptProviderSettings.value.providerPreset)
      if (preset?.model) {
        promptProviderSettings.value.litellmModel = preset.model
      }
    }

    const applyPromptProviderSettings = async () => {
      try {
        const payload = promptProviderPayload()
        const response = await axios.post('/api/prompt-provider-settings', payload)
        promptProviderSettings.value = {
          ...promptProviderSettings.value,
          promptProvider: response.data.prompt_provider || payload.prompt_provider,
          litellmModel: response.data.litellm_model || payload.litellm_model,
          litellmApiBase: response.data.litellm_api_base || payload.litellm_api_base,
          litellmApiKey: '',
          hasBackendApiKey: Boolean(response.data.has_litellm_api_key),
          message: response.data.message || 'Prompt provider settings updated.'
        }
      } catch (err) {
        promptProviderSettings.value.message = `Could not apply provider settings: ${getErrorMessage(err)}`
      }
    }

    const applyRuntimeSettings = async () => {
      try {
        const response = await axios.post('/api/runtime-settings', {
          model_gpu_memory_utilization: runtimeSettings.value.modelGpuMemoryUtilization,
          model_max_memory_gb: Number(runtimeSettings.value.modelMaxMemoryGb || 0)
        })
        runtimeSettings.value.modelLoaded = Boolean(response.data.model_loaded)
        runtimeSettings.value.message = response.data.message || 'Runtime settings updated.'
      } catch (err) {
        runtimeSettings.value.message = `Could not apply settings: ${getErrorMessage(err)}`
      }
    }

    const loadOutputHistory = async () => {
      historyLoading.value = true
      try {
        const response = await axios.get('/api/outputs')
        outputHistory.value = response.data.outputs || []
      } catch (err) {
        error.value = `Could not load saved outputs: ${getErrorMessage(err)}`
      } finally {
        historyLoading.value = false
      }
    }

    const openSavedOutput = (output) => {
      const existing = queue.value.find((job) => job.id === `saved-${output.id}`)
      if (existing) {
        selectedJobId.value = existing.id
        selectedView.value = existing.result?.target_outputs?.length ? 'targets' : 'segmented'
        return
      }

      const result = output.result || {}
      const targetNumbers = (result.target_outputs || [])
        .map((target, index) => target.target_number || index + 1)
      const job = {
        id: `saved-${output.id}`,
        serverJobId: output.id,
        file: null,
        name: output.filename || output.id,
        size: 0,
        previewUrl: result.original_image_url || output.original_image_url || '',
        previewAvailable: Boolean(result.original_image_url || output.original_image_url),
        dimensions: Array.isArray(result.original_size) ? `${result.original_size[0]} x ${result.original_size[1]}px` : 'Saved output',
        prompt: output.prompt || result.prompt || '',
        prompts: output.prompts || result.prompts || [output.prompt || result.prompt || ''].filter(Boolean),
        cropId: '',
        jobType: 'prompt',
        status: 'succeeded',
        progress: 100,
        uploadProgress: 100,
        stage: 'Loaded from history',
        message: 'Saved segmentation output',
        elapsedMs: 0,
        startedAt: null,
        completedAt: Date.now(),
        resourceLog: [],
        resourceLogTotal: 0,
        resourceSnapshot: null,
        settings: optionsToSegmentationSettings(result.options),
        promptProvider: result.prompt_provider || null,
        selectedTargetNumbers: targetNumbers,
        result,
        adjustedResult: null,
        segmentationGeneration: output.segmentation_generation || 1,
        error: null,
        isSavedOutput: true
      }
      queue.value.unshift(job)
      selectedJobId.value = job.id
      selectedTargetIndex.value = 0
      selectedPassIndex.value = 0
      selectedView.value = targetNumbers.length ? 'targets' : 'segmented'
    }

    const parseLayerPrompts = (value) => {
      return String(value || '')
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean)
    }

    const appendPromptLayer = (value) => {
      const layers = parseLayerPrompts(prompt.value)
      if (!layers.some((item) => item.toLowerCase() === value.toLowerCase())) {
        layers.push(value)
      }
      prompt.value = layers.join('\n')
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
        prompts: [],
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
        promptProvider: null,
        selectedTargetNumbers: [],
        adjustedResult: null,
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

      const queuePrompts = parseLayerPrompts(prompt.value)
      const queuePrompt = queuePrompts.join(' | ')
      const queueCropId = cropId.value.trim()
      const queueMode = runMode.value
      const queueSettings = normalizeSegmentationSettings(segmentationSettings.value)
      const queuePromptProvider = normalizePromptProviderSettings(promptProviderSettings.value)
      if (queueMode === 'prompt' && !queuePrompts.length) {
        error.value = 'Enter at least one layer prompt before starting the queue.'
        return
      }
      if (queueMode === 'prompt' && queuePromptProvider.promptProvider === 'cloud' && !queuePromptProvider.litellmModel) {
        error.value = 'Enter a LiteLLM model ID before starting a cloud prompt job.'
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
          await processJob(nextJob, queuePrompt, queueMode, queueCropId, queueSettings, queuePrompts, queuePromptProvider)
        }
      } finally {
        isQueueRunning.value = false
        currentJobId.value = null
      }
    }

    const processJob = async (job, queuePrompt, queueMode, queueCropId, queueSettings, queuePrompts = [], queuePromptProvider = null) => {
      currentJobId.value = job.id
      selectedJobId.value = job.id
      Object.assign(job, {
        prompt: queueMode === 'aether' ? '12-class AETHER inference' : queuePrompt,
        prompts: queueMode === 'aether' ? [] : queuePrompts,
        cropId: queueMode === 'aether' ? queueCropId : '',
        jobType: queueMode,
        settings: queueMode === 'prompt' ? { ...queueSettings } : null,
        promptProvider: queueMode === 'prompt' ? normalizePromptProviderSettings(queuePromptProvider || promptProviderSettings.value) : null,
        adjustedResult: null,
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
          if (queuePrompts.length > 1) {
            formData.append('prompts_json', JSON.stringify(queuePrompts))
          }
          appendSegmentationSettings(formData, queueSettings)
          appendPromptProviderSettings(formData, queuePromptProvider)
        }

        const endpoint = queueMode === 'aether' ? '/api/aether/jobs' : '/api/segment/jobs'
        let response = null
        for (let attempt = 1; attempt <= 3; attempt += 1) {
          try {
            response = await axios.post(endpoint, formData, {
              headers: { 'Content-Type': 'multipart/form-data' },
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
        let response = null
        try {
          response = await axios.get(`/api/${jobPath}/${job.serverJobId}`)
        } catch (err) {
          if (!isTransientPollError(err)) {
            throw err
          }
          job.status = job.status === 'uploading' ? 'running' : job.status
          job.stage = 'Waiting for backend response'
          job.message = getTransientPollMessage(err)
          await wait(POLL_INTERVAL_MS)
          continue
        }
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
      job.prompts = serverJob.prompts || job.prompts || []
      job.promptProvider = serverJob.prompt_provider || serverJob.result?.prompt_provider || job.promptProvider || null
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
        if (!(job.selectedTargetNumbers || []).length && targetOutputs(job).length) {
          job.selectedTargetNumbers = targetOutputs(job).map((target, index) => targetNumber(target, index))
        }
        loadOutputHistory()
        schedulePostprocess(job)
        if (selectedJobId.value === job.id) {
          selectedView.value = targetOutputs(job).length ? 'targets' : 'overlay'
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
        if (job.prompts?.length) {
          prompt.value = job.prompts.join('\n')
        }
        if (job.settings) {
          segmentationSettings.value = normalizeSegmentationSettings(job.settings)
        }
        if (job.promptProvider) {
          promptProviderSettings.value = normalizePromptProviderSettings({
            ...promptProviderSettings.value,
            promptProvider: job.promptProvider.prompt_provider || job.promptProvider.promptProvider,
            litellmModel: job.promptProvider.litellm_model || job.promptProvider.litellmModel,
            litellmApiBase: job.promptProvider.litellm_api_base || job.promptProvider.litellmApiBase,
            hasBackendApiKey: job.promptProvider.has_litellm_api_key || promptProviderSettings.value.hasBackendApiKey
          })
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
        selectedTargetNumbers: [],
        adjustedResult: null,
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
            selectedTargetNumbers: [],
            adjustedResult: null,
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

    const segmentationSettingsToApiOptions = (settings) => {
      const normalized = normalizeSegmentationSettings(settings)
      return {
        sam_mask_threshold: normalized.samMaskThreshold,
        sam_multimask_output: normalized.samMultimaskOutput,
        mask_min_area: normalized.maskMinArea,
        mask_cleanup_px: normalized.maskCleanupPx,
        mask_expand_px: normalized.maskExpandPx,
        refinement_passes: normalized.refinementPasses,
        refinement_mode: normalized.refinementMode
      }
    }

    const schedulePostprocess = (job = selectedJob.value) => {
      if (postprocessTimer) {
        window.clearTimeout(postprocessTimer)
      }
      if (!job?.result?.output_id || job.status !== 'succeeded') return
      postprocessTimer = window.setTimeout(() => {
        postprocessJob(job)
      }, 450)
    }

    const postprocessJob = async (job) => {
      if (!job?.result?.output_id) return
      try {
        const response = await axios.post(`/api/outputs/${job.result.output_id}/postprocess`, {
          options: segmentationSettingsToApiOptions(segmentationSettings.value),
          selected_target_numbers: job.selectedTargetNumbers || null
        })
        job.adjustedResult = response.data
      } catch (err) {
        error.value = `${job.name}: live post-process failed: ${getErrorMessage(err)}`
      }
    }

    const refineSavedOutput = async (job) => {
      if (!job?.result?.output_id || isQueueRunning.value) return
      try {
        const prompts = job.prompts?.length
          ? job.prompts
          : parseLayerPrompts(job.prompt || prompt.value)
        const response = await axios.post(`/api/outputs/${job.result.output_id}/refine/jobs`, {
          prompts,
          options: segmentationSettingsToApiOptions(segmentationSettings.value),
          prompt_provider: promptProviderPayload()
        })
        const refineJob = {
          id: `refine-${response.data.id}`,
          serverJobId: response.data.id,
          file: null,
          name: `Refine ${job.name}`,
          size: 0,
          previewUrl: selectedViewImage(job),
          previewAvailable: Boolean(selectedViewImage(job)),
          dimensions: job.dimensions,
          prompt: response.data.prompt || prompts.join(' | '),
          prompts,
          cropId: '',
          jobType: 'prompt',
          status: response.data.status || 'queued',
          progress: Number(response.data.progress || 0),
          uploadProgress: 100,
          stage: response.data.stage || 'Queued',
          message: response.data.message || 'Waiting to refine saved output',
          elapsedMs: 0,
          startedAt: Date.now(),
          completedAt: null,
          resourceLog: response.data.resource_log || [],
          resourceLogTotal: Number(response.data.resource_log_total || 0),
          resourceSnapshot: response.data.resource_snapshot || null,
          settings: normalizeSegmentationSettings(segmentationSettings.value),
          promptProvider: response.data.prompt_provider || normalizePromptProviderSettings(promptProviderSettings.value),
          selectedTargetNumbers: [],
          adjustedResult: null,
          result: null,
          error: null
        }
        queue.value.unshift(refineJob)
        selectedJobId.value = refineJob.id
        currentJobId.value = refineJob.id
        isQueueRunning.value = true
        await pollServerJob(refineJob)
      } catch (err) {
        error.value = `${job.name}: refinement failed: ${getErrorMessage(err)}`
      } finally {
        isQueueRunning.value = false
        currentJobId.value = null
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
      const result = displayResult(job)
      if (viewId === 'overlay') return Boolean(result?.visualization_base64 || result?.visualization_url)
      if (viewId === 'mask') return Boolean(result?.mask_base64 || result?.mask_url)
      if (viewId === 'segmented') return Boolean(result?.segmented_image_base64 || result?.segmented_image_url)
      if (viewId === 'layers') return layerOutputs(job).length > 0
      if (viewId === 'targets') return targetOutputs(job).length > 0
      if (viewId === 'passes') return passOutputs(job).length > 0
      return false
    }

    const imageSource = (base64Value, urlValue) => {
      if (base64Value) return `data:image/png;base64,${base64Value}`
      return urlValue || ''
    }

    const displayResult = (job) => {
      return job?.adjustedResult || job?.result || {}
    }

    const layerOutputs = (job) => {
      return Array.isArray(job?.result?.layers) ? job.result.layers : []
    }

    const selectedLayerOutput = (job) => {
      const outputs = layerOutputs(job)
      if (!outputs.length) return null
      const clampedIndex = Math.min(Math.max(selectedLayerIndex.value, 0), outputs.length - 1)
      if (clampedIndex !== selectedLayerIndex.value) {
        selectedLayerIndex.value = clampedIndex
      }
      return outputs[clampedIndex]
    }

    const layerSegmentedImage = (layerOutput) => {
      return imageSource(layerOutput?.segmented_image_base64, layerOutput?.segmented_image_url)
    }

    const targetOutputs = (job) => {
      return Array.isArray(job?.result?.target_outputs) ? job.result.target_outputs : []
    }

    const targetNumber = (targetOutput, index = 0) => {
      return targetOutput?.target_number || index + 1
    }

    const selectedTargetOutput = (job) => {
      const outputs = targetOutputs(job)
      if (!outputs.length) return null
      const clampedIndex = Math.min(Math.max(selectedTargetIndex.value, 0), outputs.length - 1)
      if (clampedIndex !== selectedTargetIndex.value) {
        selectedTargetIndex.value = clampedIndex
      }
      return outputs[clampedIndex]
    }

    const targetSegmentedImage = (targetOutput) => {
      return imageSource(targetOutput?.segmented_image_base64, targetOutput?.segmented_image_url)
    }

    const targetSelected = (job, targetOutput) => {
      const number = targetNumber(targetOutput)
      return (job?.selectedTargetNumbers || []).includes(number)
    }

    const selectedTargetCount = (job) => {
      return (job?.selectedTargetNumbers || []).length
    }

    const toggleTargetSelection = (job, targetOutput) => {
      if (!job) return
      const number = targetNumber(targetOutput)
      const current = new Set(job.selectedTargetNumbers || [])
      if (current.has(number)) {
        current.delete(number)
      } else {
        current.add(number)
      }
      job.selectedTargetNumbers = Array.from(current).sort((a, b) => a - b)
      schedulePostprocess(job)
    }

    const passOutputs = (job) => {
      return Array.isArray(job?.result?.pass_outputs) ? job.result.pass_outputs : []
    }

    const selectedPassOutput = (job) => {
      const outputs = passOutputs(job)
      if (!outputs.length) return null
      const clampedIndex = Math.min(Math.max(selectedPassIndex.value, 0), outputs.length - 1)
      if (clampedIndex !== selectedPassIndex.value) {
        selectedPassIndex.value = clampedIndex
      }
      return outputs[clampedIndex]
    }

    const passSegmentedImage = (passOutput) => {
      return imageSource(passOutput?.segmented_image_base64, passOutput?.segmented_image_url)
    }

    const selectedViewImage = (job) => {
      if (!job) return ''
      if (selectedView.value === 'before' && job.previewAvailable) {
        return job.previewUrl
      }
      const result = displayResult(job)
      if (selectedView.value === 'overlay') {
        return imageSource(result?.visualization_base64, result?.visualization_url)
      }
      if (selectedView.value === 'mask') {
        return imageSource(result?.mask_base64, result?.mask_url)
      }
      if (selectedView.value === 'segmented') {
        return imageSource(result?.segmented_image_base64, result?.segmented_image_url)
      }
      if (selectedView.value === 'layers') {
        const layerOutput = selectedLayerOutput(job)
        const layerImage = layerSegmentedImage(layerOutput)
        if (layerImage) return layerImage
      }
      if (selectedView.value === 'targets') {
        const targetOutput = selectedTargetOutput(job)
        const targetImage = targetSegmentedImage(targetOutput)
        if (targetImage) return targetImage
      }
      if (selectedView.value === 'passes') {
        const passOutput = selectedPassOutput(job)
        const passImage = passSegmentedImage(passOutput)
        if (passImage) return passImage
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
      const targetOutputCount = job?.result?.target_outputs?.length
      if (typeof targetOutputCount === 'number') return targetOutputCount
      const count = job?.result?.sam_prompts?.length
      if (typeof count === 'number') return count
      if (typeof job?.result?.segment_count === 'number') return job.result.segment_count
      if (job?.status === 'succeeded') return 0
      return 'Pending'
    }

    const latestResourceLog = (job) => {
      return (job?.resourceLog || []).slice(-RESOURCE_LOG_DISPLAY_LIMIT)
    }

    const resourceGraphLines = (job) => {
      const samples = latestResourceLog(job)
        .map((entry) => entry.snapshot)
        .filter(Boolean)
      if (!samples.length) return []

      const series = []
      const ramValues = samples.map((snapshot) => {
        const ram = snapshot.ram || {}
        if (!ram.total || ram.used === null || ram.used === undefined) return null
        return Math.max(0, Math.min(100, (ram.used / ram.total) * 100))
      })
      series.push({
        id: 'ram',
        label: 'RAM',
        color: '#22d3ee',
        values: ramValues,
        latestLabel: formatResourcePercent(samples.at(-1)?.ram?.used, samples.at(-1)?.ram?.total)
      })
      const cpuValues = samples.map((snapshot) => {
        const value = snapshot.cpu?.usage_percent
        return value === null || value === undefined ? null : Math.max(0, Math.min(100, Number(value)))
      })
      series.push({
        id: 'cpu',
        label: 'CPU',
        color: '#f97316',
        values: cpuValues,
        latestLabel: `${Math.round(cpuValues.at(-1) ?? 0)}%`
      })

      const maxGpuCount = Math.max(0, ...samples.map((snapshot) => (snapshot.gpus || []).length))
      const gpuColors = ['#a78bfa', '#34d399', '#fbbf24', '#fb7185']
      for (let index = 0; index < maxGpuCount; index += 1) {
        const values = samples.map((snapshot) => {
          const gpu = (snapshot.gpus || [])[index]
          if (!gpu?.total || gpu.used === null || gpu.used === undefined) return null
          return Math.max(0, Math.min(100, (gpu.used / gpu.total) * 100))
        })
        const latestGpu = (samples.at(-1)?.gpus || [])[index]
        series.push({
          id: `gpu-${index}`,
          label: `GPU ${index} VRAM`,
          color: gpuColors[index % gpuColors.length],
          values,
          latestLabel: formatResourcePercent(latestGpu?.used, latestGpu?.total)
        })
        const utilizationValues = samples.map((snapshot) => {
          const value = (snapshot.gpus || [])[index]?.utilization_percent
          return value === null || value === undefined ? null : Math.max(0, Math.min(100, Number(value)))
        })
        series.push({
          id: `gpu-${index}-util`,
          label: `GPU ${index} Compute`,
          color: gpuColors[(index + 1) % gpuColors.length],
          values: utilizationValues,
          latestLabel: `${Math.round(utilizationValues.at(-1) ?? 0)}%`
        })
      }

      return series
        .map((item) => ({
          ...item,
          points: graphPoints(item.values)
        }))
        .filter((item) => item.points)
    }

    const graphPoints = (values) => {
      const valid = values
        .map((value, index) => ({ value, index }))
        .filter((item) => item.value !== null && item.value !== undefined)
      if (!valid.length) return ''
      const denominator = Math.max(values.length - 1, 1)
      return valid
        .map((item) => {
          const x = values.length === 1 ? 100 : (item.index / denominator) * 100
          const y = 100 - item.value
          return `${x.toFixed(2)},${y.toFixed(2)}`
        })
        .join(' ')
    }

    const formatResourcePercent = (used, total) => {
      if (!total || used === null || used === undefined) return 'No data'
      return `${Math.round((used / total) * 100)}% · ${formatBytes(used)} / ${formatBytes(total)}`
    }

    const progressChecklist = (job) => {
      if (!job) return []
      const stage = String(job.stage || job.message || '').toLowerCase()
      const progress = Number(job.progress || 0)
      const failed = job.status === 'failed'
      const definitions = [
        { id: 'queued', label: 'Queued', threshold: 1, match: ['queued', 'waiting'] },
        { id: 'upload', label: 'Upload image', threshold: 2, match: ['upload'] },
        { id: 'validate', label: 'Validate image', threshold: 4, match: ['validating', 'reading uploaded'] },
        { id: 'models', label: 'Load models', threshold: 45, match: ['loading', 'models ready', 'qwen', 'sam2'] },
        { id: 'prepare', label: 'Prepare image', threshold: 50, match: ['preparing image'] },
        { id: 'prompts', label: 'Generate prompts', threshold: 58, match: ['generating spatial prompts'] },
        { id: 'parse', label: 'Parse target prompts', threshold: 64, match: ['parsing target prompts'] },
        { id: 'sam', label: 'Run SAM masks', threshold: 82, match: ['running sam', 'predicting target masks', 'predicted mask'] },
        { id: 'refine', label: 'Refine masks', threshold: 92, match: ['mask refined', 'combining masks'] },
        { id: 'render', label: 'Render outputs', threshold: 99, match: ['rendering', 'encoding result'] },
        { id: 'complete', label: 'Complete', threshold: 100, match: ['complete'] }
      ]
      const activeIndex = definitions.findIndex((item) => item.match.some((token) => stage.includes(token)))
      return definitions.map((item, index) => {
        let state = 'pending'
        if (failed && index >= activeIndex && activeIndex >= 0) {
          state = 'blocked'
        } else if (job.status === 'succeeded' || progress >= item.threshold) {
          state = 'done'
        } else if (index === activeIndex) {
          state = 'running'
        }
        return { ...item, state }
      })
    }

    const checklistItemClass = (item) => {
      if (item.state === 'done') return 'border-emerald-200 bg-emerald-50 text-emerald-900'
      if (item.state === 'running') return 'border-cyan-200 bg-cyan-50 text-cyan-900'
      if (item.state === 'blocked') return 'border-rose-200 bg-rose-50 text-rose-900'
      return 'border-slate-200 bg-white text-slate-500'
    }

    const checklistDotClass = (item) => {
      if (item.state === 'done') return 'bg-emerald-600'
      if (item.state === 'running') return 'bg-cyan-600'
      if (item.state === 'blocked') return 'bg-rose-600'
      return 'bg-slate-300'
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

    const isTransientPollError = (err) => {
      return isTransientGatewayError(err)
    }

    const getTransientPollMessage = (err) => {
      return `Gateway returned ${err.response?.status}; keeping the job alive and retrying.`
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
      loadRuntimeSettings()
      loadPromptProviderSettings()
      loadOutputHistory()
      elapsedTimer = window.setInterval(() => {
        const now = Date.now()
        queue.value.forEach((job) => {
          if (job.startedAt && !job.completedAt && !['succeeded', 'failed'].includes(job.status)) {
            job.elapsedMs = now - job.startedAt
          }
        })
      }, 1000)
    })

    watch(segmentationSettings, () => {
      schedulePostprocess(selectedJob.value)
    }, { deep: true })

    onUnmounted(() => {
      if (elapsedTimer) {
        window.clearInterval(elapsedTimer)
      }
      if (postprocessTimer) {
        window.clearTimeout(postprocessTimer)
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
      selectedPassIndex,
      selectedLayerIndex,
      selectedViewAlt,
      selectedJob,
      currentJob,
      isQueueRunning,
      isDragging,
      error,
      modelStatus,
      configPanelOpen,
      queuePanelOpen,
      historyPanelOpen,
      outputHistory,
      historyLoading,
      segmentationSettings,
      promptProviderSettings,
      cloudProviderPresets,
      runtimeSettings,
      selectedTargetIndex,
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
      loadOutputHistory,
      openSavedOutput,
      appendPromptLayer,
      applyProviderPreset,
      applyPromptProviderSettings,
      applyRuntimeSettings,
      resetSegmentationSettings,
      retryJob,
      retryFailed,
      removeJob,
      clearFinished,
      refineSavedOutput,
      downloadJob,
      resultViewAvailable,
      layerOutputs,
      layerSegmentedImage,
      targetOutputs,
      targetSegmentedImage,
      targetSelected,
      selectedTargetCount,
      toggleTargetSelection,
      passOutputs,
      passSegmentedImage,
      selectedViewImage,
      isJobBusy,
      statusText,
      statusBadgeClass,
      progressBarClass,
      queuePosition,
      targetCount,
      latestResourceLog,
      resourceGraphLines,
      progressChecklist,
      checklistItemClass,
      checklistDotClass,
      formatDuration,
      formatBytes,
      formatLogTime,
      formatResourceSnapshot
    }
  }
}
</script>
