<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getJSON, postJSON } from './services/api'
import ChatPanel from './components/ChatPanel.vue'
import TrackingPanel from './components/TrackingPanel.vue'
import CarrierDecisionPanel from './components/CarrierDecisionPanel.vue'
import AnomalyPanel from './components/AnomalyPanel.vue'
import FiguresGallery from './components/FiguresGallery.vue'

const health = ref<any>({})
const summary = ref<any>({})
const models = ref<any>({ models: {} })
const carrierAnalytics = ref<any>({})
const training = ref(false)
const trainError = ref('')

const deployedCount = computed(() => Object.values(models.value?.models || {}).filter((x: any) => x?.deployment_recommended).length)
const sourceCount = computed(() => Object.keys(models.value?.sources || summary.value?.sources || {}).length)

async function refresh() {
  health.value = await getJSON('/health')
  if (health.value.data_ready) {
    summary.value = await getJSON('/analytics/summary')
    models.value = await getJSON('/models/metadata')
    if (health.value.us_performance_ready) {
      carrierAnalytics.value = await getJSON('/carriers/analytics')
    }
  }
}

async function train() {
  training.value = true
  trainError.value = ''
  try {
    await postJSON('/train', {})
    await refresh()
  } catch (e: any) {
    trainError.value = e?.message || String(e)
  } finally {
    training.value = false
  }
}

onMounted(refresh)
</script>

<template>
  <main>
    <header>
      <div>
        <span class="eyebrow">LOGISTICS AI INTELLIGENCE PLATFORM · V2.2</span>
        <h1>Predictive Intelligence + Grounded Logistics Decisioning</h1>
        <p>Four-source architecture with governed model deployment, carrier optimization, anomaly detection, and retrieval-grounded Ask Logistics.</p>
      </div>
      <div class="status">
        <span :class="['dot', health.data_ready ? 'ok' : 'warn']"></span>{{ health.data_ready ? 'Models & knowledge ready' : 'Training required' }}
        <small>{{ health.training_architecture }} · mode: {{ health.mode }}</small>
      </div>
    </header>

    <section class="architecture-strip">
      <div><b>Tracking</b><span>Delay Risk · Delay Hours</span></div>
      <div><b>US Performance</b><span>Cost · Transit · Carrier Ranking</span></div>
      <div><b>Knowledge / RAG</b><span>~126K records · Grounded answers</span></div>
      <div><b>Decision Engine</b><span>Cheapest · Fastest · Reliable · Balanced</span></div>
    </section>

    <section class="toolbar">
      <button @click="train" :disabled="training">{{ training ? 'Training V2.2…' : 'Train / Rebuild V2.2' }}</button>
      <span class="muted">Only models passing deployment gates are exposed by prediction endpoints.</span>
    </section>
    <p v-if="trainError" class="error-text">{{ trainError }}</p>

    <section v-if="health.data_ready" class="kpis">
      <article><b>{{ summary.rows?.toLocaleString() }}</b><span>Unified records</span></article>
      <article><b>{{ sourceCount || '—' }}</b><span>Data sources</span></article>
      <article><b>{{ deployedCount }}</b><span>Deployment-approved models</span></article>
      <article><b>{{ carrierAnalytics.carrier_count ?? summary.carriers ?? '—' }}</b><span>US carriers</span></article>
    </section>

    <div v-if="health.data_ready" class="grid">
      <TrackingPanel />
      <ChatPanel />
      <CarrierDecisionPanel />
      <AnomalyPanel />
      <section class="panel governance-panel">
        <span class="section-tag">MODEL GOVERNANCE</span>
        <h2>Production Gate</h2>
        <p class="muted">Weak operations and synthetic-risk classifiers remain in evaluation metadata but are intentionally blocked from production prediction.</p>
        <div class="governance-list">
          <div v-for="(m, name) in models.models" :key="name">
            <span>{{ String(name).replaceAll('_', ' ') }}</span>
            <b :class="m.deployment_recommended ? 'pass' : 'fail'">{{ m.deployment_recommended ? 'DEPLOY' : 'RESEARCH ONLY' }}</b>
          </div>
        </div>
      </section>
    </div>

    <FiguresGallery v-if="health.data_ready" />
  </main>
</template>
