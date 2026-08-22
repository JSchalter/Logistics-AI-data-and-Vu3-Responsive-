<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getJSON } from '../services/api'

const data = ref<any>(null)
const error = ref('')

async function refresh() {
  error.value = ''
  try { data.value = await getJSON('/us-performance/anomalies?limit=12') }
  catch (e: any) { error.value = e?.message || String(e) }
}

onMounted(refresh)
</script>

<template>
  <section class="panel">
    <div class="panel-title-row">
      <div>
        <span class="section-tag">DATA QUALITY</span>
        <h2>Anomaly Monitor</h2>
      </div>
      <button class="secondary" @click="refresh">Refresh</button>
    </div>
    <p class="muted">Rule-based quality flags from the US shipment dataset; these are not fraud determinations.</p>
    <p v-if="error" class="error-text">{{ error }}</p>
    <div v-if="data" class="anomaly-stats">
      <article><b>{{ data.total_flags }}</b><span>Total flags</span></article>
      <article><b>${{ data.extreme_cost_threshold_usd?.toFixed(0) }}</b><span>Extreme-cost threshold</span></article>
    </div>
    <div v-if="data" class="anomaly-list">
      <div v-for="a in data.items" :key="a.shipment_id + a.anomaly_type" class="anomaly-row">
        <span :class="['severity', a.severity]">{{ a.severity }}</span>
        <div>
          <b>{{ a.shipment_id }} · {{ a.anomaly_type.replaceAll('_', ' ') }}</b>
          <p>{{ a.reason }}</p>
          <small>{{ a.carrier }} · {{ a.origin }} → {{ a.destination }}</small>
        </div>
      </div>
    </div>
  </section>
</template>
