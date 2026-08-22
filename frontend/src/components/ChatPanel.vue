<script setup lang="ts">
import { ref } from 'vue'
import { postJSON } from '../services/api'

const q = ref('Compare carrier cost, transit time, and delay patterns for the most relevant shipments.')
const loading = ref(false)
const result = ref<any>(null)
const error = ref('')

async function ask() {
  loading.value = true
  error.value = ''
  try { result.value = await postJSON('/ask', { question: q.value, top_k: 10 }) }
  catch (e: any) { error.value = e?.message || String(e) }
  finally { loading.value = false }
}
</script>

<template>
  <section class="panel">
    <div class="panel-title-row">
      <div>
        <span class="section-tag">KNOWLEDGE / RAG</span>
        <h2>Ask Logistics AI</h2>
      </div>
      <span class="model-chip">Grounded local AI</span>
    </div>
    <p class="muted">Search records with local semantic retrieval and get a grounded Gemma explanation. Predictive results remain from the validated models.</p>
    <div class="askrow"><input v-model="q" @keyup.enter="ask" /><button @click="ask" :disabled="loading">{{ loading ? 'Analyzing…' : 'Ask' }}</button></div>
    <p v-if="error" class="error-text">{{ error }}</p>
    <div v-if="result" class="answer">
      <p>{{ result.answer }}</p>
      <small>Grounding: {{ result.grounding }} · Retrieval: {{ result.retrieval_engine }} · Model: {{ result.answer_model || 'deterministic fallback' }} · Live conditions used: {{ result.live_conditions_used }}</small>
      <details>
        <summary>Evidence rows ({{ result.evidence.length }})</summary>
        <div v-for="e in result.evidence" :key="e.record_id" class="evidence">
          <b>{{ e.route_id || e.shipment_id || e.record_id }}</b><span> similarity {{ e.score.toFixed(3) }}</span>
          <div>{{ e.source_dataset }}</div>
          <div>
            carrier={{ e.carrier ?? 'n/a' }} · status={{ e.shipment_status ?? 'n/a' }} · transit={{ e.transit_days ?? 'n/a' }} d · delay={{ e.delay_hours ?? 'n/a' }} h · cost={{ e.shipping_cost_usd ?? 'n/a' }}
          </div>
        </div>
      </details>
    </div>
  </section>
</template>
