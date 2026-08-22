<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getJSON, postJSON } from '../services/api'

const analytics = ref<any>({ origins: [], destinations: [], carriers: [] })
const form = ref({
  origin_warehouse: 'Warehouse_MIA',
  destination: 'Detroit',
  shipment_date: new Date().toISOString().slice(0, 10),
  weight_kg: 30,
  distance_miles: 1200,
  objective: 'balanced'
})
const result = ref<any>(null)
const loading = ref(false)
const error = ref('')

onMounted(async () => {
  try {
    analytics.value = await getJSON('/carriers/analytics')
    if (analytics.value.origins?.length && !analytics.value.origins.includes(form.value.origin_warehouse)) {
      form.value.origin_warehouse = analytics.value.origins[0]
    }
    if (analytics.value.destinations?.length && !analytics.value.destinations.includes(form.value.destination)) {
      form.value.destination = analytics.value.destinations[0]
    }
  } catch {}
})

async function recommend() {
  loading.value = true
  error.value = ''
  try {
    result.value = await postJSON('/carriers/recommend', form.value)
  } catch (e: any) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="panel span-2">
    <div class="panel-title-row">
      <div>
        <span class="section-tag">DECISION ENGINE</span>
        <h2>Carrier Recommendation</h2>
      </div>
      <span class="model-chip">Cost + transit + historical reliability</span>
    </div>

    <div class="form-grid">
      <label>Origin warehouse
        <select v-model="form.origin_warehouse">
          <option v-for="x in analytics.origins" :key="x" :value="x">{{ x }}</option>
        </select>
      </label>
      <label>Destination
        <select v-model="form.destination">
          <option v-for="x in analytics.destinations" :key="x" :value="x">{{ x }}</option>
        </select>
      </label>
      <label>Shipment date<input v-model="form.shipment_date" type="date" /></label>
      <label>Weight (kg)<input v-model.number="form.weight_kg" type="number" min="0.1" step="0.1" /></label>
      <label>Distance (miles)<input v-model.number="form.distance_miles" type="number" min="1" /></label>
      <label>Objective
        <select v-model="form.objective">
          <option value="cheapest">Cheapest</option>
          <option value="fastest">Fastest</option>
          <option value="most_reliable">Most Reliable</option>
          <option value="balanced">Balanced</option>
        </select>
      </label>
    </div>

    <button @click="recommend" :disabled="loading">{{ loading ? 'Ranking carriers…' : 'Rank Carriers' }}</button>
    <p v-if="error" class="error-text">{{ error }}</p>

    <div v-if="result" class="decision-summary">
      <b>{{ result.objective.replaceAll('_', ' ') }}</b>
      <span>{{ result.decision_method }}</span>
      <small>{{ result.reliability_note }}</small>
    </div>

    <div v-if="result?.recommendations?.length" class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Carrier</th><th>Pred. Cost</th><th>Pred. Transit</th><th>Hist. Reliability</th><th>Lane Rows</th><th>Score</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in result.recommendations" :key="r.carrier" :class="{ recommended: r.recommended }">
            <td>#{{ r.rank }}</td>
            <td><b>{{ r.carrier }}</b><span v-if="r.recommended" class="best-badge">Recommended</span></td>
            <td>${{ r.predicted_cost_usd?.toFixed(2) }}</td>
            <td>{{ r.predicted_transit_days?.toFixed(2) }} d</td>
            <td>{{ (r.historical_reliability * 100).toFixed(1) }}%</td>
            <td>{{ r.lane_history_rows }}</td>
            <td>{{ r.decision_score.toFixed(3) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
