<script setup lang="ts">
import { ref } from 'vue'
import { postJSON } from '../services/api'

const form = ref({
  distance_km: 900,
  minimum_km_per_day: 350,
  planned_transit_hours: 48,
  booking_to_start_hours: 12,
  vehicle_type: '40 FT 3XL Trailer 35MT',
  origin_code: 'MIA',
  destination_code: 'DTW',
  market_regular: 'Regular',
  material_shipped: 'General Freight'
})
const result = ref<any>(null)
const loading = ref(false)
const error = ref('')

async function predict() {
  loading.value = true
  error.value = ''
  try {
    result.value = await postJSON('/tracking/predict', form.value)
  } catch (e: any) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="panel">
    <div class="panel-title-row">
      <div>
        <span class="section-tag">TRACKING INTELLIGENCE</span>
        <h2>Delay Risk & Delay Hours</h2>
      </div>
      <span class="model-chip good">Deployment-gated</span>
    </div>

    <p class="muted">Uses the trained transportation-tracking models only. No live traffic or weather is inferred.</p>

    <div class="form-grid compact">
      <label>Distance (km)<input v-model.number="form.distance_km" type="number" min="0" /></label>
      <label>Minimum km/day<input v-model.number="form.minimum_km_per_day" type="number" min="0" /></label>
      <label>Planned transit (hours)<input v-model.number="form.planned_transit_hours" type="number" /></label>
      <label>Booking → start (hours)<input v-model.number="form.booking_to_start_hours" type="number" /></label>
      <label>Origin code<input v-model="form.origin_code" /></label>
      <label>Destination code<input v-model="form.destination_code" /></label>
      <label class="wide">Vehicle type<input v-model="form.vehicle_type" /></label>
      <label class="wide">Material<input v-model="form.material_shipped" /></label>
    </div>

    <button @click="predict" :disabled="loading">{{ loading ? 'Predicting…' : 'Predict Shipment Delay' }}</button>
    <p v-if="error" class="error-text">{{ error }}</p>

    <div v-if="result" class="result-grid">
      <article class="metric-card">
        <span>Delay risk</span>
        <b>{{ result.delay_probability == null ? '—' : (result.delay_probability * 100).toFixed(1) + '%' }}</b>
        <small>{{ result.predicted_delayed ? 'Predicted delayed' : 'Predicted on-time class' }}</small>
      </article>
      <article class="metric-card">
        <span>Predicted delay</span>
        <b>{{ result.predicted_delay_hours == null ? '—' : result.predicted_delay_hours.toFixed(1) + ' h' }}</b>
        <small>Experimental regression output</small>
      </article>
    </div>
  </section>
</template>
