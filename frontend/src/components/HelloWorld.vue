<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  msg: string
}>()
const helloResponse = ref("Hello Placeholder")
const predictResponse = ref("[Predict Placeholder]")
const explanationSource = ref("/img/explanationFiller.png")

async function getResponse() {
  helloResponse.value = await fetch('http://localhost:5000/')
        .then(response => response.json().then(json => json.lastmessage))
        .catch(error => error.toString())
}

async function getPrediction() {
  predictResponse.value = await fetch('http://localhost:5000/predict')
        .then(response => response.json().then(json => JSON.stringify(json)))
        .catch(error => error.toString())
}

async function getExplanation() {
  explanationSource.value = "http://localhost:5000/explain"
}
</script>

<template>
  <div class="greetings">
    <h1 class="green">{{ msg }}</h1>
    <h1>{{helloResponse}}</h1>
    <h1>Prediction: {{predictResponse}}</h1>
    <button @click="getResponse()">Get Response</button>
    <button @click="getPrediction()">Get Prediction</button>
    <button @click="getExplanation()">Get Explanation</button>
    <img :src="explanationSource" />
    <h3>
      You’ve successfully created a project with
      <a href="https://vitejs.dev/" target="_blank" rel="noopener">Vite</a> +
      <a href="https://vuejs.org/" target="_blank" rel="noopener">Vue 3</a>.
    </h3>
  </div>
</template>

<style scoped>
h1 {
  font-weight: 500;
  font-size: 2.6rem;
  top: -10px;
}

h3 {
  font-size: 1.2rem;
}

.greetings h1,
.greetings h3 {
  text-align: center;
}

@media (min-width: 1024px) {
  .greetings h1,
  .greetings h3 {
    text-align: left;
  }
}
</style>
