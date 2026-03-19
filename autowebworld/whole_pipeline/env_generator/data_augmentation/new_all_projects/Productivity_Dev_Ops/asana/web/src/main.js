import { createApp } from 'vue'
import { createPinia } from 'pinia'
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import App from './App.vue'
import router from './router'
import './style.css'
import { useDataStore } from './stores/data'

const app = createApp(App)
const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

// Ensure mock data is ready before any route renders
const dataStore = useDataStore(pinia)
dataStore.initializeMockData()

app.use(pinia)
app.use(router)

app.mount('#app')
