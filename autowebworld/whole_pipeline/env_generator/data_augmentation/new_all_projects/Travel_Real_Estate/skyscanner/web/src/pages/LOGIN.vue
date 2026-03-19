<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-md p-8">
      <div class="text-center mb-8">
        <h2 class="text-3xl font-bold text-gray-900">Welcome Back</h2>
        <p class="text-gray-500 mt-2">Sign in to access your trips and alerts</p>
      </div>
      
      <div class="space-y-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
          <input 
            id="login-email-input"
            type="email" 
            @input="handleEmailInput"
            class="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
            placeholder="you@example.com"
          />
        </div>
        
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Password</label>
          <input 
            id="login-password-input"
            type="password" 
            @input="handlePasswordInput"
            class="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
            placeholder="••••••••"
          />
        </div>
        
        <button 
          id="login-validate-button"
          @click="validateForm"
          class="w-full py-3 bg-gray-900 text-white font-semibold rounded-lg hover:bg-gray-800 transition-colors"
        >
          Validate Credentials
        </button>

        <button 
          v-if="isValid"
          id="login-submit-button"
          @click="submitLogin"
          class="w-full py-3 bg-blue-600 text-white font-bold rounded-lg shadow-lg shadow-blue-600/30 hover:bg-blue-700 transition-all transform hover:-translate-y-0.5"
        >
          Sign In
        </button>
      </div>
      
      <div class="mt-8 text-center">
        <button id="back-home" @click="goHome" class="text-sm text-gray-500 hover:text-blue-600 underline">
          Back to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'LOGIN',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const isValid = computed(() => store.login_form_valid)

    const handleEmailInput = (e) => {
      store.login_email_entered = true
      // Store actual value if needed, but FSM just tracks entered status
    }

    const handlePasswordInput = (e) => {
      store.login_password_entered = true
    }

    const validateForm = () => {
      if (store.login_email_entered && store.login_password_entered) {
        store.login_form_valid = true
      }
    }

    const submitLogin = async () => {
      if (store.login_form_valid) {
        store.currentPageId = 'ACCOUNT_OVERVIEW'
        await router.push({ name: 'ACCOUNT_OVERVIEW' })
      }
    }

    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      isValid,
      handleEmailInput,
      handlePasswordInput,
      validateForm,
      submitLogin,
      goHome
    }
  }
}
</script>