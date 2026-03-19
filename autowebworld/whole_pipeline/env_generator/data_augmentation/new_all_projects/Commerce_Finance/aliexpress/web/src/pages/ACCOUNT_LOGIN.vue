<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
        Sign in to AliExpress
      </h2>
      <p class="mt-2 text-center text-sm text-gray-600">
        Or
        <a href="#" class="font-medium text-red-600 hover:text-red-500">
          register for a new account
        </a>
      </p>
    </div>

    <div class="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 relative">
        <!-- Back Home (Absolute) -->
        <button 
          id="login-back-home" 
          class="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          @click="handleBackHome"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>

        <div class="space-y-6">
          <div>
            <label for="email" class="block text-sm font-medium text-gray-700"> Email address </label>
            <div class="mt-1">
              <input 
                id="login-email-input" 
                name="email" 
                type="email" 
                autocomplete="email" 
                required 
                class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm"
                :value="signatureStore.login_email"
                @input="handleEmailInput"
              />
            </div>
          </div>

          <div>
            <label for="password" class="block text-sm font-medium text-gray-700"> Password </label>
            <div class="mt-1">
              <input 
                id="login-password-input" 
                name="password" 
                type="password" 
                autocomplete="current-password" 
                required 
                class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-red-500 focus:border-red-500 sm:text-sm"
                :value="signatureStore.login_password"
                @input="handlePasswordInput"
              />
            </div>
          </div>

          <div>
            <button 
              id="login-submit-button" 
              type="submit" 
              class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              :disabled="!canSubmit"
              @click="handleSubmit"
            >
              Sign in
            </button>
          </div>
        </div>

        <div class="mt-6">
          <div class="relative">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-gray-300"></div>
            </div>
            <div class="relative flex justify-center text-sm">
              <span class="px-2 bg-white text-gray-500"> Or continue with </span>
            </div>
          </div>

          <div class="mt-6 grid grid-cols-3 gap-3">
            <div>
              <a href="#" class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                <span class="sr-only">Sign in with Facebook</span>
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M20 10c0-5.523-4.477-10-10-10S0 4.477 0 10c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V10h2.54V7.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V10h2.773l-.443 2.89h-2.33v6.988C16.343 19.128 20 14.991 20 10z" clip-rule="evenodd" /></svg>
              </a>
            </div>

            <div>
              <a href="#" class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                <span class="sr-only">Sign in with Google</span>
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 0C4.477 0 0 4.48 0 10c0 4.48 3.603 8.272 7.898 9.688.5.092.683-.217.683-.482 0-.237-.009-.866-.013-1.7-3.503.76-4.241-1.488-4.241-1.488-.455-1.156-1.11-1.464-1.11-1.464-1.143-.781.086-.765.086-.765 1.263.089 1.928 1.297 1.928 1.297 1.126 1.929 2.954 1.371 3.674 1.049.114-.815.44-1.37.8-1.685-2.8-3.18-5.63-1.29-5.63-5.745 0-1.27.45-2.31 1.19-3.12-.12-.29-.51-1.48.11-3.08 0 0 .97-.31 3.18 1.19.92-.26 1.91-.39 2.9-.39.99 0 1.98.13 2.9.39 2.21-1.5 3.18-1.19 3.18-1.19.63 1.6.23 2.79.11 3.08.74.81 1.19 1.85 1.19 3.12 0 4.47-2.84 5.56-5.53 5.84.46.4 0.87 1.18 0.87 2.38 0 1.72-.015 3.11-.015 3.53 0 .27.18.58.69.48C16.397 18.27 20 14.48 20 10c0-5.52-4.477-10-10-10z" clip-rule="evenodd" /></svg>
              </a>
            </div>

            <div>
              <a href="#" class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                <span class="sr-only">Sign in with Twitter</span>
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M6.29 18.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0020 3.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.073 4.073 0 01.8 7.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" /></svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT_LOGIN',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canSubmit = computed(() => {
      return signatureStore.login_email && signatureStore.login_email.length > 0 &&
             signatureStore.login_password && signatureStore.login_password.length > 0
    })

    const handleEmailInput = (e) => {
      signatureStore.login_email = e.target.value
    }

    const handlePasswordInput = (e) => {
      signatureStore.login_password = e.target.value
    }

    const handleSubmit = async () => {
      // In a real app, validate creds. Here we assume success as per FSM flow to ACCOUNT_OVERVIEW
      signatureStore.current_user_id = "user_123" // Simulate login success effect if needed, though not strictly in action effects
      signatureStore.currentPageId = 'ACCOUNT_OVERVIEW'
      await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleBackHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      signatureStore,
      canSubmit,
      handleEmailInput,
      handlePasswordInput,
      handleSubmit,
      handleBackHome
    }
  }
}
</script>