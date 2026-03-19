<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <!-- Logo as back home link -->
      <div 
        id="header-logo-home"
        class="flex justify-center cursor-pointer mb-6"
        @click="goHome"
      >
        <span class="text-3xl font-bold text-blue-700">Coursera</span>
      </div>
      <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
        Log in to your account
      </h2>
    </div>

    <div class="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
      <div 
        id="login-form" 
        class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10"
        @click="checkCanSubmit"
      >
        <div class="space-y-6">
          <div>
            <label for="login-email-input" class="block text-sm font-medium text-gray-700">
              Email address
            </label>
            <div class="mt-1">
              <input 
                id="login-email-input" 
                name="email" 
                type="email" 
                autocomplete="email" 
                required 
                class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                @input="handleEmailInput"
              >
            </div>
          </div>

          <div>
            <label for="login-password-input" class="block text-sm font-medium text-gray-700">
              Password
            </label>
            <div class="mt-1">
              <input 
                id="login-password-input" 
                name="password" 
                type="password" 
                autocomplete="current-password" 
                required 
                class="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                @input="handlePasswordInput"
              >
            </div>
          </div>

          <div>
            <button 
              id="login-submit-button"
              type="submit" 
              class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!store.login_can_submit"
              @click="handleSubmit"
            >
              Log In
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'LOGIN',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    function handleEmailInput(e) {
      if (e.target.value.length > 0) {
        store.login_email_filled = true
      } else {
        store.login_email_filled = false
      }
      checkCanSubmit()
    }

    function handlePasswordInput(e) {
      if (e.target.value.length > 0) {
        store.login_password_filled = true
      } else {
        store.login_password_filled = false
      }
      checkCanSubmit()
    }

    function checkCanSubmit() {
      store.login_can_submit = store.login_email_filled && store.login_password_filled
    }

    async function handleSubmit() {
      if (store.login_can_submit) {
        store.setCurrentPageId('COURSE_DISCOVERY')
        await router.push({ name: 'COURSE_DISCOVERY' })
      }
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      handleEmailInput,
      handlePasswordInput,
      checkCanSubmit,
      handleSubmit,
      goHome
    }
  }
}
</script>