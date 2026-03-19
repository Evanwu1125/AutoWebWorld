<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 font-sans">
    <div class="max-w-md w-full space-y-8 bg-white p-10 rounded-xl shadow-lg">
      <div class="text-center">
        <h2 class="mt-6 text-3xl font-extrabold text-gray-900">Sign in to your account</h2>
      </div>
      <div class="mt-8 space-y-6">
        <div class="rounded-md shadow-sm space-y-4">
          <div>
            <label for="customer-email" class="sr-only">Email address</label>
            <input 
                id="customer-email" 
                name="email" 
                type="email" 
                v-model="email"
                required 
                class="appearance-none rounded-lg relative block w-full px-4 py-3 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#008060] focus:border-[#008060] focus:z-10 sm:text-sm" 
                placeholder="Email address" 
            />
          </div>
          <div>
            <label for="customer-password" class="sr-only">Password</label>
            <input 
                id="customer-password" 
                name="password" 
                type="password" 
                v-model="password"
                required 
                class="appearance-none rounded-lg relative block w-full px-4 py-3 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#008060] focus:border-[#008060] focus:z-10 sm:text-sm" 
                placeholder="Password" 
            />
          </div>
        </div>

        <div>
          <button 
            id="customer-login-submit"
            @click="login"
            class="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-[#008060] hover:bg-[#004C3F] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#008060]"
          >
            Sign in
          </button>
        </div>
        
        <div class="text-center">
             <button 
                id="customer-back-home"
                @click="goHome"
                class="text-sm font-medium text-[#008060] hover:text-[#004C3F]"
            >
                Return to Store
            </button>
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
  name: 'CUSTOMER_LOGIN',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const email = computed({
        get: () => signatureStore.email,
        set: (val) => signatureStore.email = val
    })
    const password = computed({
        get: () => signatureStore.password,
        set: (val) => signatureStore.password = val
    })

    const login = async () => {
        if (email.value && password.value) {
            signatureStore.currentPageId = 'ACCOUNT_DASHBOARD'
            await router.push({ name: 'ACCOUNT_DASHBOARD' })
        } else {
            alert('Please enter credentials')
        }
    }

    const goHome = async () => {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        email,
        password,
        login,
        goHome
    }
  }
}
</script>