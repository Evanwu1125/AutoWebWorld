<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8 bg-white p-10 rounded-xl shadow-2xl">
      <div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-[#005DAA]">
          Sign in to your account
        </h2>
        <p class="mt-2 text-center text-sm text-gray-600">
          Or
          <a id="back-home" @click="handleBackHome" class="font-medium text-[#009CDE] hover:text-blue-500 cursor-pointer">
            return to home
          </a>
        </p>
      </div>
      <div class="mt-8 space-y-6">
        <div class="rounded-md shadow-sm -space-y-px">
          <div class="mb-4">
            <label for="login-email" class="sr-only">Email address</label>
            <input
              id="login-email"
              name="email"
              type="email"
              required
              class="appearance-none rounded-md relative block w-full px-3 py-3 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#009CDE] focus:border-[#009CDE] focus:z-10 sm:text-sm"
              placeholder="Email address"
              @input="handleEmailInput"
            />
          </div>
          <div>
            <label for="login-password" class="sr-only">Password</label>
            <input
              id="login-password"
              name="password"
              type="password"
              required
              class="appearance-none rounded-md relative block w-full px-3 py-3 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-[#009CDE] focus:border-[#009CDE] focus:z-10 sm:text-sm"
              placeholder="Password"
              @input="handlePasswordInput"
            />
          </div>
        </div>

        <div>
          <button
            id="login-submit"
            @click="handleSubmit"
            class="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-[#005DAA] hover:bg-[#004a87] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#005DAA] shadow-md transition-colors"
          >
            Sign in
          </button>
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

    const handleEmailInput = (e) => {
      // ACT_LOGIN_TYPE_EMAIL
      store.login_email_entered = e.target.value
    }

    const handlePasswordInput = (e) => {
      // ACT_LOGIN_TYPE_PASSWORD
      store.login_password_entered = e.target.value
    }

    const handleSubmit = async () => {
      // ACT_LOGIN_SUBMIT
      // Preconditions: email > 0, password > 0
      if (store.login_email_entered.length > 0 && store.login_password_entered.length > 0) {
        store.login_can_submit = true
        store.setCurrentPageId('DASHBOARD')
        await router.push({ name: 'DASHBOARD' })
      } else {
        alert('Please enter email and password')
      }
    }

    const handleBackHome = async () => {
      // ACT_LOGIN_BACK_HOME
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      handleEmailInput,
      handlePasswordInput,
      handleSubmit,
      handleBackHome
    }
  }
}
</script>