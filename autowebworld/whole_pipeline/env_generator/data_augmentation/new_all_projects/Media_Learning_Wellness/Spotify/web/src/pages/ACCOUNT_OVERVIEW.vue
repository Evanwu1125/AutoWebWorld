<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <!-- Sidebar -->
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div class="text-white mb-8 flex items-center space-x-2">
         <svg viewBox="0 0 167.5 167.5" class="w-8 h-8 fill-current text-white"><path d="M83.7 0C37.5 0 0 37.5 0 83.7c0 46.3 37.5 83.7 83.7 83.7 46.3 0 83.7-37.5 83.7-83.7S130 0 83.7 0zM122 120.8c-1.4 2.5-4.6 3.2-7.1 1.7-19.8-12.1-44.8-14.9-74.2-8.1-2.8.6-5.6-1.1-6.2-3.9-.6-2.8 1.1-5.6 3.9-6.2 32-7.3 59.6-4.2 81.9 9.3 2.5 1.5 3.4 4.7 1.7 7.2zm10.1-22.5c-1.8 3-5.6 3.9-8.5 2.1-22.8-14-57.6-18.1-84.5-9.9-3.3 1-6.9-1-7.9-4.3-1-3.3 1-6.9 4.3-7.9 30.3-9.2 69.2-4.6 94.6 11 3 1.8 3.9 5.6 2 8.5zm.4-23c-27.3-16.2-72.3-17.7-98.4-9.7-4.2 1.3-8.6-1-9.9-5.2-1.3-4.2 1-8.6 5.2-9.9 30.3-9.2 79.7-7.4 111 11.2 3.8 2.2 5 7.1 2.8 10.9-2.2 3.9-7.2 5.1-10.7 2.7z"/></svg>
         <span class="text-2xl font-bold">Spotify</span>
      </div>
      <nav class="space-y-2">
         <div id="back-home" @click="handleBackHome" class="flex items-center space-x-4 px-4 py-3 text-[#B3B3B3] hover:text-white cursor-pointer hover:bg-[#282828] rounded font-bold">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
            <span>Home</span>
         </div>
         <div class="px-4 py-3 bg-[#282828] text-white rounded font-bold flex items-center space-x-4">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
             <span>Account Overview</span>
         </div>
      </nav>
    </aside>

    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-4xl mx-auto w-full">
      <h1 class="text-4xl font-bold mb-8">Account Overview</h1>
      
      <!-- Profile Section -->
      <section class="mb-10">
         <h2 class="text-2xl font-bold mb-4">Profile</h2>
         <div class="bg-[#181818] p-6 rounded-lg border border-[#282828]">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
               <div>
                  <label class="text-[#B3B3B3] text-sm font-bold uppercase block mb-1">Username</label>
                  <div class="text-white text-lg">user123</div>
               </div>
               <div>
                  <label class="text-[#B3B3B3] text-sm font-bold uppercase block mb-1">Email</label>
                  <div class="text-white text-lg">user@example.com</div>
               </div>
               <div>
                  <label class="text-[#B3B3B3] text-sm font-bold uppercase block mb-1">Date of birth</label>
                  <div class="text-white text-lg">May 10, 1995</div>
               </div>
               <div>
                  <label class="text-[#B3B3B3] text-sm font-bold uppercase block mb-1">Country</label>
                  <div class="text-white text-lg">USA</div>
               </div>
            </div>
            <div class="mt-6 pt-6 border-t border-[#282828]">
               <button class="border border-[#727272] hover:border-white text-white font-bold py-3 px-8 rounded-full uppercase tracking-widest text-sm transition-colors">
                  Edit Profile
               </button>
            </div>
         </div>
      </section>

      <!-- Plan Section -->
      <section class="mb-10">
         <h2 class="text-2xl font-bold mb-4">Your Plan</h2>
         <div class="bg-[#242424] p-8 rounded-lg relative overflow-hidden">
            <div class="absolute top-0 right-0 bg-gradient-to-l from-[#9d4edd] to-transparent w-1/2 h-full opacity-20"></div>
            <div class="relative z-10">
               <h3 class="text-3xl font-bold mb-2">Spotify Free</h3>
               <p class="text-[#B3B3B3] mb-6">Play music in shuffle mode only, with ads.</p>
               
               <div class="bg-gradient-to-r from-[#af2896] to-[#509bf5] p-6 rounded-lg text-white mb-6">
                  <h4 class="text-xl font-bold mb-2">Premium Individual</h4>
                  <ul class="list-disc list-inside mb-4 space-y-1 text-sm">
                     <li>Ad-free music listening</li>
                     <li>Play anywhere - even offline</li>
                     <li>On-demand playback</li>
                  </ul>
                  <button 
                     id="account-go-premium" 
                     @click="handleGoPremium" 
                     class="bg-white text-black font-bold py-3 px-8 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm"
                  >
                     Join Premium
                  </button>
               </div>
            </div>
         </div>
      </section>

      <!-- Quick Links -->
      <section class="grid grid-cols-1 md:grid-cols-2 gap-6">
         <div 
           id="account-settings-link"
           @click="handleGoSettings"
           class="bg-[#181818] p-6 rounded-lg cursor-pointer hover:bg-[#282828] transition-colors group"
         >
            <div class="flex items-center justify-between mb-2">
               <h3 class="text-xl font-bold group-hover:underline">App Settings</h3>
               <svg class="w-6 h-6 text-[#B3B3B3]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
            </div>
            <p class="text-[#B3B3B3] text-sm">Theme, content filters, notifications.</p>
         </div>

         <div 
           id="account-payment-methods-link"
           @click="handleGoPaymentMethods"
           class="bg-[#181818] p-6 rounded-lg cursor-pointer hover:bg-[#282828] transition-colors group"
         >
            <div class="flex items-center justify-between mb-2">
               <h3 class="text-xl font-bold group-hover:underline">Payment Methods</h3>
               <svg class="w-6 h-6 text-[#B3B3B3]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"/></svg>
            </div>
            <p class="text-[#B3B3B3] text-sm">Manage cards and billing info.</p>
         </div>
      </section>
    </main>
  </div>
</template>

<script>
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'ACCOUNT_OVERVIEW',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const handleBackHome = async () => {
       store.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    const handleGoPremium = async () => {
       store.setCurrentPageId('PREMIUM_UPSELL')
       await router.push({ name: 'PREMIUM_UPSELL' })
    }

    const handleGoSettings = async () => {
       store.setCurrentPageId('SETTINGS')
       await router.push({ name: 'SETTINGS' })
    }

    const handleGoPaymentMethods = async () => {
       store.setCurrentPageId('PAYMENT_METHODS')
       await router.push({ name: 'PAYMENT_METHODS' })
    }

    return {
       handleBackHome,
       handleGoPremium,
       handleGoSettings,
       handleGoPaymentMethods
    }
  }
}
</script>