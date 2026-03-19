<template>
  <div class="min-h-screen bg-white">
    <!-- Navigation -->
    <nav class="border-b border-gray-200 sticky top-0 bg-white/95 backdrop-blur z-40">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16 items-center">
          <div class="flex items-center">
            <h1 class="text-3xl font-black tracking-tighter font-serif mr-8">Medium</h1>
          </div>
          <div class="flex items-center space-x-6">
            <!-- Stories Nav (Click to navigate directly, Hover to show menu) -->
            <div id="nav-stories" class="relative group">
              <button class="text-gray-500 hover:text-gray-900 text-sm font-sans" @click="handleGoToPostListDirect">Stories</button>
              <!-- Hover Menu -->
              <div class="absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 ring-1 ring-black ring-opacity-5 z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                <div id="nav-stories-hover" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToPostListHover">View Stories</div>
              </div>
            </div>

            <!-- More Menu (Click to show menu with Stories option) -->
            <div class="relative">
              <button id="nav-more" class="text-gray-500 hover:text-gray-900 text-sm font-sans" @click="toggleMoreMenu">More</button>

              <!-- More Menu Dropdown -->
              <div v-if="moreMenuOpen" class="absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 ring-1 ring-black ring-opacity-5 z-50">
                <div id="nav-more-stories" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToPostListMenu">Stories</div>
              </div>
            </div>

            <!-- Write Nav (Direct Click) -->
            <button id="nav-write-direct" class="text-gray-500 hover:text-gray-900 flex items-center space-x-2 text-sm font-sans" @click="handleGoToNewStoryDirect">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span>Write</span>
            </button>

            <!-- Write Nav (Hover Menu) -->
            <div id="nav-write" class="relative group">
              <button class="text-gray-500 hover:text-gray-900 flex items-center space-x-2 text-sm font-sans">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                <span>Write Hover</span>
              </button>

              <!-- Hover Menu -->
              <div class="absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 ring-1 ring-black ring-opacity-5 z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                <div id="nav-write-new-story" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToNewStoryHover">New Story</div>
              </div>
            </div>

            <!-- Profile & Menu -->
            <div class="relative ml-4 flex items-center space-x-2">
              <!-- Hover Menu Container with Direct Profile Link -->
              <div id="nav-profile" class="relative group">
                <!-- Direct Profile Link (Click to navigate directly, Hover to show menu) -->
                <button id="nav-profile-direct" class="flex items-center focus:outline-none" @click="handleGoToProfileDirect">
                  <img :src="currentUser.avatar" class="h-8 w-8 rounded-full border border-gray-200 object-cover" alt="User avatar" />
                </button>

                <!-- Hover Menu (Triggered by hovering over #nav-profile) -->
                <div class="absolute right-0 top-full mt-2 w-56 bg-white rounded-md shadow-lg py-1 ring-1 ring-black ring-opacity-5 z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                  <div id="nav-profile-hover-view" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileHover">Profile</div>
                  <div id="nav-profile-hover-settings" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileMenu('settings')">Settings</div>
                  <div id="nav-profile-hover-lists" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileMenu('lists')">Lists</div>
                  <div class="border-t border-gray-100 my-1"></div>
                  <div id="nav-profile-hover-stories" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToPostListMenu">Stories</div>
                  <div id="nav-profile-hover-write" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToNewStoryMenu">Write a Story</div>
                </div>
              </div>

              <!-- Toggle Button for Click Menu -->
              <button id="nav-profile-toggle" class="flex items-center focus:outline-none" @click="toggleMenu">
                <svg class="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <!-- Click Menu (Triggered by toggle button) -->
              <div v-if="menuOpen" id="nav-profile-menu" class="absolute right-0 top-full mt-2 w-56 bg-white rounded-md shadow-lg py-1 ring-1 ring-black ring-opacity-5 z-50">
                <div id="nav-profile-view" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileMenu('profile')">Profile</div>
                <div id="nav-profile-settings" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileMenu('settings')">Settings</div>
                <div id="nav-profile-lists" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToProfileMenu('lists')">Lists</div>
                <div class="border-t border-gray-100 my-1"></div>
                <div id="nav-more-stories" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToPostListMenu">Stories</div>
                <div id="nav-profile-write-story" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer font-sans" @click="handleGoToNewStoryMenu">Write a Story</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="bg-[#FFC017] border-b border-gray-900">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 flex items-center justify-between">
        <div class="max-w-xl">
          <h2 class="text-8xl font-serif mb-6 tracking-tight">Stay curious.</h2>
          <p class="text-2xl font-serif mb-10 leading-snug">Discover stories, thinking, and expertise from writers on any topic.</p>
          <button class="bg-black text-white px-8 py-3 rounded-full text-xl font-sans font-medium hover:bg-gray-800 transition-colors" @click="handleGoToPostListDirect">Start reading</button>
        </div>
        <div class="hidden lg:block">
           <!-- Using ImageGetter via explicit path logic in store, assuming hero image -->
           <!-- Placeholder for artistic element -->
           <div class="w-96 h-96 opacity-80 mix-blend-multiply" :style="{ backgroundImage: 'url(/images/HeroIllustration.jpg)', backgroundSize: 'contain', backgroundRepeat: 'no-repeat' }"></div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col lg:flex-row gap-12">
      <!-- Feed -->
      <div class="flex-1" id="home-feed">
        <div v-for="post in feedPosts" :key="post.id" :class="`data-id-${post.id} flex gap-8 mb-12 items-start group cursor-pointer`">
          <div class="flex-1">
            <div class="flex items-center gap-2 mb-2">
              <img :src="getUser(post.author_id).avatar" class="w-5 h-5 rounded-full" />
              <span class="text-xs font-sans font-medium">{{ getUser(post.author_id).name }}</span>
              <span class="text-xs text-gray-500 font-sans">{{ formatDate(post.published_date) }}</span>
            </div>
            <h3 class="text-xl font-bold font-serif mb-1 group-hover:underline decoration-gray-800 decoration-2 post-visible" @click="handleOpenPost(post.id)">{{ post.title }}</h3>
            <p class="text-gray-500 font-serif text-base mb-4 line-clamp-2">{{ post.content }}</p>
            <div class="flex items-center gap-4">
              <span class="bg-gray-100 px-2 py-1 rounded-full text-xs text-gray-700 font-sans">{{ post.tag || 'General' }}</span>
              <span class="text-xs text-gray-500 font-sans">{{ post.length_minutes }} min read</span>
            </div>
          </div>
          <div class="w-32 h-32 flex-shrink-0">
            <img :src="post.image" class="w-full h-full object-cover rounded-sm" :alt="post.title" />
          </div>
        </div>
      </div>

      <!-- Sidebar -->
      <aside class="w-full lg:w-80 border-l border-gray-100 lg:pl-8 hidden lg:block">
        <div class="sticky top-24">
          <h4 class="font-sans font-bold text-sm tracking-wide uppercase mb-4">Discover more of what matters to you</h4>
          <div class="flex flex-wrap gap-2 mb-8">
            <span class="px-4 py-2 border border-gray-200 rounded-full text-sm text-gray-600 font-sans hover:border-gray-400 cursor-pointer transition-colors">Programming</span>
            <span class="px-4 py-2 border border-gray-200 rounded-full text-sm text-gray-600 font-sans hover:border-gray-400 cursor-pointer transition-colors">Data Science</span>
            <span class="px-4 py-2 border border-gray-200 rounded-full text-sm text-gray-600 font-sans hover:border-gray-400 cursor-pointer transition-colors">Technology</span>
            <span class="px-4 py-2 border border-gray-200 rounded-full text-sm text-gray-600 font-sans hover:border-gray-400 cursor-pointer transition-colors">Self Improvement</span>
            <span class="px-4 py-2 border border-gray-200 rounded-full text-sm text-gray-600 font-sans hover:border-gray-400 cursor-pointer transition-colors">Writing</span>
          </div>
          
          <div class="border-t border-gray-100 pt-8">
             <h4 class="font-sans font-bold text-sm tracking-wide mb-4">Recommended</h4>
             <!-- Mini list -->
             <div class="space-y-6">
                <div v-for="i in 3" :key="i" class="flex items-start gap-4">
                   <div class="flex-1">
                      <div class="flex items-center gap-2 mb-1">
                         <div class="w-5 h-5 bg-gray-200 rounded-full"></div>
                         <div class="h-3 w-20 bg-gray-100 rounded"></div>
                      </div>
                      <div class="h-4 w-full bg-gray-100 rounded mb-1"></div>
                      <div class="h-4 w-2/3 bg-gray-100 rounded"></div>
                   </div>
                </div>
             </div>
          </div>
        </div>
      </aside>
    </main>
    
    <!-- Nav Hidden Links for Hover Actions -->
    <div id="nav-write" class="hidden"></div>
    <div id="nav-profile" class="hidden"></div>
    <div id="nav-more" class="hidden" @click="toggleMenu"></div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const menuOpen = ref(false)
    const moreMenuOpen = ref(false)

    const currentUser = computed(() => dataStore.getUserById(signatureStore.current_user_id))
    const feedPosts = computed(() => dataStore.posts.slice(0, 10)) // Show first 10 posts

    const getUser = (id) => dataStore.getUserById(id)

    const formatDate = (dateStr) => {
      return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    }

    const toggleMenu = () => {
      menuOpen.value = !menuOpen.value
    }

    const toggleMoreMenu = () => {
      moreMenuOpen.value = !moreMenuOpen.value
    }

    // Navigation Actions
    const handleGoToPostListDirect = async () => {
      signatureStore.setCurrentPageId('POST_LIST')
      await router.push({ name: 'POST_LIST' })
    }
    
    const handleGoToPostListHover = async () => {
      signatureStore.setCurrentPageId('POST_LIST')
      await router.push({ name: 'POST_LIST' })
    }
    
    const handleGoToPostListMenu = async () => {
      signatureStore.setCurrentPageId('POST_LIST')
      await router.push({ name: 'POST_LIST' })
    }
    
    const handleGoToNewStoryDirect = async () => {
      signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
      await router.push({ name: 'NEW_STORY_EDITOR' })
    }
    
    const handleGoToNewStoryHover = async () => {
      signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
      await router.push({ name: 'NEW_STORY_EDITOR' })
    }
    
    const handleGoToNewStoryMenu = async () => {
      signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
      await router.push({ name: 'NEW_STORY_EDITOR' })
    }
    
    const handleGoToProfileDirect = async () => {
      signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
      await router.push({ name: 'PROFILE_OVERVIEW' })
    }
    
    const handleGoToProfileHover = async () => {
      signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
      await router.push({ name: 'PROFILE_OVERVIEW' })
    }
    
    const handleGoToProfileMenu = async (view) => {
      // View param is for specific logic, but FSM maps all to PROFILE_OVERVIEW for now or specific flows
      // In FSM: 
      // #nav-profile-view -> PROFILE_OVERVIEW
      // #nav-profile-settings -> PROFILE_OVERVIEW (Wait, check FSM)
      // Actually FSM says "ACT_HOME_GO_TO_PROFILE_MENU" goes to PROFILE_OVERVIEW.
      // But the options have different selectors. 
      // The FSM example uses #nav-profile-view.
      // We should implement handlers for all options if they are in gui_procedure ui_elements.
      // But FSM only defines one action for this dropdown flow.
      // So all lead to PROFILE_OVERVIEW for now based on this specific action definition.
      signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
      await router.push({ name: 'PROFILE_OVERVIEW' })
    }

    const handleOpenPost = async (postId) => {
      signatureStore.home_selected_post_id = postId
      signatureStore.home_viewport_anchor_id = null // Clear anchor
      signatureStore.setCurrentPageId('POST_DETAIL')
      await router.push({ name: 'POST_DETAIL', params: { id: postId } })
    }

    return {
      currentUser,
      feedPosts,
      getUser,
      formatDate,
      menuOpen,
      moreMenuOpen,
      toggleMenu,
      toggleMoreMenu,
      handleGoToPostListDirect,
      handleGoToPostListHover,
      handleGoToPostListMenu,
      handleGoToNewStoryDirect,
      handleGoToNewStoryHover,
      handleGoToNewStoryMenu,
      handleGoToProfileDirect,
      handleGoToProfileHover,
      handleGoToProfileMenu,
      handleOpenPost
    }
  }
}
</script>