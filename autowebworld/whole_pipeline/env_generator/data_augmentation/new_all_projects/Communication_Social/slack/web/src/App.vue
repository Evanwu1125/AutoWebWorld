<template>
  <div id="app" class="h-full w-full font-sans antialiased text-gray-900 bg-white">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
/* Global resets and base styles if needed */
body {
  margin: 0;
  padding: 0;
}
</style>