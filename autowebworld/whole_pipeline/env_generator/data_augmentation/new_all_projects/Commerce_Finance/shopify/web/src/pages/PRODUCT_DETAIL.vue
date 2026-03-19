<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
    <!-- Breadcrumb Nav (Simulated for visual hierarchy) -->
    <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center space-x-2 text-sm">
            <span id="back-to-collections" @click="goBack" class="text-gray-500 hover:text-[#008060] cursor-pointer flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Collections
            </span>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12" v-if="product">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-12 lg:gap-16">
        <!-- Product Images -->
        <div class="space-y-4">
            <div class="aspect-square bg-white rounded-2xl overflow-hidden shadow-sm border border-gray-100">
                <img :src="product.image" :alt="product.title" class="w-full h-full object-cover" />
            </div>
            <!-- Thumbnails (Visual only) -->
            <div class="grid grid-cols-4 gap-4">
                <div class="aspect-square rounded-lg overflow-hidden border border-[#008060] ring-2 ring-[#008060] ring-offset-2">
                    <img :src="product.image" class="w-full h-full object-cover" />
                </div>
            </div>
        </div>

        <!-- Product Info -->
        <div>
            <div class="mb-6">
                <h1 class="text-4xl font-extrabold text-gray-900 mb-2">{{ product.title }}</h1>
                <div class="flex items-center space-x-4 mb-4">
                    <span class="text-2xl font-bold text-gray-900">${{ product.price.toFixed(2) }}</span>
                    <span v-if="product.compare_at_price" class="text-lg text-gray-500 line-through">${{ product.compare_at_price.toFixed(2) }}</span>
                    <span v-if="product.compare_at_price" class="bg-red-100 text-red-800 text-xs font-bold px-2 py-1 rounded">SALE</span>
                </div>
                <div class="flex items-center space-x-1 mb-6 cursor-pointer hover:underline" id="reviews-tab" @click="goToReviews">
                    <div class="flex text-yellow-400">
                        <span>★</span><span>★</span><span>★</span><span>★</span><span>★</span>
                    </div>
                    <span class="text-gray-500 text-sm">(24 reviews)</span>
                </div>
            </div>

            <div class="space-y-6">
                <!-- Variant Selector -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Select Variant</label>
                    <div class="relative" id="variant-dropdown">
                        <!-- Custom Dropdown Trigger -->
                        <div
                            @click="toggleVariantDropdown"
                            class="w-full border border-gray-300 rounded-lg shadow-sm py-3 px-4 bg-white cursor-pointer flex justify-between items-center hover:border-[#008060] transition-colors"
                        >
                            <span :class="selectedSku ? 'text-gray-900' : 'text-gray-500'">
                                {{ selectedVariantTitle || 'Choose an option' }}
                            </span>
                            <svg class="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </div>

                        <!-- Custom Dropdown Options -->
                        <div
                            v-if="variantDropdownOpen"
                            class="absolute z-10 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto"
                        >
                            <div
                                v-for="(variant, index) in product.variants"
                                :key="variant.sku"
                                :id="`variant-option-${index + 1}`"
                                @click="selectVariant(variant.sku, variant.title)"
                                :class="[
                                    'px-4 py-3 cursor-pointer transition-colors',
                                    selectedSku === variant.sku
                                        ? 'bg-[#008060] text-white'
                                        : 'hover:bg-gray-100 text-gray-700'
                                ]"
                            >
                                {{ variant.title }}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Quantity -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">Quantity</label>
                    <input 
                        id="quantity-input"
                        type="number" 
                        v-model.number="quantity" 
                        @input="handleQuantityChange"
                        min="1"
                        class="w-32 border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] focus:ring-opacity-50 py-3 px-4"
                    />
                </div>

                <!-- Actions -->
                <div class="space-y-3 pt-4">
                    <button 
                        id="add-to-cart-button" 
                        @click="addToCart"
                        :disabled="!selectedSku || quantity < 1"
                        class="w-full bg-white border border-[#008060] text-[#008060] hover:bg-gray-50 font-bold py-4 px-8 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Add to Cart
                    </button>
                    <button 
                        id="buy-now-button" 
                        @click="buyNow"
                        :disabled="!selectedSku || quantity < 1"
                        class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-8 rounded-lg shadow-md hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Buy it Now
                    </button>
                </div>
                
                <!-- Description -->
                <div class="prose prose-sm text-gray-600 mt-8 pt-8 border-t border-gray-200">
                    <p>{{ product.description }}</p>
                </div>
            </div>
        </div>
      </div>
    </main>
    
    <div v-else class="min-h-screen flex items-center justify-center">
        <div class="text-center">
            <h2 class="text-2xl font-bold text-gray-900">Product not found</h2>
            <button @click="goBack" class="mt-4 text-[#008060] hover:underline">Return to Collections</button>
        </div>
    </div>
  </div>
</template>

<script>
import { computed, ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRODUCT_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const product = computed(() => dataStore.products.find(p => p.id === route.params.id))

    const selectedSku = ref('')
    const selectedVariantTitle = ref('')
    const quantity = ref(1)
    const variantDropdownOpen = ref(false)

    onMounted(() => {
        if (product.value) {
            signatureStore.selected_product_id = product.value.id
            // Reset local state
            selectedSku.value = ''
            selectedVariantTitle.value = ''
            quantity.value = 1
        }
    })

    const toggleVariantDropdown = () => {
        variantDropdownOpen.value = !variantDropdownOpen.value
    }

    const selectVariant = (sku, title) => {
        selectedSku.value = sku
        selectedVariantTitle.value = title
        variantDropdownOpen.value = false
        handleVariantSelect()
    }

    const handleVariantSelect = () => {
        signatureStore.selected_variant_sku = selectedSku.value
    }

    const handleQuantityChange = () => {
        signatureStore.selected_quantity = quantity.value
    }

    const goBack = async () => {
        signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
        await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
    }

    const goToReviews = async () => {
        signatureStore.currentPageId = 'PRODUCT_REVIEWS'
        await router.push({ name: 'PRODUCT_REVIEWS', params: { id: route.params.id } })
    }

    const addToCart = async () => {
        if (!selectedSku.value || quantity.value < 1) return

        // Add to global cart
        const variant = product.value.variants.find(v => v.sku === selectedSku.value)
        const newItem = {
            product_id: product.value.id,
            sku: selectedSku.value,
            title: product.value.title,
            variant_title: variant ? variant.title : '',
            price: product.value.price,
            quantity: quantity.value,
            image: product.value.image,
            // Unique ID for cart item
            id: `${product.value.id}-${selectedSku.value}-${Date.now()}`
        }
        
        signatureStore.cart_items.push(newItem)
        
        // Update subtotal
        signatureStore.cart_subtotal = signatureStore.cart_items.reduce((sum, item) => sum + (item.price * item.quantity), 0)

        signatureStore.currentPageId = 'CART'
        await router.push({ name: 'CART' })
    }

    const buyNow = async () => {
        if (!selectedSku.value || quantity.value < 1) return
        
        // Just navigate, logic handled in checkout flow (usually pass items, but FSM simplifies to direct nav)
        signatureStore.currentPageId = 'CHECKOUT_INFORMATION_BUY_NOW'
        await router.push({ name: 'CHECKOUT_INFORMATION_BUY_NOW' })
    }

    return {
        product,
        selectedSku,
        selectedVariantTitle,
        quantity,
        variantDropdownOpen,
        toggleVariantDropdown,
        selectVariant,
        handleVariantSelect,
        handleQuantityChange,
        goBack,
        goToReviews,
        addToCart,
        buyNow
    }
  }
}
</script>