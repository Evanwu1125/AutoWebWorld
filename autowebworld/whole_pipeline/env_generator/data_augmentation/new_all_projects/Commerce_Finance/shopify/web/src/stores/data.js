import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    products: [
      {
        id: 'prod_1',
        title: 'Minimalist Desk Lamp',
        price: 89.99,
        compare_at_price: 120.00,
        vendor: 'Lumina',
        product_type: 'Lighting',
        tags: ['sale', 'indoor', 'modern'],
        image: '/images/products_prod_1.jpg',
        salesCount: 1250,
        variants: [
            { sku: 'sku1_1', title: 'White', available: true },
            { sku: 'sku1_2', title: 'Black', available: true }
        ],
        description: 'A sleek and modern desk lamp perfect for any workspace.'
      },
      {
        id: 'prod_2',
        title: 'Ergonomic Office Chair',
        price: 299.00,
        vendor: 'ComfortSeating',
        product_type: 'Furniture',
        tags: ['office', 'ergonomic'],
        image: '/images/products_prod_2.jpg',
        salesCount: 890,
        variants: [
            { sku: 'sku2_1', title: 'Black Mesh', available: true }
        ],
        description: 'Designed for all-day comfort with adjustable lumbar support.'
      },
      {
        id: 'prod_3',
        title: 'Wireless Noise-Canceling Headphones',
        price: 201.50,
        compare_at_price: 249.99,
        vendor: 'AudioTech',
        product_type: 'Electronics',
        tags: ['sale', 'audio', 'wireless'],
        image: '/images/products_prod_3.jpg',
        salesCount: 2340,
        variants: [
            { sku: 'sku3_1', title: 'Silver', available: true },
            { sku: 'sku3_2', title: 'Matte Black', available: false }
        ],
        description: 'Immerse yourself in music with active noise cancellation.'
      },
      {
        id: 'prod_4',
        title: 'Smart Watch Series 5',
        price: 349.00,
        vendor: 'TechGear',
        product_type: 'Electronics',
        tags: ['wearable', 'smart'],
        image: '/images/products_prod_4.jpg',
        salesCount: 1580,
        variants: [
             { sku: 'sku4_1', title: '40mm', available: true },
             { sku: 'sku4_2', title: '44mm', available: true }
        ],
        description: 'Track your fitness and notifications on your wrist.'
      },
      {
        id: 'prod_5',
        title: 'Organic Cotton T-Shirt',
        price: 25.00,
        vendor: 'EcoWear',
        product_type: 'Clothing',
        tags: ['eco-friendly', 'clothing'],
        image: '/images/products_prod_5.jpg',
        salesCount: 3200,
        variants: [
            { sku: 'sku5_1', title: 'S', available: true },
            { sku: 'sku5_2', title: 'M', available: true },
            { sku: 'sku5_3', title: 'L', available: true }
        ],
        description: 'Soft, breathable, and made from 100% organic cotton.'
      },
      {
        id: 'prod_6',
        title: 'Ceramic Coffee Mug',
        price: 17.00,
        vendor: 'HomeGoods',
        product_type: 'Kitchen',
        tags: ['kitchen', 'gift'],
        image: '/images/products_prod_6.jpg',
        salesCount: 1890,
        variants: [
             { sku: 'sku6_1', title: 'White', available: true }
        ],
        description: 'Perfect for your morning brew.'
      },
      {
        id: 'prod_7',
        title: 'Leather Wallet',
        price: 45.00,
        vendor: 'LeatherCraft',
        product_type: 'Accessories',
        tags: ['leather', 'accessory'],
        image: '/images/products_prod_7.jpg',
        salesCount: 1120,
        variants: [
             { sku: 'sku7_1', title: 'Brown', available: true },
             { sku: 'sku7_2', title: 'Black', available: true }
        ],
        description: 'Genuine leather wallet with multiple card slots.'
      },
      {
        id: 'prod_8',
        title: 'Running Shoes',
        price: 110.00,
        compare_at_price: 130.00,
        vendor: 'SpeedRun',
        product_type: 'Footwear',
        tags: ['sale', 'sports', 'shoes'],
        image: '/images/products_prod_8.jpg',
        salesCount: 2670,
        variants: [
             { sku: 'sku8_1', title: 'US 9', available: true },
             { sku: 'sku8_2', title: 'US 10', available: true }
        ],
        description: 'Lightweight running shoes for maximum performance.'
      },
      {
        id: 'prod_9',
        title: 'Bamboo Cutting Board',
        price: 22.00,
        vendor: 'KitchenPro',
        product_type: 'Kitchen',
        tags: ['kitchen', 'eco-friendly'],
        image: '/images/products_prod_9.jpg',
        salesCount: 780,
        variants: [
             { sku: 'sku9_1', title: 'Standard', available: true }
        ],
        description: 'Durable and sustainable bamboo cutting board.'
      },
      {
        id: 'prod_10',
        title: 'Yoga Mat',
        price: 35.00,
        vendor: 'ZenLife',
        product_type: 'Sports',
        tags: ['fitness', 'yoga'],
        image: '/images/products_prod_10.jpg',
        salesCount: 1450,
        variants: [
             { sku: 'sku10_1', title: 'Purple', available: true },
             { sku: 'sku10_2', title: 'Blue', available: true }
        ],
        description: 'Non-slip yoga mat for all your poses.'
      },
      {
        id: 'prod_11',
        title: 'Bluetooth Speaker',
        price: 59.99,
        vendor: 'SoundWave',
        product_type: 'Electronics',
        tags: ['audio', 'portable'],
        image: '/images/products_prod_11.jpg',
        salesCount: 2100,
        variants: [
             { sku: 'sku11_1', title: 'Black', available: true }
        ],
        description: 'Portable speaker with powerful bass.'
      },
      {
        id: 'prod_12',
        title: 'Canvas Backpack',
        price: 48.00,
        vendor: 'TravelMate',
        product_type: 'Accessories',
        tags: ['travel', 'bag'],
        image: '/images/products_prod_12.jpg',
        salesCount: 950,
        variants: [
             { sku: 'sku12_1', title: 'Grey', available: true },
             { sku: 'sku12_2', title: 'Green', available: true }
        ],
        description: 'Stylish and spacious backpack for daily commute.'
      },
      {
        id: 'prod_13',
        title: 'Stainless Steel Water Bottle',
        price: 18.00,
        vendor: 'HydroLife',
        product_type: 'Accessories',
        tags: ['eco-friendly', 'drinkware'],
        image: '/images/products_prod_13.jpg',
        salesCount: 2850,
        variants: [
             { sku: 'sku13_1', title: 'Silver', available: true }
        ],
        description: 'Keep your drinks cold for 24 hours.'
      },
      {
        id: 'prod_14',
        title: 'Mechanical Keyboard',
        price: 149.99,
        vendor: 'KeyMaster',
        product_type: 'Electronics',
        tags: ['gaming', 'computer'],
        image: '/images/products_prod_14.jpg',
        salesCount: 1340,
        variants: [
             { sku: 'sku14_1', title: 'RGB', available: true }
        ],
        description: 'Tactile mechanical switches for precision typing.'
      },
      {
        id: 'prod_15',
        title: 'Sunglasses',
        price: 85.00,
        vendor: 'SunStyle',
        product_type: 'Accessories',
        tags: ['fashion', 'summer'],
        image: '/images/products_prod_15.jpg',
        salesCount: 670,
        variants: [
             { sku: 'sku15_1', title: 'Aviator', available: true }
        ],
        description: 'Classic aviator sunglasses with UV protection.'
      },
      {
        id: 'prod_16',
        title: 'Succulent Plant Set',
        price: 29.00,
        vendor: 'GreenThumb',
        product_type: 'Home Decor',
        tags: ['plants', 'indoor'],
        image: '/images/products_prod_16.jpg',
        salesCount: 1020,
        variants: [
             { sku: 'sku16_1', title: 'Set of 3', available: true }
        ],
        description: 'Low maintenance indoor plants.'
      }
    ],
    orders: [
      {
        id: 'ord_1001',
        order_number: '#1001',
        date: '2023-10-15',
        financial_status: 'Paid',
        fulfillment_status: 'Fulfilled',
        total: 125.50,
        customer_id: 'cust_1',
        items: [
            { title: 'Minimalist Desk Lamp', quantity: 1, price: 89.99 },
            { title: 'Organic Cotton T-Shirt', quantity: 1, price: 25.00 }
        ],
        shipping_address: { address1: '123 Main St', city: 'Anytown', postcode: '12345' }
      },
      {
        id: 'ord_1002',
        order_number: '#1002',
        date: '2023-11-02',
        financial_status: 'Paid',
        fulfillment_status: 'Unfulfilled',
        total: 299.00,
        customer_id: 'cust_1',
        items: [
             { title: 'Ergonomic Office Chair', quantity: 1, price: 299.00 }
        ],
        shipping_address: { address1: '123 Main St', city: 'Anytown', postcode: '12345' }
      },
      {
        id: 'ord_1003',
        order_number: '#1003',
        date: '2023-11-20',
        financial_status: 'Refunded',
        fulfillment_status: 'Restocked',
        total: 45.00,
        customer_id: 'cust_1',
        items: [
             { title: 'Leather Wallet', quantity: 1, price: 45.00 }
        ],
        shipping_address: { address1: '123 Main St', city: 'Anytown', postcode: '12345' }
      },
      {
        id: 'ord_1004',
        order_number: '#1004',
        date: '2023-12-05',
        financial_status: 'Paid',
        fulfillment_status: 'Fulfilled',
        total: 55.00,
        customer_id: 'cust_1',
        items: [
             { title: 'Yoga Mat', quantity: 1, price: 35.00 },
             { title: 'Ceramic Coffee Mug', quantity: 1, price: 15.00 }
        ],
        shipping_address: { address1: '456 Elm St', city: 'Othertown', postcode: '67890' }
      },
      {
        id: 'ord_1005',
        order_number: '#1005',
        date: '2023-12-10',
        financial_status: 'Pending',
        fulfillment_status: 'Unfulfilled',
        total: 349.00,
        customer_id: 'cust_1',
        items: [
             { title: 'Smart Watch Series 5', quantity: 1, price: 349.00 }
        ],
        shipping_address: { address1: '123 Main St', city: 'Anytown', postcode: '12345' }
      }
    ],
    addresses: [
      {
        id: 'addr_1',
        first_name: 'John',
        last_name: 'Doe',
        address1: '123 Main St',
        city: 'Anytown',
        country: 'United States',
        postcode: '12345',
        default: true
      },
      {
        id: 'addr_2',
        first_name: 'John',
        last_name: 'Doe',
        address1: '456 Elm St',
        city: 'Othertown',
        country: 'United States',
        postcode: '67890',
        default: false
      }
    ]
  }),
  persist: {
    storage: sessionStorage
  }
})