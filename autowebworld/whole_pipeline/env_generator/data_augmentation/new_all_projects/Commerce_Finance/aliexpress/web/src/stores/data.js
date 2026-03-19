import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    products: [
      {
        id: 'prod_1',
        name: 'Global Version Xiaomi Redmi Note 13 Pro 5G Smartphone 200MP Camera',
        price: 249.99,
        originalPrice: 329.99,
        discount: 24,
        stock: 85,
        image: '/images/products_prod_1.jpg',
        shipping: 'Free',
        sold: '10k+',
        rating: 4.8
      },
      {
        id: 'prod_2',
        name: 'Lenovo LP40 Pro Wireless Earphones Bluetooth 5.1 Waterproof Headset',
        price: 12.99,
        originalPrice: 25.99,
        discount: 50,
        stock: 45,
        image: '/images/products_prod_2.jpg',
        shipping: 'Free',
        sold: '50k+',
        rating: 4.7
      },
      {
        id: 'prod_3',
        name: 'Men\'s Casual Slim Fit T-Shirt Cotton Blend Solid Color Tops',
        price: 8.50,
        originalPrice: 15.00,
        discount: 43,
        stock: 92,
        image: '/images/products_prod_3.jpg',
        shipping: '$2.00',
        sold: '5k+',
        rating: 4.5
      },
      {
        id: 'prod_4',
        name: 'Women\'s Summer Floral Print Maxi Dress Boho Beach Party',
        price: 18.99,
        originalPrice: 29.99,
        discount: 37,
        stock: 68,
        image: '/images/products_prod_4.jpg',
        shipping: 'Free',
        sold: '2k+',
        rating: 4.6
      },
      {
        id: 'prod_5',
        name: 'Smart Watch Series 9 Men Women NFC Bluetooth Call Waterproof',
        price: 22.99,
        originalPrice: 49.99,
        discount: 54,
        stock: 23,
        image: '/images/products_prod_5.jpg',
        shipping: 'Free',
        sold: '20k+',
        rating: 4.4
      },
      {
        id: 'prod_6',
        name: 'Professional Hair Clipper Trimmer for Men Barber Beard Shaver',
        price: 15.99,
        originalPrice: 30.99,
        discount: 48,
        stock: 56,
        image: '/images/products_prod_6.jpg',
        shipping: 'Free',
        sold: '8k+',
        rating: 4.8
      },
      {
        id: 'prod_7',
        name: 'Portable Mini Humidifier USB Cool Mist Aroma Diffuser with LED Light',
        price: 5.99,
        originalPrice: 12.99,
        discount: 54,
        stock: 78,
        image: '/images/products_prod_7.jpg',
        shipping: '$1.50',
        sold: '3k+',
        rating: 4.3
      },
      {
        id: 'prod_8',
        name: 'High Speed SSD 1TB 2TB External Hard Drive Portable Storage',
        price: 28.99,
        originalPrice: 59.99,
        discount: 52,
        stock: 34,
        image: '/images/products_prod_8.jpg',
        shipping: 'Free',
        sold: '15k+',
        rating: 4.9
      },
      {
        id: 'prod_9',
        name: 'Car Phone Holder Mount Stand Dashboard Windshield Clip',
        price: 3.99,
        originalPrice: 9.99,
        discount: 60,
        stock: 12,
        image: '/images/products_prod_9.jpg',
        shipping: '$0.99',
        sold: '100k+',
        rating: 4.7
      },
      {
        id: 'prod_10',
        name: 'Cute Cat Paw Cushion Seat Back Pillow Plush Chair Pad',
        price: 14.50,
        originalPrice: 22.00,
        discount: 34,
        stock: 88,
        image: '/images/products_prod_10.jpg',
        shipping: 'Free',
        sold: '1k+',
        rating: 4.8
      },
      {
        id: 'prod_11',
        name: 'LED Strip Lights RGB 5050 Decoration Bedroom Neon Tape',
        price: 7.99,
        originalPrice: 14.99,
        discount: 47,
        stock: 61,
        image: '/images/products_prod_11.jpg',
        shipping: 'Free',
        sold: '30k+',
        rating: 4.6
      },
      {
        id: 'prod_12',
        name: 'Adjustable Laptop Stand Portable Aluminum Holder for Macbook',
        price: 11.99,
        originalPrice: 24.99,
        discount: 52,
        stock: 29,
        image: '/images/products_prod_12.jpg',
        shipping: 'Free',
        sold: '6k+',
        rating: 4.8
      },
      {
        id: 'prod_13',
        name: 'Vintage Polarized Sunglasses Men Women Brand Designer',
        price: 6.99,
        originalPrice: 18.99,
        discount: 63,
        stock: 15,
        image: '/images/products_prod_13.jpg',
        shipping: 'Free',
        sold: '12k+',
        rating: 4.5
      },
      {
        id: 'prod_14',
        name: 'Electric Toothbrush USB Rechargeable Waterproof 5 Modes',
        price: 9.99,
        originalPrice: 19.99,
        discount: 50,
        stock: 72,
        image: '/images/products_prod_14.jpg',
        shipping: 'Free',
        sold: '4k+',
        rating: 4.7
      },
      {
        id: 'prod_15',
        name: 'Kitchen Vegetable Chopper Cutter Slicer Multifunctional Peeler',
        price: 13.99,
        originalPrice: 27.99,
        discount: 50,
        stock: 41,
        image: '/images/products_prod_15.jpg',
        shipping: 'Free',
        sold: '9k+',
        rating: 4.6
      },
      {
        id: 'prod_16',
        name: 'Baby Girl Dress Princess Party Birthday Wedding Clothes',
        price: 16.99,
        originalPrice: 32.99,
        discount: 49,
        stock: 53,
        image: '/images/products_prod_16.jpg',
        shipping: '$3.00',
        sold: '1.5k+',
        rating: 4.8
      },
      {
        id: 'prod_17',
        name: 'Pet Dog Bed Sofa Soft Fleece Warm Cat House',
        price: 19.99,
        originalPrice: 39.99,
        discount: 50,
        stock: 38,
        image: '/images/products_prod_17.jpg',
        shipping: 'Free',
        sold: '5k+',
        rating: 4.7
      },
      {
        id: 'prod_18',
        name: 'Canvas Backpack School Bags for Teenage Girls Women',
        price: 17.50,
        originalPrice: 35.00,
        discount: 50,
        stock: 66,
        image: '/images/products_prod_18.jpg',
        shipping: 'Free',
        sold: '7k+',
        rating: 4.6
      }
    ],

    // Default cart items
    cart_items: [
      {
        productId: 'prod_1',
        quantity: 1,
        selectedSku: '256GB, Black'
      },
      {
        productId: 'prod_2',
        quantity: 2,
        selectedSku: 'Black, L'
      },
      {
        productId: 'prod_5',
        quantity: 1,
        selectedSku: 'Silver, M'
      },
      {
        productId: 'prod_9',
        quantity: 3,
        selectedSku: 'Black'
      },
      {
        productId: 'prod_11',
        quantity: 1,
        selectedSku: '5m RGB'
      },
      {
        productId: 'prod_13',
        quantity: 2,
        selectedSku: 'Black Frame'
      }
    ]
  }),
  persist: {
    storage: sessionStorage,
  },
})