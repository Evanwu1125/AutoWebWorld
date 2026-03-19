import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  
  // Electronics Products
  const electronics = ref([
    { id: 'elec_001', name: 'Ultra HD 4K Smart TV 55"', brand: 'Samsung', price: 498.00, rating: 4.5, image: '/images/electronics_elec_001.jpg', description: 'Crystal clear 4K resolution with smart features.' },
    { id: 'elec_002', name: 'Noise Cancelling Headphones', brand: 'Sony', price: 248.00, rating: 4.8, image: '/images/electronics_elec_002.jpg', description: 'Industry leading noise cancellation.' },
    { id: 'elec_003', name: 'Gaming Laptop 15.6"', brand: 'HP', price: 899.00, rating: 4.2, image: '/images/electronics_elec_003.jpg', description: 'Powerful performance for gaming and work.' },
    { id: 'elec_004', name: 'Wireless Earbuds', brand: 'Apple', price: 129.00, rating: 4.6, image: '/images/electronics_elec_004.jpg', description: 'Spatial audio with dynamic head tracking.' },
    { id: 'elec_005', name: 'Smart Watch Series 7', brand: 'Apple', price: 329.00, rating: 4.7, image: '/images/electronics_elec_005.jpg', description: 'Advanced health features and fitness tracking.' },
    { id: 'elec_006', name: 'Bluetooth Speaker', brand: 'JBL', price: 89.95, rating: 4.4, image: '/images/electronics_elec_006.jpg', description: 'Portable waterproof speaker with big sound.' },
    { id: 'elec_007', name: 'Tablet 10.2"', brand: 'Apple', price: 329.00, rating: 4.8, image: '/images/electronics_elec_007.jpg', description: 'Versatile and easy to use.' },
    { id: 'elec_008', name: 'Digital Camera Mirrorless', brand: 'Canon', price: 649.00, rating: 4.5, image: '/images/electronics_elec_008.jpg', description: 'Capture stunning photos and 4K video.' },
    { id: 'elec_009', name: 'Smartphone Galaxy S22', brand: 'Samsung', price: 799.00, rating: 4.3, image: '/images/electronics_elec_009.jpg', description: 'Nightography camera and all-day battery.' },
    { id: 'elec_010', name: 'Soundbar with Subwoofer', brand: 'Vizio', price: 148.00, rating: 4.1, image: '/images/electronics_elec_010.jpg', description: 'Immersive audio experience.' },
    { id: 'elec_011', name: 'External Hard Drive 2TB', brand: 'Seagate', price: 59.99, rating: 4.6, image: '/images/electronics_elec_011.jpg', description: 'Portable storage for your files.' },
    { id: 'elec_012', name: 'Wireless Mouse', brand: 'Logitech', price: 29.99, rating: 4.7, image: '/images/electronics_elec_012.jpg', description: 'Ergonomic design and long battery life.' },
    { id: 'elec_013', name: 'Mechanical Keyboard', brand: 'Corsair', price: 129.99, rating: 4.8, image: '/images/electronics_elec_013.jpg', description: 'High-performance gaming keyboard.' },
    { id: 'elec_014', name: 'Action Camera 4K', brand: 'GoPro', price: 349.00, rating: 4.5, image: '/images/electronics_elec_014.jpg', description: 'Rugged and waterproof camera.' },
    { id: 'elec_015', name: 'Smart Home Hub', brand: 'Google', price: 99.00, rating: 4.2, image: '/images/electronics_elec_015.jpg', description: 'Control your smart home devices.' },
    { id: 'elec_016', name: 'Drone with Camera', brand: 'DJI', price: 449.00, rating: 4.6, image: '/images/electronics_elec_016.jpg', description: 'Capture aerial shots easily.' },
    { id: 'elec_017', name: 'E-Reader Paperwhite', brand: 'Amazon', price: 139.99, rating: 4.8, image: '/images/electronics_elec_017.jpg', description: 'Read comfortably with adjustable warm light.' },
    { id: 'elec_018', name: 'Streaming Stick 4K', brand: 'Roku', price: 39.00, rating: 4.4, image: '/images/electronics_elec_018.jpg', description: 'Stream your favorite shows in 4K.' },
    { id: 'elec_019', name: 'Wi-Fi 6 Router', brand: 'Netgear', price: 149.99, rating: 4.3, image: '/images/electronics_elec_019.jpg', description: 'Fast and reliable internet coverage.' },
    { id: 'elec_020', name: 'Instant Photo Printer', brand: 'HP', price: 89.00, rating: 4.1, image: '/images/electronics_elec_020.jpg', description: 'Print photos directly from your phone.' }
  ])

  // Grocery Products
  const groceries = ref([
    { id: 'groc_001', name: 'Organic Bananas', type: 'organic', price: 0.58, unit: 'lb', image: '/images/groceries_groc_001.jpg' },
    { id: 'groc_002', name: 'Gala Apples', type: 'regular', price: 1.28, unit: 'lb', image: '/images/groceries_groc_002.jpg' },
    { id: 'groc_003', name: 'Whole Milk', type: 'regular', price: 3.48, unit: 'gal', image: '/images/groceries_groc_003.jpg' },
    { id: 'groc_004', name: 'Organic Eggs Large Brown', type: 'organic', price: 4.98, unit: 'doz', image: '/images/groceries_groc_004.jpg' },
    { id: 'groc_005', name: 'Sourdough Bread', type: 'regular', price: 2.98, unit: 'loaf', image: '/images/groceries_groc_005.jpg' },
    { id: 'groc_006', name: 'Avocados', type: 'regular', price: 0.98, unit: 'each', image: '/images/groceries_groc_006.jpg' },
    { id: 'groc_007', name: 'Organic Spinach', type: 'organic', price: 2.98, unit: 'bag', image: '/images/groceries_groc_007.jpg' },
    { id: 'groc_008', name: 'Chicken Breast Boneless', type: 'regular', price: 4.98, unit: 'lb', image: '/images/groceries_groc_008.jpg' },
    { id: 'groc_009', name: 'Ground Beef 80/20', type: 'regular', price: 5.48, unit: 'lb', image: '/images/groceries_groc_009.jpg' },
    { id: 'groc_010', name: 'Cheddar Cheese Block', type: 'regular', price: 3.98, unit: 'block', image: '/images/groceries_groc_010.jpg' },
    { id: 'groc_011', name: 'Organic Strawberries', type: 'organic', price: 3.98, unit: 'lb', image: '/images/groceries_groc_011.jpg' },
    { id: 'groc_012', name: 'Orange Juice Pulp Free', type: 'regular', price: 3.78, unit: 'bottle', image: '/images/groceries_groc_012.jpg' },
    { id: 'groc_013', name: 'Greek Yogurt Vanilla', type: 'regular', price: 4.48, unit: 'tub', image: '/images/groceries_groc_013.jpg' },
    { id: 'groc_014', name: 'Organic Carrots', type: 'organic', price: 1.98, unit: 'bunch', image: '/images/groceries_groc_014.jpg' },
    { id: 'groc_015', name: 'Russet Potatoes 5lb', type: 'regular', price: 2.98, unit: 'bag', image: '/images/groceries_groc_015.jpg' },
    { id: 'groc_016', name: 'Pasta Spaghetti', type: 'regular', price: 0.98, unit: 'box', image: '/images/groceries_groc_016.jpg' },
    { id: 'groc_017', name: 'Tomato Sauce', type: 'regular', price: 1.48, unit: 'jar', image: '/images/groceries_groc_017.jpg' },
    { id: 'groc_018', name: 'Organic Coffee Beans', type: 'organic', price: 12.98, unit: 'bag', image: '/images/groceries_groc_018.jpg' },
    { id: 'groc_019', name: 'Almond Milk Unsweetened', type: 'regular', price: 2.98, unit: 'carton', image: '/images/groceries_groc_019.jpg' },
    { id: 'groc_020', name: 'Butter Salted', type: 'regular', price: 3.98, unit: 'box', image: '/images/groceries_groc_020.jpg' }
  ])

  // Orders
  const orders = ref([
    { id: 'ord_1001', date: '2025-10-01', total: 156.45, status: 'delivered', items: ['elec_004', 'elec_012'] },
    { id: 'ord_1002', date: '2025-10-05', total: 45.20, status: 'delivered', items: ['groc_001', 'groc_003', 'groc_008'] },
    { id: 'ord_1003', date: '2025-10-10', total: 899.00, status: 'shipped', items: ['elec_003'] },
    { id: 'ord_1004', date: '2025-10-12', total: 248.00, status: 'processing', items: ['elec_002'] },
    { id: 'ord_1005', date: '2025-09-15', total: 32.50, status: 'delivered', items: ['groc_010', 'groc_016'] },
    { id: 'ord_1006', date: '2025-09-20', total: 498.00, status: 'delivered', items: ['elec_001'] },
    { id: 'ord_1007', date: '2025-09-25', total: 129.99, status: 'delivered', items: ['elec_013'] },
    { id: 'ord_1008', date: '2025-10-15', total: 649.00, status: 'cancelled', items: ['elec_008'] },
    { id: 'ord_1009', date: '2025-08-30', total: 19.95, status: 'delivered', items: ['groc_002', 'groc_006'] },
    { id: 'ord_1010', date: '2025-10-18', total: 799.00, status: 'processing', items: ['elec_009'] },
    { id: 'ord_1011', date: '2025-07-12', total: 59.99, status: 'delivered', items: ['elec_011'] },
    { id: 'ord_1012', date: '2025-10-19', total: 89.95, status: 'shipped', items: ['elec_006'] },
    { id: 'ord_1013', date: '2025-06-05', total: 148.00, status: 'delivered', items: ['elec_010'] },
    { id: 'ord_1014', date: '2025-10-20', total: 29.98, status: 'processing', items: ['groc_004', 'groc_018'] },
    { id: 'ord_1015', date: '2025-05-20', total: 329.00, status: 'delivered', items: ['elec_005'] }
  ])

  // Pickup Stores
  const stores = ref([
    { id: 'store_123', name: 'Supercenter #123 (Downtown)', address: '123 Market St' },
    { id: 'store_456', name: 'Neighborhood Market #456 (Uptown)', address: '456 Hill Rd' }
  ])

  return {
    electronics,
    groceries,
    orders,
    stores
  }
}, {
  persist: {
    storage: sessionStorage
  }
})