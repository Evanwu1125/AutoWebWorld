import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useDataStore = defineStore('data', () => {
  // --- Electronics Products ---
  const electronics = ref([
    { id: 'e1', name: 'iPhone 15 Pro Max', price: 1199, brand: 'Apple', image: '/images/electronics_e1.jpg', rating: 4.9, sales: 5000, tags: ['Self-Operated', 'Flash Sale'] },
    { id: 'e2', name: 'Samsung Galaxy S24 Ultra', price: 1299, brand: 'Samsung', image: '/images/electronics_e2.jpg', rating: 4.8, sales: 3000, tags: ['Self-Operated'] },
    { id: 'e3', name: 'MacBook Pro 16 M3', price: 2499, brand: 'Apple', image: '/images/electronics_e3.jpg', rating: 4.9, sales: 1000, tags: ['Self-Operated'] },
    { id: 'e4', name: 'Dell XPS 15', price: 1899, brand: 'Dell', image: '/images/electronics_e4.jpg', rating: 4.6, sales: 800, tags: [] },
    { id: 'e5', name: 'Sony WH-1000XM5', price: 348, brand: 'Sony', image: '/images/electronics_e5.jpg', rating: 4.7, sales: 12000, tags: ['Self-Operated'] },
    { id: 'e6', name: 'iPad Air 5', price: 599, brand: 'Apple', image: '/images/electronics_e6.jpg', rating: 4.8, sales: 6000, tags: [] },
    { id: 'e7', name: 'Nintendo Switch OLED', price: 349, brand: 'Nintendo', image: '/images/electronics_e7.jpg', rating: 4.9, sales: 20000, tags: ['Flash Sale'] },
    { id: 'e8', name: 'Logitech MX Master 3S', price: 99, brand: 'Logitech', image: '/images/electronics_e8.jpg', rating: 4.8, sales: 4000, tags: [] },
    { id: 'e9', name: 'Asus ROG Zephyrus', price: 1999, brand: 'Asus', image: '/images/electronics_e9.jpg', rating: 4.7, sales: 500, tags: ['Self-Operated'] },
    { id: 'e10', name: 'GoPro Hero 12', price: 399, brand: 'GoPro', image: '/images/electronics_e10.jpg', rating: 4.6, sales: 1500, tags: [] },
    { id: 'e11', name: 'Kindle Paperwhite', price: 139, brand: 'Amazon', image: '/images/electronics_e11.jpg', rating: 4.8, sales: 8000, tags: ['Self-Operated'] },
    { id: 'e12', name: 'Bose QuietComfort Ultra', price: 429, brand: 'Bose', image: '/images/electronics_e12.jpg', rating: 4.7, sales: 2500, tags: [] },
    { id: 'e13', name: 'Canon EOS R6', price: 2299, brand: 'Canon', image: '/images/electronics_e13.jpg', rating: 4.9, sales: 300, tags: ['Self-Operated'] },
    { id: 'e14', name: 'DJI Mini 4 Pro', price: 759, brand: 'DJI', image: '/images/electronics_e14.jpg', rating: 4.9, sales: 1000, tags: [] },
    { id: 'e15', name: 'Apple Watch Series 9', price: 399, brand: 'Apple', image: '/images/electronics_e15.jpg', rating: 4.7, sales: 9000, tags: ['Self-Operated'] },
    { id: 'e16', name: 'Samsung Odyssey G9', price: 1199, brand: 'Samsung', image: '/images/electronics_e16.jpg', rating: 4.6, sales: 400, tags: ['Flash Sale'] },
  ]);

  // --- Supermarket Products ---
  const supermarket = ref([
    { id: 's1', name: 'Organic Bananas', price: 2.99, category: 'Fresh', image: '/images/supermarket_s1.jpg', rating: 4.8, sales: 10000, fresh: true },
    { id: 's2', name: 'Whole Milk 1 Gallon', price: 3.99, category: 'Dairy', image: '/images/supermarket_s2.jpg', rating: 4.7, sales: 8000, fresh: true },
    { id: 's3', name: 'Sourdough Bread', price: 4.50, category: 'Bakery', image: '/images/supermarket_s3.jpg', rating: 4.9, sales: 5000, fresh: true },
    { id: 's4', name: 'Free Range Eggs (12)', price: 5.99, category: 'Dairy', image: '/images/supermarket_s4.jpg', rating: 4.8, sales: 6000, fresh: true },
    { id: 's5', name: 'Chicken Breast (1lb)', price: 6.99, category: 'Meat', image: '/images/supermarket_s5.jpg', rating: 4.6, sales: 4000, fresh: true },
    { id: 's6', name: 'Atlantic Salmon', price: 12.99, category: 'Seafood', image: '/images/supermarket_s6.jpg', rating: 4.8, sales: 2000, fresh: true },
    { id: 's7', name: 'Avocados (Bag)', price: 4.99, category: 'Fresh', image: '/images/supermarket_s7.jpg', rating: 4.7, sales: 7000, fresh: true },
    { id: 's8', name: 'Cheddar Cheese', price: 5.49, category: 'Dairy', image: '/images/supermarket_s8.jpg', rating: 4.8, sales: 3000, fresh: true },
    { id: 's9', name: 'Orange Juice', price: 3.99, category: 'Beverage', image: '/images/supermarket_s9.jpg', rating: 4.6, sales: 4500, fresh: true },
    { id: 's10', name: 'Greek Yogurt', price: 1.29, category: 'Dairy', image: '/images/supermarket_s10.jpg', rating: 4.7, sales: 9000, fresh: true },
    { id: 's11', name: 'Strawberries (1lb)', price: 3.99, category: 'Fresh', image: '/images/supermarket_s11.jpg', rating: 4.5, sales: 6000, fresh: true },
    { id: 's12', name: 'Ground Beef', price: 5.99, category: 'Meat', image: '/images/supermarket_s12.jpg', rating: 4.7, sales: 3500, fresh: true },
    { id: 's13', name: 'Toilet Paper (12 Rolls)', price: 14.99, category: 'Household', image: '/images/supermarket_s13.jpg', rating: 4.8, sales: 5000, fresh: false },
    { id: 's14', name: 'Laundry Detergent', price: 12.99, category: 'Household', image: '/images/supermarket_s14.jpg', rating: 4.8, sales: 4000, fresh: false },
    { id: 's15', name: 'Olive Oil', price: 9.99, category: 'Pantry', image: '/images/supermarket_s15.jpg', rating: 4.9, sales: 2000, fresh: false },
    { id: 's16', name: 'Pasta (Spaghetti)', price: 1.99, category: 'Pantry', image: '/images/supermarket_s16.jpg', rating: 4.7, sales: 8000, fresh: false },
  ]);

  // --- Users ---
  const users = ref([
    { id: 'u1', username: 'jd_fan', name: 'John Doe', avatar: '/images/Avatar.jpg' },
    { id: 'u2', username: 'shopaholic', name: 'Jane Smith', avatar: '/images/User.jpg' }
  ]);

  // --- Addresses ---
  const addresses = ref([
    { id: 'a1', name: 'John Doe', detail: '123 Tech Park, Beijing', isDefault: true },
    { id: 'a2', name: 'John Office', detail: '456 Innovation Hub, Shanghai', isDefault: false },
    { id: 'a3', name: 'Parents', detail: '789 Garden Villa, Guangzhou', isDefault: false }
  ]);

  // --- Payment Methods ---
  const paymentMethods = ref([
    { id: 'p1', cardHolder: 'JOHN DOE', last4: '4242', type: 'Credit' },
    { id: 'p2', cardHolder: 'JOHN DOE', last4: '8888', type: 'Debit' }
  ]);

  // --- Cart ---
  const cart = ref([
    { id: 'c1', productId: 'e1', name: 'iPhone 15 Pro Max', price: 1199, quantity: 1, image: '/images/cart_c1.jpg' },
    { id: 'c2', productId: 's1', name: 'Organic Bananas', price: 2.99, quantity: 2, image: '/images/cart_c2.jpg' }
  ]);

  // --- Orders ---
  const orders = ref([
    { id: 'o1', date: '2025-11-01', total: 1205, status: 'Delivered', items: [
      { name: 'iPhone 15 Pro Max', image: '/images/orders_o1.jpg', price: 1199 },
      { name: 'Screen Protector', image: '/images/ScreenProtector.jpg', price: 6 }
    ]},
    { id: 'o2', date: '2025-10-25', total: 45, status: 'Shipped', items: [
      { name: 'Organic Bananas', image: '/images/orders_o2.jpg', price: 3 },
      { name: 'Milk', image: '/images/Milk.jpg', price: 4 },
      { name: 'Eggs', image: '/images/Eggs.jpg', price: 6 }
    ]},
    { id: 'o3', date: '2025-10-10', total: 2499, status: 'Processing', items: [
      { name: 'MacBook Pro 16', image: '/images/orders_o3.jpg', price: 2499 }
    ]},
    { id: 'o4', date: '2025-09-30', total: 348, status: 'Completed', items: [
      { name: 'Sony WH-1000XM5', image: '/images/orders_o4.jpg', price: 348 }
    ]},
    { id: 'o5', date: '2025-09-15', total: 150, status: 'Cancelled', items: [
      { name: 'Kindle Paperwhite', image: '/images/orders_o5.jpg', price: 139 }
    ]}
  ]);

  // --- Reviews ---
  const reviews = ref([
    { id: 'r1', user: 'Alice', rating: 5, text: 'Amazing product!', date: '2025-12-01' },
    { id: 'r2', user: 'Bob', rating: 4, text: 'Good value for money.', date: '2025-11-28' },
    { id: 'r3', user: 'Charlie', rating: 5, text: 'Fast shipping by JD.', date: '2025-11-25' }
  ]);

  return {
    electronics,
    supermarket,
    users,
    addresses,
    paymentMethods,
    cart,
    orders,
    reviews
  };
}, {
  persist: {
    storage: sessionStorage,
  },
});