import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Accounts Data
  const accounts = ref([
    { id: 'acc_1', name: 'Main Account', currency: 'USD', balance: 5420.50, type: 'Checking', image: '/images/accounts_acc_1.jpg' },
    { id: 'acc_2', name: 'Savings Vault', currency: 'USD', balance: 12500.00, type: 'Savings', image: '/images/accounts_acc_2.jpg' },
    { id: 'acc_3', name: 'Travel Fund', currency: 'EUR', balance: 2100.75, type: 'Checking', image: '/images/accounts_acc_3.jpg' },
    { id: 'acc_4', name: 'Daily Expenses', currency: 'GBP', balance: 450.20, type: 'Checking', image: '/images/accounts_acc_4.jpg' },
    { id: 'acc_5', name: 'Investment Cash', currency: 'USD', balance: 0.00, type: 'Investment', image: '/images/accounts_acc_5.jpg' },
    { id: 'acc_6', name: 'Joint Account', currency: 'USD', balance: 3200.00, type: 'Joint', image: '/images/accounts_acc_6.jpg' },
    { id: 'acc_7', name: 'Business Expense', currency: 'USD', balance: 890.10, type: 'Business', image: '/images/accounts_acc_7.jpg' },
    { id: 'acc_8', name: 'Euro Savings', currency: 'EUR', balance: 5000.00, type: 'Savings', image: '/images/accounts_acc_8.jpg' },
    { id: 'acc_9', name: 'Kids Allowance', currency: 'USD', balance: 150.00, type: 'Checking', image: '/images/accounts_acc_9.jpg' },
    { id: 'acc_10', name: 'Crypto Wallet', currency: 'BTC', balance: 0.45, type: 'Crypto', image: '/images/accounts_acc_10.jpg' },
    { id: 'acc_11', name: 'Yen Trading', currency: 'JPY', balance: 150000, type: 'Trading', image: '/images/accounts_acc_11.jpg' },
    { id: 'acc_12', name: 'Emergency Fund', currency: 'USD', balance: 10000.00, type: 'Savings', image: '/images/accounts_acc_12.jpg' },
    { id: 'acc_13', name: 'Freelance Income', currency: 'USD', balance: 2300.50, type: 'Business', image: '/images/accounts_acc_13.jpg' },
    { id: 'acc_14', name: 'Shopping', currency: 'USD', balance: 300.00, type: 'Checking', image: '/images/accounts_acc_14.jpg' },
    { id: 'acc_15', name: 'Swiss Francs', currency: 'CHF', balance: 1200.00, type: 'Checking', image: '/images/accounts_acc_15.jpg' },
    { id: 'acc_16', name: 'Gold Reserve', currency: 'XAU', balance: 5.2, type: 'Commodity', image: '/images/accounts_acc_16.jpg' }
  ])

  // Beneficiaries Data
  const beneficiaries = ref([
    { id: 'ben_1', name: 'Alice Smith', accountNumber: '**** 1234', bank: 'Chase', isFavorite: true, image: '/images/beneficiaries_ben_1.jpg' },
    { id: 'ben_2', name: 'Bob Jones', accountNumber: '**** 5678', bank: 'BoA', isFavorite: true, image: '/images/beneficiaries_ben_2.jpg' },
    { id: 'ben_3', name: 'Charlie Brown', accountNumber: '**** 9012', bank: 'Wells Fargo', isFavorite: false, image: '/images/beneficiaries_ben_3.jpg' },
    { id: 'ben_4', name: 'Diana Prince', accountNumber: '**** 3456', bank: 'Citi', isFavorite: true, image: '/images/beneficiaries_ben_4.jpg' },
    { id: 'ben_5', name: 'Evan Wright', accountNumber: '**** 7890', bank: 'HSBC', isFavorite: false, image: '/images/beneficiaries_ben_5.jpg' },
    { id: 'ben_6', name: 'Fiona Green', accountNumber: '**** 1122', bank: 'Barclays', isFavorite: false, image: '/images/beneficiaries_ben_6.jpg' },
    { id: 'ben_7', name: 'George Hall', accountNumber: '**** 3344', bank: 'Chase', isFavorite: true, image: '/images/beneficiaries_ben_7.jpg' },
    { id: 'ben_8', name: 'Hannah Lee', accountNumber: '**** 5566', bank: 'US Bank', isFavorite: false, image: '/images/beneficiaries_ben_8.jpg' },
    { id: 'ben_9', name: 'Ian Scott', accountNumber: '**** 7788', bank: 'PNC', isFavorite: false, image: '/images/beneficiaries_ben_9.jpg' },
    { id: 'ben_10', name: 'Julia Roberts', accountNumber: '**** 9900', bank: 'Capital One', isFavorite: true, image: '/images/beneficiaries_ben_10.jpg' },
    { id: 'ben_11', name: 'Kevin Hart', accountNumber: '**** 2233', bank: 'TD Bank', isFavorite: false, image: '/images/beneficiaries_ben_11.jpg' },
    { id: 'ben_12', name: 'Laura Croft', accountNumber: '**** 4455', bank: 'Santander', isFavorite: true, image: '/images/beneficiaries_ben_12.jpg' },
    { id: 'ben_13', name: 'Mike Ross', accountNumber: '**** 6677', bank: 'Chase', isFavorite: false, image: '/images/beneficiaries_ben_13.jpg' },
    { id: 'ben_14', name: 'Nina Simone', accountNumber: '**** 8899', bank: 'BoA', isFavorite: true, image: '/images/beneficiaries_ben_14.jpg' },
    { id: 'ben_15', name: 'Oscar Wilde', accountNumber: '**** 0011', bank: 'Wells Fargo', isFavorite: false, image: '/images/beneficiaries_ben_15.jpg' }
  ])

  // Exchange Pairs Data
  const exchangePairs = ref([
    { id: 'pair_1', from: 'USD', to: 'EUR', rate: 0.92, change: 0.5, isMajor: true, image: '/images/exchangePairs_pair_1.jpg' },
    { id: 'pair_2', from: 'USD', to: 'GBP', rate: 0.79, change: -0.2, isMajor: true, image: '/images/exchangePairs_pair_2.jpg' },
    { id: 'pair_3', from: 'EUR', to: 'USD', rate: 1.09, change: 0.5, isMajor: true, image: '/images/exchangePairs_pair_3.jpg' },
    { id: 'pair_4', from: 'GBP', to: 'USD', rate: 1.27, change: -0.2, isMajor: true, image: '/images/exchangePairs_pair_4.jpg' },
    { id: 'pair_5', from: 'USD', to: 'JPY', rate: 148.50, change: 1.2, isMajor: true, image: '/images/exchangePairs_pair_5.jpg' },
    { id: 'pair_6', from: 'USD', to: 'CAD', rate: 1.35, change: 0.1, isMajor: true, image: '/images/exchangePairs_pair_6.jpg' },
    { id: 'pair_7', from: 'USD', to: 'AUD', rate: 1.52, change: -0.3, isMajor: true, image: '/images/exchangePairs_pair_7.jpg' },
    { id: 'pair_8', from: 'USD', to: 'CHF', rate: 0.88, change: 0.0, isMajor: true, image: '/images/exchangePairs_pair_8.jpg' },
    { id: 'pair_9', from: 'EUR', to: 'GBP', rate: 0.85, change: -0.1, isMajor: true, image: '/images/exchangePairs_pair_9.jpg' },
    { id: 'pair_10', from: 'EUR', to: 'JPY', rate: 161.20, change: 0.8, isMajor: false, image: '/images/exchangePairs_pair_10.jpg' },
    { id: 'pair_11', from: 'GBP', to: 'EUR', rate: 1.17, change: 0.1, isMajor: true, image: '/images/exchangePairs_pair_11.jpg' },
    { id: 'pair_12', from: 'USD', to: 'CNY', rate: 7.19, change: 0.0, isMajor: false, image: '/images/exchangePairs_pair_12.jpg' },
    { id: 'pair_13', from: 'USD', to: 'SGD', rate: 1.34, change: 0.2, isMajor: false, image: '/images/exchangePairs_pair_13.jpg' },
    { id: 'pair_14', from: 'USD', to: 'NZD', rate: 1.63, change: -0.4, isMajor: false, image: '/images/exchangePairs_pair_14.jpg' },
    { id: 'pair_15', from: 'USD', to: 'MXN', rate: 17.05, change: 0.6, isMajor: false, image: '/images/exchangePairs_pair_15.jpg' },
    { id: 'pair_16', from: 'BTC', to: 'USD', rate: 52000.00, change: 2.5, isMajor: true, image: '/images/exchangePairs_pair_16.jpg' }
  ])

  // Cards Data
  const cards = ref([
    { id: 'card_1', nickname: 'Main Physical', type: 'Physical', last4: '4242', expiry: '12/26', status: 'Active', scheme: 'Visa', limit: 5000, created: '2023-01-15', image: '/images/cards_card_1.jpg' },
    { id: 'card_2', nickname: 'Online Shopping', type: 'Virtual', last4: '9876', expiry: '09/25', status: 'Active', scheme: 'Mastercard', limit: 1000, created: '2023-03-10', image: '/images/cards_card_2.jpg' },
    { id: 'card_3', nickname: 'Travel Card', type: 'Physical', last4: '5544', expiry: '11/27', status: 'Frozen', scheme: 'Visa', limit: 10000, created: '2022-11-05', image: '/images/cards_card_3.jpg' },
    { id: 'card_4', nickname: 'Subscription', type: 'Virtual', last4: '1122', expiry: '01/28', status: 'Active', scheme: 'Mastercard', limit: 200, created: '2023-06-20', image: '/images/cards_card_4.jpg' },
    { id: 'card_5', nickname: 'Kids Card', type: 'Physical', last4: '3344', expiry: '05/29', status: 'Active', scheme: 'Visa', limit: 100, created: '2024-01-01', image: '/images/cards_card_5.jpg' },
    { id: 'card_6', nickname: 'Backup', type: 'Physical', last4: '7788', expiry: '08/26', status: 'Inactive', scheme: 'Visa', limit: 2000, created: '2021-08-15', image: '/images/cards_card_6.jpg' },
    { id: 'card_7', nickname: 'Disposable', type: 'Virtual', last4: '9900', expiry: '02/24', status: 'Active', scheme: 'Visa', limit: 500, created: '2024-02-10', image: '/images/cards_card_7.jpg' },
    { id: 'card_8', nickname: 'Business', type: 'Physical', last4: '2211', expiry: '10/28', status: 'Active', scheme: 'Mastercard', limit: 20000, created: '2023-09-01', image: '/images/cards_card_8.jpg' },
    { id: 'card_9', nickname: 'Joint Card', type: 'Physical', last4: '6655', expiry: '04/27', status: 'Active', scheme: 'Visa', limit: 3000, created: '2022-04-20', image: '/images/cards_card_9.jpg' },
    { id: 'card_10', nickname: 'Spare', type: 'Virtual', last4: '0099', expiry: '12/25', status: 'Frozen', scheme: 'Visa', limit: 1000, created: '2022-12-12', image: '/images/cards_card_10.jpg' }
  ])

  // Topup Methods
  const topupMethods = ref([
    { id: 'method_1', name: 'Chase Bank **** 1234', type: 'Bank Link', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_1.jpg' },
    { id: 'method_2', name: 'Apple Pay', type: 'Digital Wallet', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_2.jpg' },
    { id: 'method_3', name: 'Google Pay', type: 'Digital Wallet', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_3.jpg' },
    { id: 'method_4', name: 'Debit Card **** 5678', type: 'Card', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_4.jpg' },
    { id: 'method_5', name: 'Direct Deposit', type: 'Bank Transfer', isLinked: false, processingTime: '1-3 Days', image: '/images/topupMethods_method_5.jpg' },
    { id: 'method_6', name: 'Wire Transfer', type: 'Bank Transfer', isLinked: false, processingTime: '1-2 Days', image: '/images/topupMethods_method_6.jpg' },
    { id: 'method_7', name: 'Wells Fargo **** 9012', type: 'Bank Link', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_7.jpg' },
    { id: 'method_8', name: 'Citi Bank **** 3456', type: 'Bank Link', isLinked: false, processingTime: 'Instant', image: '/images/topupMethods_method_8.jpg' },
    { id: 'method_9', name: 'PayPal', type: 'Digital Wallet', isLinked: false, processingTime: 'Instant', image: '/images/topupMethods_method_9.jpg' },
    { id: 'method_10', name: 'Cash Deposit', type: 'Cash', isLinked: false, processingTime: 'Instant', image: '/images/topupMethods_method_10.jpg' },
    { id: 'method_11', name: 'Crypto Transfer', type: 'Crypto', isLinked: true, processingTime: '10-30 Mins', image: '/images/topupMethods_method_11.jpg' },
    { id: 'method_12', name: 'Credit Card **** 1122', type: 'Card', isLinked: true, processingTime: 'Instant', image: '/images/topupMethods_method_12.jpg' }
  ])

  return {
    accounts,
    beneficiaries,
    exchangePairs,
    cards,
    topupMethods
  }
}, {
  persist: {
    storage: sessionStorage
  }
})