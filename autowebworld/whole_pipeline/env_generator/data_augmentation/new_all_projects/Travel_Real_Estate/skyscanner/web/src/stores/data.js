import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- Flights Data ---
  const flights = ref([
    {
      id: 'flight_1',
      airline: 'British Airways',
      origin: 'LHR',
      destination: 'JFK',
      price: 450,
      duration: '7h 55m',
      departure_time: '10:00',
      arrival_time: '12:55',
      stops: 0,
      image: '/images/flights_flight_1.jpg'
    },
    {
      id: 'flight_2',
      airline: 'Virgin Atlantic',
      origin: 'LHR',
      destination: 'JFK',
      price: 420,
      duration: '8h 05m',
      departure_time: '14:30',
      arrival_time: '17:35',
      stops: 0,
      image: '/images/flights_flight_2.jpg'
    },
    {
      id: 'flight_3',
      airline: 'American Airlines',
      origin: 'LHR',
      destination: 'JFK',
      price: 380,
      duration: '8h 15m',
      departure_time: '08:45',
      arrival_time: '11:30',
      stops: 1,
      image: '/images/flights_flight_3.jpg'
    },
    {
      id: 'flight_4',
      airline: 'Delta',
      origin: 'LHR',
      destination: 'JFK',
      price: 460,
      duration: '8h 00m',
      departure_time: '11:15',
      arrival_time: '14:15',
      stops: 0,
      image: '/images/flights_flight_4.jpg'
    },
    {
      id: 'flight_5',
      airline: 'United',
      origin: 'LHR',
      destination: 'JFK',
      price: 400,
      duration: '8h 20m',
      departure_time: '09:00',
      arrival_time: '12:20',
      stops: 1,
      image: '/images/flights_flight_5.jpg'
    },
    {
      id: 'flight_6',
      airline: 'Lufthansa',
      origin: 'LHR',
      destination: 'JFK',
      price: 390,
      duration: '10h 30m',
      departure_time: '07:30',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flights_flight_6.jpg'
    },
    {
      id: 'flight_7',
      airline: 'Air France',
      origin: 'LHR',
      destination: 'JFK',
      price: 395,
      duration: '9h 45m',
      departure_time: '08:00',
      arrival_time: '12:45',
      stops: 1,
      image: '/images/flights_flight_7.jpg'
    },
    {
      id: 'flight_8',
      airline: 'KLM',
      origin: 'LHR',
      destination: 'JFK',
      price: 410,
      duration: '10h 00m',
      departure_time: '06:45',
      arrival_time: '11:45',
      stops: 1,
      image: '/images/flights_flight_8.jpg'
    },
    {
      id: 'flight_9',
      airline: 'Aer Lingus',
      origin: 'LHR',
      destination: 'JFK',
      price: 350,
      duration: '9h 15m',
      departure_time: '09:30',
      arrival_time: '13:45',
      stops: 1,
      image: '/images/flights_flight_9.jpg'
    },
    {
      id: 'flight_10',
      airline: 'Norse Atlantic',
      origin: 'LGW',
      destination: 'JFK',
      price: 280,
      duration: '8h 10m',
      departure_time: '13:00',
      arrival_time: '16:10',
      stops: 0,
      image: '/images/flights_flight_10.jpg'
    },
    {
      id: 'flight_11',
      airline: 'JetBlue',
      origin: 'LHR',
      destination: 'JFK',
      price: 430,
      duration: '8h 15m',
      departure_time: '10:45',
      arrival_time: '14:00',
      stops: 0,
      image: '/images/flights_flight_11.jpg'
    },
    {
      id: 'flight_12',
      airline: 'Finnair',
      origin: 'LHR',
      destination: 'JFK',
      price: 440,
      duration: '11h 00m',
      departure_time: '07:00',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flights_flight_12.jpg'
    },
    {
      id: 'flight_13',
      airline: 'Iberia',
      origin: 'LHR',
      destination: 'JFK',
      price: 415,
      duration: '11h 30m',
      departure_time: '06:30',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flights_flight_13.jpg'
    },
    {
      id: 'flight_14',
      airline: 'Swiss',
      origin: 'LHR',
      destination: 'JFK',
      price: 425,
      duration: '10h 45m',
      departure_time: '08:30',
      arrival_time: '14:15',
      stops: 1,
      image: '/images/flights_flight_14.jpg'
    },
    {
      id: 'flight_15',
      airline: 'TAP Air Portugal',
      origin: 'LHR',
      destination: 'JFK',
      price: 360,
      duration: '12h 00m',
      departure_time: '06:00',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flights_flight_15.jpg'
    },
    {
      id: 'flight_16',
      airline: 'Emirates',
      origin: 'LHR',
      destination: 'DXB',
      price: 550,
      duration: '7h 00m',
      departure_time: '14:00',
      arrival_time: '00:00',
      stops: 0,
      image: '/images/flights_flight_16.jpg'
    },
    {
      id: 'flight_17',
      airline: 'Qatar Airways',
      origin: 'LHR',
      destination: 'DOH',
      price: 530,
      duration: '6h 45m',
      departure_time: '15:00',
      arrival_time: '00:45',
      stops: 0,
      image: '/images/flights_flight_17.jpg'
    }
  ])

  // --- Multi-City Flights Data ---
  const multiFlights = ref([
    {
      id: 'multi_1',
      legs: ['LHR -> JFK', 'JFK -> LAX'],
      airline: 'American Airlines',
      price: 650,
      total_duration: '16h 00m',
      stops: 1,
      image: '/images/multiFlights_multi_1.jpg'
    },
    {
      id: 'multi_2',
      legs: ['LHR -> JFK', 'JFK -> SFO'],
      airline: 'Delta',
      price: 680,
      total_duration: '17h 00m',
      stops: 1,
      image: '/images/multiFlights_multi_2.jpg'
    },
    {
      id: 'multi_3',
      legs: ['LHR -> DXB', 'DXB -> SYD'],
      airline: 'Emirates',
      price: 1200,
      total_duration: '22h 00m',
      stops: 1,
      image: '/images/multiFlights_multi_3.jpg'
    },
    {
      id: 'multi_4',
      legs: ['LHR -> SIN', 'SIN -> SYD'],
      airline: 'Singapore Airlines',
      price: 1150,
      total_duration: '21h 30m',
      stops: 1,
      image: '/images/multiFlights_multi_4.jpg'
    },
    {
      id: 'multi_5',
      legs: ['LHR -> HND', 'HND -> CTS'],
      airline: 'JAL',
      price: 900,
      total_duration: '15h 00m',
      stops: 1,
      image: '/images/multiFlights_multi_5.jpg'
    }
  ])

  // --- Hotels Data ---
  const hotels = ref([
    {
      id: 'hotel_1',
      name: 'The Ritz London',
      location: 'London, UK',
      price: 850,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotels_hotel_1.jpg'
    },
    {
      id: 'hotel_2',
      name: 'The Savoy',
      location: 'London, UK',
      price: 750,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_2.jpg'
    },
    {
      id: 'hotel_3',
      name: 'Shangri-La The Shard',
      location: 'London, UK',
      price: 900,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_3.jpg'
    },
    {
      id: 'hotel_4',
      name: 'Claridge\'s',
      location: 'London, UK',
      price: 800,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotels_hotel_4.jpg'
    },
    {
      id: 'hotel_5',
      name: 'The Langham',
      location: 'London, UK',
      price: 600,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_5.jpg'
    },
    {
      id: 'hotel_6',
      name: 'Hilton London Metropole',
      location: 'London, UK',
      price: 250,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotels_hotel_6.jpg'
    },
    {
      id: 'hotel_7',
      name: 'Marriott County Hall',
      location: 'London, UK',
      price: 450,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_7.jpg'
    },
    {
      id: 'hotel_8',
      name: 'Park Plaza Westminster Bridge',
      location: 'London, UK',
      price: 300,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotels_hotel_8.jpg'
    },
    {
      id: 'hotel_9',
      name: 'Premier Inn London County Hall',
      location: 'London, UK',
      price: 150,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotels_hotel_9.jpg'
    },
    {
      id: 'hotel_10',
      name: 'Travelodge London Central',
      location: 'London, UK',
      price: 120,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotels_hotel_10.jpg'
    },
    {
      id: 'hotel_11',
      name: 'Plaza Hotel',
      location: 'New York, USA',
      price: 950,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_11.jpg'
    },
    {
      id: 'hotel_12',
      name: 'The St. Regis New York',
      location: 'New York, USA',
      price: 1100,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotels_hotel_12.jpg'
    },
    {
      id: 'hotel_13',
      name: 'Four Seasons Hotel New York',
      location: 'New York, USA',
      price: 1200,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotels_hotel_13.jpg'
    },
    {
      id: 'hotel_14',
      name: 'Ace Hotel New York',
      location: 'New York, USA',
      price: 350,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotels_hotel_14.jpg'
    },
    {
      id: 'hotel_15',
      name: 'Pod 51 Hotel',
      location: 'New York, USA',
      price: 180,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotels_hotel_15.jpg'
    }
  ])

  // --- Cars Data ---
  const cars = ref([
    {
      id: 'car_1',
      model: 'Ford Fiesta',
      type: 'Economy',
      price: 45,
      seats: 5,
      transmission: 'Manual',
      image: '/images/cars_car_1.jpg'
    },
    {
      id: 'car_2',
      model: 'Volkswagen Golf',
      type: 'Compact',
      price: 55,
      seats: 5,
      transmission: 'Manual',
      image: '/images/cars_car_2.jpg'
    },
    {
      id: 'car_3',
      model: 'Ford Focus',
      type: 'Compact',
      price: 60,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_3.jpg'
    },
    {
      id: 'car_4',
      model: 'BMW 3 Series',
      type: 'Premium',
      price: 120,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_4.jpg'
    },
    {
      id: 'car_5',
      model: 'Mercedes C-Class',
      type: 'Premium',
      price: 130,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_5.jpg'
    },
    {
      id: 'car_6',
      model: 'Toyota Corolla',
      type: 'Intermediate',
      price: 70,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_6.jpg'
    },
    {
      id: 'car_7',
      model: 'Nissan Qashqai',
      type: 'SUV',
      price: 85,
      seats: 5,
      transmission: 'Manual',
      image: '/images/cars_car_7.jpg'
    },
    {
      id: 'car_8',
      model: 'Range Rover Evoque',
      type: 'Luxury SUV',
      price: 180,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_8.jpg'
    },
    {
      id: 'car_9',
      model: 'Fiat 500',
      type: 'Mini',
      price: 40,
      seats: 4,
      transmission: 'Manual',
      image: '/images/cars_car_9.jpg'
    },
    {
      id: 'car_10',
      model: 'Audi A4',
      type: 'Standard',
      price: 110,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_10.jpg'
    },
    {
      id: 'car_11',
      model: 'Tesla Model 3',
      type: 'Electric',
      price: 140,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_11.jpg'
    },
    {
      id: 'car_12',
      model: 'Vauxhall Corsa',
      type: 'Economy',
      price: 48,
      seats: 5,
      transmission: 'Manual',
      image: '/images/cars_car_12.jpg'
    },
    {
      id: 'car_13',
      model: 'Jeep Wrangler',
      type: 'SUV',
      price: 160,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/cars_car_13.jpg'
    },
    {
      id: 'car_14',
      model: 'Porsche 911',
      type: 'Sports',
      price: 350,
      seats: 2,
      transmission: 'Automatic',
      image: '/images/cars_car_14.jpg'
    },
    {
      id: 'car_15',
      model: 'Volkswagen Transporter',
      type: 'Van',
      price: 100,
      seats: 9,
      transmission: 'Manual',
      image: '/images/cars_car_15.jpg'
    }
  ])

  // --- Price Alerts Data ---
  const priceAlerts = ref([
    {
      id: 'alert_1',
      origin: 'LHR',
      destination: 'JFK',
      target_price: 400,
      current_price: 380,
      active: true,
      image: '/images/priceAlerts_alert_1.jpg'
    },
    {
      id: 'alert_2',
      origin: 'LHR',
      destination: 'DXB',
      target_price: 500,
      current_price: 550,
      active: true,
      image: '/images/priceAlerts_alert_2.jpg'
    },
    {
      id: 'alert_3',
      origin: 'LHR',
      destination: 'SYD',
      target_price: 1000,
      current_price: 1100,
      active: false,
      image: '/images/priceAlerts_alert_3.jpg'
    },
    {
      id: 'alert_4',
      origin: 'LHR',
      destination: 'TYO',
      target_price: 850,
      current_price: 900,
      active: true,
      image: '/images/priceAlerts_alert_4.jpg'
    },
    {
      id: 'alert_5',
      origin: 'LGW',
      destination: 'AGP',
      target_price: 150,
      current_price: 120,
      active: false,
      image: '/images/priceAlerts_alert_5.jpg'
    }
  ])

  return {
    flights,
    multiFlights,
    hotels,
    cars,
    priceAlerts
  }
}, {
  persist: {
    storage: sessionStorage,
  },
})