import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    // PROVIDERS
    providers: [
      { id: 'prov_1', name: 'Dr. Sarah Johnson', specialty: 'Primary Care', rating: 4.9, image: '/images/providers_prov_1.jpg', next_slot: 'Today' },
      { id: 'prov_2', name: 'Dr. Michael Chen', specialty: 'Cardiology', rating: 4.8, image: '/images/providers_prov_2.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_3', name: 'Dr. Emily Davis', specialty: 'Dermatology', rating: 4.7, image: '/images/providers_prov_3.jpg', next_slot: 'Today' },
      { id: 'prov_4', name: 'Dr. Robert Wilson', specialty: 'Primary Care', rating: 4.5, image: '/images/providers_prov_4.jpg', next_slot: 'In 2 days' },
      { id: 'prov_5', name: 'Dr. Jessica Taylor', specialty: 'Pediatrics', rating: 4.9, image: '/images/providers_prov_5.jpg', next_slot: 'Today' },
      { id: 'prov_6', name: 'Dr. David Anderson', specialty: 'Primary Care', rating: 4.6, image: '/images/providers_prov_6.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_7', name: 'Dr. Jennifer Martinez', specialty: 'Dermatology', rating: 4.8, image: '/images/providers_prov_7.jpg', next_slot: 'In 3 days' },
      { id: 'prov_8', name: 'Dr. James Thomas', specialty: 'Cardiology', rating: 4.7, image: '/images/providers_prov_8.jpg', next_slot: 'Today' },
      { id: 'prov_9', name: 'Dr. Lisa White', specialty: 'Primary Care', rating: 4.4, image: '/images/providers_prov_9.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_10', name: 'Dr. Daniel Harris', specialty: 'Orthopedics', rating: 4.9, image: '/images/providers_prov_10.jpg', next_slot: 'In 2 days' },
      { id: 'prov_11', name: 'Dr. Mary Martin', specialty: 'Primary Care', rating: 4.8, image: '/images/providers_prov_11.jpg', next_slot: 'Today' },
      { id: 'prov_12', name: 'Dr. Christopher Thompson', specialty: 'Neurology', rating: 4.7, image: '/images/providers_prov_12.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_13', name: 'Dr. Patricia Garcia', specialty: 'Dermatology', rating: 4.6, image: '/images/providers_prov_13.jpg', next_slot: 'In 4 days' },
      { id: 'prov_14', name: 'Dr. Matthew Robinson', specialty: 'Primary Care', rating: 4.9, image: '/images/providers_prov_14.jpg', next_slot: 'Today' },
      { id: 'prov_15', name: 'Dr. Elizabeth Clark', specialty: 'Pediatrics', rating: 4.8, image: '/images/providers_prov_15.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_16', name: 'Dr. Joseph Rodriguez', specialty: 'Cardiology', rating: 4.7, image: '/images/providers_prov_16.jpg', next_slot: 'In 2 days' },
      { id: 'prov_17', name: 'Dr. Linda Lewis', specialty: 'Primary Care', rating: 4.5, image: '/images/providers_prov_17.jpg', next_slot: 'Today' },
      { id: 'prov_18', name: 'Dr. Thomas Lee', specialty: 'Orthopedics', rating: 4.8, image: '/images/providers_prov_18.jpg', next_slot: 'Tomorrow' },
      { id: 'prov_19', name: 'Dr. Barbara Walker', specialty: 'Dermatology', rating: 4.6, image: '/images/providers_prov_19.jpg', next_slot: 'In 3 days' },
      { id: 'prov_20', name: 'Dr. Charles Hall', specialty: 'Primary Care', rating: 4.9, image: '/images/providers_prov_20.jpg', next_slot: 'Today' }
    ],

    // MENTAL HEALTH THERAPISTS
    therapists: [
      { id: 'th_1', name: 'Amanda Wilson, LMFT', specialty: 'Anxiety & Depression', experience: 10, image: '/images/therapists_th_1.jpg' },
      { id: 'th_2', name: 'Dr. Brian Miller, PsyD', specialty: 'Trauma', experience: 15, image: '/images/therapists_th_2.jpg' },
      { id: 'th_3', name: 'Catherine Moore, LCSW', specialty: 'Family Therapy', experience: 8, image: '/images/therapists_th_3.jpg' },
      { id: 'th_4', name: 'David Brown, LPC', specialty: 'Addiction', experience: 12, image: '/images/therapists_th_4.jpg' },
      { id: 'th_5', name: 'Dr. Eleanor Davis, PhD', specialty: 'Child Psychology', experience: 20, image: '/images/therapists_th_5.jpg' },
      { id: 'th_6', name: 'Frank Wright, LMFT', specialty: 'Couples Counseling', experience: 14, image: '/images/therapists_th_6.jpg' },
      { id: 'th_7', name: 'Grace Green, LCSW', specialty: 'Anxiety', experience: 6, image: '/images/therapists_th_7.jpg' },
      { id: 'th_8', name: 'Dr. Henry Baker, PsyD', specialty: 'Depression', experience: 18, image: '/images/therapists_th_8.jpg' },
      { id: 'th_9', name: 'Isabella King, LPC', specialty: 'Stress Management', experience: 9, image: '/images/therapists_th_9.jpg' },
      { id: 'th_10', name: 'Jack Scott, LMFT', specialty: 'Grief Counseling', experience: 11, image: '/images/therapists_th_10.jpg' },
      { id: 'th_11', name: 'Kelly Adams, LCSW', specialty: 'Anxiety & Depression', experience: 7, image: '/images/therapists_th_11.jpg' },
      { id: 'th_12', name: 'Dr. Larry Nelson, PhD', specialty: 'Trauma', experience: 22, image: '/images/therapists_th_12.jpg' },
      { id: 'th_13', name: 'Megan Carter, LPC', specialty: 'Family Therapy', experience: 5, image: '/images/therapists_th_13.jpg' },
      { id: 'th_14', name: 'Nathan Mitchell, LMFT', specialty: 'Addiction', experience: 13, image: '/images/therapists_th_14.jpg' },
      { id: 'th_15', name: 'Olivia Perez, LCSW', specialty: 'Child Psychology', experience: 16, image: '/images/therapists_th_15.jpg' }
    ],

    // PRESCRIPTIONS
    prescriptions: [
      { id: 'rx_1', name: 'Lisinopril', dosage: '10mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_1.jpg' },
      { id: 'rx_2', name: 'Metformin', dosage: '500mg', status: 'Active', supply: '90 days', image: '/images/prescriptions_rx_2.jpg' },
      { id: 'rx_3', name: 'Atorvastatin', dosage: '20mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_3.jpg' },
      { id: 'rx_4', name: 'Levothyroxine', dosage: '50mcg', status: 'Active', supply: '90 days', image: '/images/prescriptions_rx_4.jpg' },
      { id: 'rx_5', name: 'Amlodipine', dosage: '5mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_5.jpg' },
      { id: 'rx_6', name: 'Metoprolol', dosage: '25mg', status: 'Inactive', supply: '30 days', image: '/images/prescriptions_rx_6.jpg' },
      { id: 'rx_7', name: 'Omeprazole', dosage: '20mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_7.jpg' },
      { id: 'rx_8', name: 'Losartan', dosage: '50mg', status: 'Active', supply: '90 days', image: '/images/prescriptions_rx_8.jpg' },
      { id: 'rx_9', name: 'Gabapentin', dosage: '300mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_9.jpg' },
      { id: 'rx_10', name: 'Hydrochlorothiazide', dosage: '12.5mg', status: 'Inactive', supply: '30 days', image: '/images/prescriptions_rx_10.jpg' },
      { id: 'rx_11', name: 'Sertraline', dosage: '50mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_11.jpg' },
      { id: 'rx_12', name: 'Simvastatin', dosage: '20mg', status: 'Active', supply: '90 days', image: '/images/prescriptions_rx_12.jpg' },
      { id: 'rx_13', name: 'Montelukast', dosage: '10mg', status: 'Active', supply: '30 days', image: '/images/prescriptions_rx_13.jpg' },
      { id: 'rx_14', name: 'Escitalopram', dosage: '10mg', status: 'Inactive', supply: '30 days', image: '/images/prescriptions_rx_14.jpg' },
      { id: 'rx_15', name: 'Albuterol', dosage: '90mcg', status: 'Active', supply: 'Inhaler', image: '/images/prescriptions_rx_15.jpg' }
    ],

    // APPOINTMENTS
    appointments: [
      { id: 'apt_1', provider: 'Dr. Sarah Johnson', date: '2025-10-25', time: '10:00 AM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_1.jpg' },
      { id: 'apt_2', provider: 'Dr. Michael Chen', date: '2025-10-28', time: '02:30 PM', type: 'In-Person', status: 'Upcoming', image: '/images/appointments_apt_2.jpg' },
      { id: 'apt_3', provider: 'Amanda Wilson, LMFT', date: '2025-11-01', time: '11:00 AM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_3.jpg' },
      { id: 'apt_4', provider: 'Dr. Emily Davis', date: '2025-11-05', time: '09:15 AM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_4.jpg' },
      { id: 'apt_5', provider: 'Dr. Robert Wilson', date: '2025-11-10', time: '03:45 PM', type: 'In-Person', status: 'Upcoming', image: '/images/appointments_apt_5.jpg' },
      { id: 'apt_6', provider: 'Dr. Sarah Johnson', date: '2025-09-15', time: '10:00 AM', type: 'Virtual', status: 'Past', image: '/images/appointments_apt_6.jpg' },
      { id: 'apt_7', provider: 'Dr. Jessica Taylor', date: '2025-08-20', time: '01:00 PM', type: 'Virtual', status: 'Past', image: '/images/appointments_apt_7.jpg' },
      { id: 'apt_8', provider: 'Dr. David Anderson', date: '2025-07-10', time: '11:30 AM', type: 'In-Person', status: 'Past', image: '/images/appointments_apt_8.jpg' },
      { id: 'apt_9', provider: 'Catherine Moore, LCSW', date: '2025-06-05', time: '04:00 PM', type: 'Virtual', status: 'Past', image: '/images/appointments_apt_9.jpg' },
      { id: 'apt_10', provider: 'Dr. Jennifer Martinez', date: '2025-05-12', time: '09:00 AM', type: 'Virtual', status: 'Past', image: '/images/appointments_apt_10.jpg' },
      { id: 'apt_11', provider: 'Dr. James Thomas', date: '2025-11-15', time: '02:00 PM', type: 'In-Person', status: 'Upcoming', image: '/images/appointments_apt_11.jpg' },
      { id: 'apt_12', provider: 'Dr. Lisa White', date: '2025-11-20', time: '10:30 AM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_12.jpg' },
      { id: 'apt_13', provider: 'Dr. Daniel Harris', date: '2025-11-25', time: '01:45 PM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_13.jpg' },
      { id: 'apt_14', provider: 'Dr. Mary Martin', date: '2025-12-01', time: '11:15 AM', type: 'In-Person', status: 'Upcoming', image: '/images/appointments_apt_14.jpg' },
      { id: 'apt_15', provider: 'Dr. Christopher Thompson', date: '2025-12-05', time: '03:30 PM', type: 'Virtual', status: 'Upcoming', image: '/images/appointments_apt_15.jpg' }
    ],

    // BILLS
    bills: [
      { id: 'bill_1', description: 'Office Visit - Dr. Johnson', date: '2025-09-15', amount: 50.00, status: 'Due', image: '/images/bills_bill_1.jpg' },
      { id: 'bill_2', description: 'Lab Work', date: '2025-09-15', amount: 25.00, status: 'Due', image: '/images/bills_bill_2.jpg' },
      { id: 'bill_3', description: 'Therapy Session', date: '2025-08-20', amount: 80.00, status: 'Paid', image: '/images/bills_bill_3.jpg' },
      { id: 'bill_4', description: 'Specialist Consultation', date: '2025-07-10', amount: 120.00, status: 'Paid', image: '/images/bills_bill_4.jpg' },
      { id: 'bill_5', description: 'X-Ray Services', date: '2025-06-05', amount: 150.00, status: 'Paid', image: '/images/bills_bill_5.jpg' },
      { id: 'bill_6', description: 'Annual Checkup', date: '2025-05-12', amount: 0.00, status: 'Paid', image: '/images/bills_bill_6.jpg' },
      { id: 'bill_7', description: 'Prescription Copay', date: '2025-09-10', amount: 15.00, status: 'Due', image: '/images/bills_bill_7.jpg' },
      { id: 'bill_8', description: 'Telehealth Visit', date: '2025-08-15', amount: 40.00, status: 'Paid', image: '/images/bills_bill_8.jpg' },
      { id: 'bill_9', description: 'Blood Test', date: '2025-07-20', amount: 30.00, status: 'Paid', image: '/images/bills_bill_9.jpg' },
      { id: 'bill_10', description: 'MRI Scan', date: '2025-04-10', amount: 250.00, status: 'Paid', image: '/images/bills_bill_10.jpg' }
    ],

    // PLANS
    plans: [
      { id: 'plan_1', name: 'Gold Premium Plan', type: 'PPO', eligible: true, image: '/images/plans_plan_1.jpg' },
      { id: 'plan_2', name: 'Silver Saver Plan', type: 'HMO', eligible: true, image: '/images/plans_plan_2.jpg' },
      { id: 'plan_3', name: 'Bronze Basic Plan', type: 'EPO', eligible: true, image: '/images/plans_plan_3.jpg' },
      { id: 'plan_4', name: 'Dental Plus', type: 'Dental', eligible: false, image: '/images/plans_plan_4.jpg' },
      { id: 'plan_5', name: 'Vision Basic', type: 'Vision', eligible: true, image: '/images/plans_plan_5.jpg' }
    ]
  }),
  persist: {
    storage: sessionStorage
  }
})