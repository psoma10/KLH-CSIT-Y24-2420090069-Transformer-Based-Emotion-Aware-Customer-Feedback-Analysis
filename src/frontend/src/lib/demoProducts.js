// Hardcoded demo catalogue for the Analyze page. These stand in for a real
// product service — the page is a realistic review-composer surface whose only
// live backend dependency is the emotion model, so the product data is static
// on purpose.

export const DEMO_PRODUCTS = [
  {
    id: "b09-anc-headphones",
    title:
      "SonicWave Pro ANC Wireless Over-Ear Headphones, 40H Battery, Hi-Res Audio, Multipoint Bluetooth 5.3",
    brand: "SonicWave",
    price: 249.99,
    listPrice: 329.99,
    rating: 4.3,
    ratingCount: 18427,
    inStock: true,
    accent: "#4f46e5",
    emoji: "🎧",
    bullets: [
      "Adaptive hybrid noise cancellation with 6-mic array",
      "40-hour playtime, 5-minute quick charge for 4 hours",
      "Memory-foam earcups with breathable protein leather",
    ],
    histogram: { 5: 62, 4: 19, 3: 8, 2: 4, 1: 7 },
  },
  {
    id: "b07-espresso",
    title:
      "Brevari BaristaOne Semi-Automatic Espresso Machine with Conical Burr Grinder and Steam Wand",
    brand: "Brevari",
    price: 699.0,
    listPrice: 799.0,
    rating: 4.6,
    ratingCount: 9034,
    inStock: true,
    accent: "#b45309",
    emoji: "☕️",
    bullets: [
      "Integrated 30-setting conical burr grinder",
      "3-second PID heat-up with digital temperature control",
      "Microfoam steam wand for latte art at home",
    ],
    histogram: { 5: 71, 4: 16, 3: 6, 2: 3, 1: 4 },
  },
  {
    id: "b0c-standing-desk",
    title:
      "Northloop Ascend Electric Standing Desk, 60\" x 30\" Bamboo Top, Dual Motor, 4 Memory Presets",
    brand: "Northloop",
    price: 429.5,
    listPrice: 549.0,
    rating: 3.9,
    ratingCount: 4211,
    inStock: false,
    accent: "#0f766e",
    emoji: "🪑",
    bullets: [
      "Dual-motor lift, 27.5\"–47\" range, 265 lb capacity",
      "Solid bamboo surface with anti-collision sensors",
      "Cable tray and grommets included",
    ],
    histogram: { 5: 44, 4: 21, 3: 14, 2: 9, 1: 12 },
  },
];

export function formatPrice(value) {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD" });
}

export function formatCount(value) {
  return value.toLocaleString("en-US");
}
