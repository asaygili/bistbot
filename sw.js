/* ═══════════════════════════════════════════════
   BIST BOT — Service Worker v2.0
   Offline cache + arka plan güncelleme
═══════════════════════════════════════════════ */

const CACHE_NAME = 'bistbot-v2';
const STATIC = [
  './index.html',
  './manifest.json',
  'https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap'
];

// Kurulum: statik dosyaları önbelleğe al
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

// Aktivasyon: eski önbellekleri temizle
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: Network-first (Yahoo Finance için), Cache-fallback (statik için)
self.addEventListener('fetch', e => {
  const url = e.request.url;

  // Yahoo Finance / News API — her zaman ağdan çek
  if (url.includes('finance.yahoo.com') || url.includes('news.google.com') ||
      url.includes('allorigins.win')) {
    e.respondWith(
      fetch(e.request).catch(() => new Response(
        JSON.stringify({error: 'Çevrimdışı — veri alınamadı'}),
        {headers: {'Content-Type': 'application/json'}}
      ))
    );
    return;
  }

  // Statik dosyalar — önce önbellek, sonra ağ
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(resp => {
        if (resp && resp.status === 200 && resp.type !== 'opaque') {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
        }
        return resp;
      }).catch(() => caches.match('./index.html'));
    })
  );
});

// Arka plan senkronizasyon mesajı al
self.addEventListener('message', e => {
  if (e.data === 'skipWaiting') self.skipWaiting();
});
