# TR–RU Forecast: Mevcut Durum ve Tamamlama Planı

Tarih: 7 Ekim 2026; ilerleme son güncelleme: 16 Ekim 2026. Bu belge mevcut checkout, kaynak kod, yerel geliştirme çıktıları ve specification v1.1 üzerinden hazırlanmıştır. Her milestone ayrı görev ve kabul kapısıyla ilerler. Yeni bir kapsam/specification freeze yerine mevcut M0–M15 yol haritasını somutlaştırır.

## İlerleme checklisti

Kural: Bir madde ancak dosya, komut sonucu, test veya raporla doğrulanınca `[x]` yapılır. Kısmen çalışılan/eksik kalan iş `[ ]` kalır. Her görev sonunda bu liste, doğrulama kaydı ve aşağıdaki durum tablosu aynı oturumda güncellenir.

### Başlangıç incelemesi

- [x] Repository, Git durumu, son commitler ve specification v1.1 okundu.
- [x] M0–M15 yol haritası ile mevcut dokümantasyon/kod durumu eşleştirildi.
- [x] Dört model için 3.934 tahmin, 42 metrik grubu ve validation karşılaştırması yerel artifactlerden doğrulandı.
- [x] M9 kayıtlı input checksum'ları manifestler, development features ve frozen validation folds ile doğrulandı.
- [x] `ruff check .` ve frontend `npm run build` geçti.
- [x] M9 hedefli testleri: 40 geçti; bir dependency deprecation uyarısı kayda geçti.
- [x] Tam Python suite, testlerin geçici dizinlere yazdığı izole koşumda GREEN: M9 kabulünde 364, M10 sonrası kilitli temiz ortamda 415 test geçti (1 mevcut dependency uyarısı).
- [x] Proje durum ve tamamlama planı bu dosyada oluşturuldu.

### A — M9 kabulü ve yeniden üretilebilirlik

- [x] A1. Mevcut M9 kaynak/config/test diff'i dosya dosya incelendi; bulgular `M9_ACCEPTANCE_REPORT.md` içinde kaydedildi.
- [x] A2. M9 ML fonksiyonları <=30 satır; imzalar tipli, `Any`/type-ignore yok. 40 M9 testi ve frozen CV replay (3.934 tahmin / 82.614 SHAP satırı) sayısal eşdeğerliği doğruladı; `M11_ACCEPTANCE_REPORT.md`.
- [x] A3. Testler pytest geçici processed-data kopyasına yönlendirildi; tam suite bu izolasyonla geçti (364 test).
- [x] A4. Run metadata gerçek commit/dirty state, M9 source fingerprint'i ve input/output checksum'larını kaydediyor; ikinci temiz run ile doğrulandı.
- [x] A5. `uv.lock` ile runtime/dev extras temiz Python 3.13 ortamına kuruldu; production `ComtradeClient` ve runtime `httpx` import edildi.
- [x] A6. M4, M8 ve M9 rapor statüleri ile README mevcut milestone ve kanıtlara göre hizalandı.
- [x] A7. M4/M5 veri vintage, yaklaşık macro release date ve forecast decision-time kayıtları denetlendi; point-in-time/gerçek zaman iddiası kanıtlanmadığı için kapsam `M9_ACCEPTANCE_REPORT.md` içinde daraltıldı.
- [x] A8. M9 ayrı run klasöründe yeniden üretildi; manifest, prediction, metric, interval, SHAP ve metadata tutarlılığı doğrulandı.
- [x] A9. M9 kabul kaydı ve Stage A takip kaydı tamamlandı; A2 kapandı, uyarı kaynağında giderildi ve M11 gate açıldı.

### B — M10 champion seçimi

- [x] B1. 14 seri × 3 horizon için dört modelin aynı validation fold anahtarlarını birleştir: 281 ortak aylık origin; 15.736 audit satırı.
- [x] B2. Tie, sıfır denominator, failed/missing candidate ve yüzde 2 göreli kazanç kuralları seçici ve 51 M10 testiyle doğrulandı.
- [x] B3. Seasonal Naive interval politikası registry referansında açıkça unavailable olarak belirlendi; yöntem uydurulmadı.
- [x] B4. Seçici ve registry uygulandı; 42 champion/fallback kararı (SARIMA 33, XGBoost 9) üretildi.
- [x] B5. `M10_CHAMPION_SELECTION_REPORT.md` ve `M10_ACCEPTANCE_REPORT.md` ile tekrar üretim/immutability/checksum kanıtı kaydedildi; holdout kapalı. Stage A takip kaydı A2 ve M8 eşdeğerlik gate'ini kapattı; M11 sonuçları aşağıda kaydedildi.

### C — M11 öncesi deney freeze

- [x] C1. 504 task-origin hücresi için origin_date, forecast_period, train_end ve deterministik erişim kuralları kaydedildi; gerçek tarihsel yayın zamanı kanıtı iddia edilmedi.
- [x] C2. Kullanıcının onayıyla aylık rolling origin/refit ve retrospective latest-vintage kapsamı, scoring öncesi Protocol §19 ve holdout_spec_v1_1.yaml ile kilitlendi.
- [x] C3. Origin sonrası target ve deterministik availability tarihi sonrası macro mutasyonlarının origin özelliklerini değiştirmediği test edildi. Bu, gerçek published_at/vintage kanıtı değildir.
- [x] C4. CV-seed ve sonraki gerçekleşmiş residual havuzlarında target <= origin şartı test/audit ile doğrulandı; minimum XGBoost kalibrasyon sayısı 86.
- [x] C5. Champion karar hash'i, orijinal registry checksum arşivi, gate input hashleri ve M11 source fingerprint'i kaydedildi; ilk değerlendirme ve izinli tekrar üretimler ayrı immutable paketlerde korundu.

### D — M11 final holdout

- [x] D1. 42 görev × 12 hedef × 3 rol = 1.512 kayıt üretildi; model, özellik, order ve hiperparametreler yeniden seçilmedi.
- [x] D2. 42 satırlık leaderboard ile sMAPE/MAE/RMSE, paired-origin skill, raw/operational metrikler ve CV drift hesaplandı.
- [x] D3. Coverage/width ve paydalar raporlandı; bir negatif champion ve bir negatif fallback ham olarak korundu. XGBoost 80% undercoverage gizlenmedi.
- [x] D4. Tüm input/output hashleri ve kaynak fingerprint'i doğrulandı; bağımsız make koşumu altı sayısal veri artifactinin hash'ini eşledi.
- [x] D5. `M11_FINAL_HOLDOUT_REPORT.md` ve `M11_ACCEPTANCE_REPORT.md` tamamlandı. 441 test/0 uyarı; teslimde 31 hedefli test/0 uyarı ve tekrar üretim doğrulandı.

### E–F — Serving ve M12 API

- [ ] E1. Backtest artefactleriyle live serving modellerinin yaşam döngüsünü ayır; yüklenebilir inference modellerini üret.
- [ ] E2. PostgreSQL/Supabase schema, migration, unique constraints, idempotent upsert ve read-only demo erişimini kur.
- [x] E3. Artifact API; source/reporter/flow, manifest/run digests, origin/target, horizon, unit ve interval availability metadata'sını taşıyor. İleriye dönük serving-model lifecycle E1 kapsamında açık.
- [x] F1. Overview, trade, forecasts, metrics/leaderboard, sectors, mirror, indicators ve methodology GET endpointleri typed sözleşmelerle uygulandı.
- [x] F2. Immutable artifact repository → service → router; typed response/error, pagination, tarih/sector/horizon filtreleri ve unavailable yanıtları uygulandı. Frozen observation cutoff ile retrieval tarihi ayrı; güncel veri freshness iddiası yok.
- [x] F3. Kullanıcının disk-artifact M12 kapsamı için exact-origin CORS, checksum/readiness, missing/corrupt paket 503 ve API contract testleri doğrulandı (70 API testi/0 uyarı; tam suite 509 test/0 uyarı, 806.75 s). DB bağlantısı/migration E2 kapsamında açık; DB unavailable kanıtı iddia edilmiyor.

### G — M13 dashboard

- [ ] G1. API contract'a bağlı Overview → Trade Explorer → Forecasts ilk uçtan uca dilimini tamamla.
- [ ] G2. Gerçek TR/EN/RU locale switching, URL'de filtreler, sayı/tarih yerelleştirmesi ve erişilebilir kontroller ekle.
- [ ] G3. Sector Analysis, Model Lab, Mirror Statistics, Economic Indicators ve Methodology modüllerini tamamla.
- [ ] G4. Empty/loading/error/stale, eksik mirror, uncertainty, mobile ve desktop durumlarını test et.
- [ ] G5. Tahmin/actual/backtest ayrımını ve nominal–gerçekleşen interval coverage farkını anlaşılır göster.

### H–I — Araştırma, portfolyo ve release

- [ ] H1. README/architecture/data dictionary/methodology/limitations/reproduction rehberini gerçek implementasyona göre güncelle.
- [ ] H2. İngilizce araştırma raporu, Türkçe proje özeti ve Rusça abstract'ı gerçekleşmiş sonuçlarla yaz.
- [ ] H3. Artifactlerden otomatik sonuç tabloları, demo ekranları/video ve portfolio card hazırla; lisansları doğrula.
- [ ] I1. Temiz clone kurulumunu; Python test/lint, frontend build/E2E, migration ve secrets kontrollerini CI'da çalıştır.
- [ ] I2. Docker/local setup, staging/prod config, logs, health, rollback ve UAT runbook hazırla.
- [ ] I3. MUST maddelerini kanıtlarla kapat; API/UI/data/report uyumunu son kez kontrol et.
- [ ] I4. İstenmiş ayrı yayın yetkisiyle semver tag, changelog, artifact checksum ve kalıcı demo release'ini tamamla.

**M12 kapsam kararı:** Kullanıcının 16 Ekim promptu mevcut M10/M11 artifactlerini diskten sunan API'yi istedi; model eğitimi/serialization ve PostgreSQL E1–E2 açık kalır. M13 ilk UI dilimi bu artifact API sözleşmesine bağlanabilir; ayrı kullanıcı görevi olmadan başlanmaz.

## 1. Projenin amacı ve portfolyo konumu

Türkiye–Rusya aylık ikili ticaretini iki yönde, TOTAL ve seçilmiş HS2 sektörlerinde 1/3/6 ay ilerisi için tahmin eden; Seasonal Naive, SARIMA, SARIMAX ve XGBoost'u ortak zaman serisi doğrulamasında karşılaştıran, belirsizlik ve veri kalitesini gösteren çok dilli araştırma platformu.

Portfolyodaki esas kanıt zinciri: **kaynak ve veri kalitesi → sızıntı denetimi → adil deney → bağımsız holdout → çalışan ürün → yeniden üretilebilir araştırma**. Daha karmaşık modelin kazanması başarı şartı değildir. Modelin neden seçildiğini ve sonucun nerede zayıfladığını açıklayabilmek başarı ölçütüdür.

Authoritative belgeler `specification/` içindeki v1.1 dosyalarıdır. `TR_RU_Forecasting_Specification_Pack_v1.0/` tarihsel pakettir; uygulama görevlerinde aktif kaynak olarak kullanılmamalıdır.

## 2. Nerede kalındı?

| Katman / milestone | Mevcut kanıt | Durum |
| --- | --- | --- |
| M0–M2: temel, ingestion, mirror | Config/logging, Comtrade client/pipeline/storage, mirror pairing/quality modülleri ve testleri | Uygulanmış; bütün zincirin temiz ortam kabulü tekrar yapılmalı |
| Pre-M3, M3: veri denetimi ve sektörler | Full-history raporu, target materializer, eligibility/ranking, frozen sektör YAML | Uygulanmış; v1.1 target politikası kullanılmalı |
| M4: makro veriler | Beş gösterge, kaynak registry, yaklaşık yayın gecikmeleri, RU IP linkage | Uygulanmış; raporun başlığı hâlâ BLOCKED, son verdict READY; kabul kaydı tutarsız |
| M5: özellikler | Development feature panel, lag/rolling/calendar/macro alignment ve audit | Uygulanmış; tarihsel vintage ve karar zamanı iddiaları ayrıca denetlenmeli |
| M6: Seasonal Naive | 3.934 tahmin, 42 metrik grubu | Uygulanmış; mevcut çıktı 80/95 aralık içermiyor |
| M7: SARIMA | Model/order selection/backtest/diagnostics/failure log | Git geçmişinde dondurulmuş; 3.934 geçerli tahmin mevcut |
| M8: SARIMAX | Son commit `9d539b8 feat: complete M8 SARIMAX`; 1.344 başarılı fit, 3.934 tahmin | Git'e alınmış; rapor FINAL CANDIDATE / AWAITING EXTERNAL REVIEW diyor |
| M9: XGBoost | Direct modeller, tuning, conformal, SHAP, provenance ve ayrı acceptance run | A2 refactor/uyarı düzeltmesi ve tam sayısal CV eşdeğerliği doğrulandı |
| M10 | CV-only selector, 168 aday, 42 champion/fallback, registry, checksum ve rapor | Uygulandı; 42 karar korundu, pre-M11 gate koşulları Stage A ile kapandı |
| M11 | 1.512 rol kaydı, 42 görevlik matris, native/conformal aralıklar, 9.072 SHAP satırı, hash/provenance | Retrospective latest-vintage deterministik lag kapsamıyla tamamlandı; 441 test / 0 uyarı |
| M12: API | Typed, checksum-pinned frozen artifact GET API; 14 sectors/42 tasks/168 CV candidates, 1,512 forecasts and 9,072 SHAP rows | Verified: 70 API tests + 509 full-suite tests, zero warnings; no deployment claim |
| M13: UI | Tek ana sayfa ve dört metinlik TR/EN/RU sözlüğü; varsayılan TR | İskelet; grafikler, veri bağlantısı ve dil değiştirme akışı yok |
| M14–M15 | Araştırma klasörü yalnız `.gitkeep`; README hâlâ M1/v1.0 | Paketleme ve release kabulü eksik |

Kesin devam noktası: **M13 artifact API'ye bağlı UI → M14 → M15**; ileriye dönük inference ve PostgreSQL (E1–E2) ürünün tam V1 kabulü için ayrıca açık. M1'den yeniden başlanmamalı; M9 sadece rapor checkbox'larına bakılarak tamamlanmış sayılmamalı.

Git çalışma alanında M9'a ait yeni dosyalar ile `pyproject.toml` ve mevcut testlerde değişiklikler var. Bu değişiklikler korunmalı; reset/clean yapılmamalı. Kod değişikliği gerektiren sonraki işte bu çalışma durumunu içeren ayrı checkout/worktree kullanılmalı. HEAD'den oluşturulan sıradan worktree commit edilmemiş M9'u kendiliğinden içermez.

## 3. Mevcut deneyin sınırları ve gözlenen sonuçlar

Bu bölüm ilk M9 incelemesinin tarihsel bulgularını korur. Güncel M11 sonuçları üstteki ilerleme kaydında ve `M11_ACCEPTANCE_REPORT.md` içindedir.

- Forecast target iki yönde de Türkiye-reported: TR→RU ihracat, RU→TR ithalat. Russia-reported veriler tarihsel mirror/reference; hedefi tamamlamak veya ortalamasını almak için kullanılamaz.
- Frozen veri evreni: 2012-01–2025-12. Development: 2012-01–2024-12. Final holdout: 2025-01–2025-12.
- 14 seri, 21 XGBoost predictor, 1/3/6 ay horizons, toplam 281 ortak validation fold ve model başına 3.934 tahmin; 42 seri/horizon metrik grubu.
- M9 tuning: 2016 hedefleri üzerinde 16 aday/horizon; seçilmiş üç parametre seti sektörler arasında paylaşılır. Tuning 2017–2024 resmi validation hedeflerine dayanmamalı.
- M9 input manifest ve development features için kayıtlı beş SHA-256 değeri mevcut dosyalarla eşleşti.
- Bu incelemede 2025 gerçek değerleri üzerinden sonuç hesaplanmadı; yeni holdout deneyleri çalıştırılmadı. Önceki bütün oturumların holdout'a erişmediği yalnız mevcut dosyalardan kesin olarak ispatlanamaz.
- Frozen snapshot'ın 2025-12'de bitmesi bugün kaynakta daha yeni veri bulunmadığı anlamına gelmez. Güncel kaynak erişimi bu incelemede yenilenmedi.

Yerel metrik dosyalarından hesaplanan, 14 seriye eşit ağırlık veren validation sMAPE ortalamaları:

| Model | h1 | h3 | h6 |
| --- | ---: | ---: | ---: |
| Seasonal Naive | %38,95 | %39,03 | %39,12 |
| SARIMA | %22,98 | %27,51 | %32,66 |
| SARIMAX | %26,34 | %32,42 | %38,99 |
| XGBoost | %23,73 | %30,26 | %33,01 |

SARIMA bu özet karşılaştırmada en düşük hataya sahip. Bunlar final holdout sonucu veya 42 grubun champion seçimi değildir. SARIMAX'ın ek makro değişkenlerle bu deneyde ortalama hatayı düşürmediği açıkça raporlanmalı. XGBoost SHAP sıralaması, makronun tahmin faydasını tek başına kanıtlamaz; nedensellik de göstermez.

XGBoost aralıkları 3.038 tahminde mevcut; 896 erken validation tahmininde yeterli geçmiş kalibrasyon artığı yok. Gerçekleşen kapsamalar:

| Horizon | Nominal %80 aralık kapsaması | Nominal %95 aralık kapsaması |
| --- | ---: | ---: |
| h1 | %65,5 | %91,0 |
| h3 | %62,0 | %89,3 |
| h6 | %62,9 | %87,5 |

Nominal aralıkların altında kapsama önemli bir araştırma bulgusudur. Mevcut yöntem ve sonuç korunmalı; UI bunları garanti gibi sunmamalı. Başka kalibrasyon yöntemi araştırılacaksa yalnız development üzerinde, sürümlenmiş ek deney olarak ve M11'den önce tasarlanmalı.

## 4. Önce kapatılması gereken açıklar

1. **Kabul ve belge tutarlılığı — M9 için düzeltildi:** README M9/v1.1 konumuna çekildi; M4/M8 statüleri kanıta göre düzenlendi; M9 raporu development validation olarak üretiliyor. UI/dashboard hâlâ iskelet.
2. **Kod provenance — M9 için düzeltildi:** Gerçek SHA, dirty paths, kaynak fingerprint'i, input/output hashleri ve benzersiz run ID metadata'ya yazılıyor. Önceki run kayıtları geriye dönük değiştirilmedi.
3. **Artifact yaşam döngüsü:** Veriler/modeller gitignore altında; temiz clone yerel araştırma sonuçlarını içermez. Çıktı manifesti, checksum, taşınabilir relative paths ve offline örnek paket gerekli. `models/` şu anda yalnız `.gitkeep` içeriyor; üretim inference için eğitilmiş modeller ve yükleme sözleşmesi eksik.
4. **Test izolasyonu — M9 doğrulandı:** Pytest processed-data klasörünü geçici kopyaya alıyor; 364 test ana research artifact'lerine yazmadan geçti.
5. **Bağımlılıklar — M9 doğrulandı:** `uv.lock` üretildi, `httpx` runtime dependency oldu ve kilitli production/dev kurulumu temiz Python 3.13 ortamında doğrulandı. Frontend lock/build kontrolleri release aşamasında tekrarlanmalı.
6. **Tarihsel knowability — sınırlama belgelendi:** Makro as-of filtreleri yaklaşık yayın tarihleri kullanıyor; latest-vintage verinin geçmişte bilinen değerlerle aynı olduğu kanıtlanmıyor. M9 raporu point-in-time/gerçek zaman iddiası taşımıyor.
7. **Trade karar zamanı — kapsam sınırı belgelendi:** Gerçek yayın zamanı/vintage kanıtı yok. Kullanıcının açık talebiyle M11 retrospective latest-vintage ve deterministik lag sözleşmesiyle tamamlandı; gerçek zamanlı knowability iddiası yok.
8. **Holdout geometrisi — çözüldü:** Protokol §19 ve holdout spec, her 2025 hedefi için target-minus-h origin ve expanding refit tanımlar; 42 görev × 12 hedef tamamlandı. Erken originlerin retrospective reconstruction sınırı raporlandı.
9. **Baseline belirsizliği:** Seasonal Naive champion olabilir ama mevcut çıktısında aralık yok. M10 öncesinde development-only, seri/horizon bazlı residual interval yöntemi ve kalibrasyon erişilebilirliği dondurulmalı veya ürünün interval unavailable durumuna ilişkin spec değişikliği açıkça kabul edilmeli.
10. **Servis sınırı:** API'de wildcard CORS + credentials, typed domain response yok; PostgreSQL/Supabase yalnız config alanı. Public demo read-only, sunucu yetkileri ve resource limits somutlaştırılmalı.
11. **Bakım sözleşmesi — A2 kapandı:** Typed kayıtlar ve küçük fonksiyonlara ayırma tamamlandı. AST uzunluk/tip denetimi, 40 M9 testi ve tam sayısal CV replay eşdeğerliği doğrulandı; Stage A kabul kaydı mevcut.

Tarihsel vintage ayrımı için resmi kaynak: [ALFRED download help](https://alfred.stlouisfed.org/help/downloaddata). Multi-step rolling-origin değerlendirme için: [Forecasting: Principles and Practice — time series cross-validation](https://otexts.com/fpp3/tscv.html). Bunlar proje kaynaklarının tümü için vintage erişiminin bulunduğu anlamına gelmez.

## 5. Uçtan uca uygulama sırası

### A — M9 acceptance ve yeniden üretilebilirlik kapısı

**İş:** Mevcut M9 diff'ini incele; önceki manifest/çıktı eşleşmelerini kontrol et; test izolasyonu, run provenance, dependency lock, artifact manifest ve durum belgelerini düzelt. Kritik kodda mevcut style guide'a göre typed domain objects, anlamlı hata türleri, küçük fonksiyonlar ve testler uygula; tüm ingestion/model kodunu yeniden yazma.

**Çıktı:** `M9_ACCEPTANCE_REPORT.md`, güncel durum tablosu, bağımlılık kilidi, ayrı test komutları, development artifact doğrulayıcı ve kaynak/vintage/karar zamanı denetim notu. Sayısal yöntem değişmiyorsa mevcut metriklerle eşleşme; değişiyorsa yeni run ve yazılı spec etkisi.

**Kabul:** Temiz kurulum; kritik unit/leakage/mutation testleri; araştırma klasörüne test yazımı yok; 14 seri/42 grup/281 fold sözleşmesi; checksum eşleşmesi; M9 bağımsız review bulguları kapalı; holdout kapalı. Eksik vintage varsa iddia kapsamı buna göre daraltılır.

**Dur:** M10'a ayrı görev verilmeden geçme.

### B — M10 champion selection ve registry

**İş:** Dört modelin 42 grup sonuçlarını aynı fold anahtarlarıyla birleştir. Her grup için frozen kuralı uygula: düşük sMAPE → MAE tie-break → karmaşık modelin daha basite göre göreli kazancı <%2 ise basit model. Eşitlik, sıfır denominator, missing/failed fit, kapsamı eksik aday ve çoklu basit adayın nasıl ele alınacağı deterministik testlerle açıklansın.

**Çıktı:** `evaluation/champion_selector.py`, registry şeması/serializer, `m10_champion_selection` artifact, karar gerekçeleri, model/config/snapshot hashleri, `M10_CHAMPION_REPORT.md`. Repo Python filename düzenini koru; toplu rename yapma.

**Kabul:** 42 anahtarın tamamı bir champion veya açık unavailable reason üretir; global ortalama tek model seçtirmez; sonuç deterministik; %2 relative gain ile yüzde puan farkı ayrılır; aday/fold coverage doğrulanır. Seasonal Naive için ürünün interval politikası kapatılır. Champion snapshot immutable olur; holdout kapalı kalır.

### C — M11 öncesi final deney freeze kapısı

**İş:** Sektörler, özellikler, model aileleri, tuning, metrikler, champion, interval yöntemi ve holdout ayları için tek freeze manifesti üret. Holdout'a erişmeden evaluation origin takvimini kilitle.

**Önerilen geometri:** 2025 hedef aylarını h1/h3/h6 için ayrı rolling origins ile değerlendir. Ocak/Şubat gibi erken h3/h6 hedeflerinin origin'i 2024 içindedir; o origin için training verisi 2024-12'ye kadar uzatılamaz. Her tahminde yalnız origin anında erişilebilir gözlemler kullanılmalı. İlk 2024-12 origin'den itibaren başlatılacaksa horizon başına kapsanan holdout hedef sayısı azalır ve açık raporlanır.

**Çözülmesi gereken sözleşme:** Parametre/champion freeze ile model refit farklı işlemlerdir. Geçmiş holdout ayları gözlendikçe eğitim/lag input olarak kullanılacak mı? Mevcut protokolün “refit using development data” hükmüyle bu öneri arasında yorum farkı varsa yazılı amendment gerekir. Strict fixed-development alternatifinde tek-origin h1/h3/h6 sonucunu “12 ayın tamamı değerlendirildi” diye adlandırma. Hangi yol seçilirse seçilsin sonuç görüldükten sonra değiştirilemez.

**Kalibrasyon:** Final aralıklar sadece development residual havuzundan; origin'e göre residual availability kontrolü sürer. Erken holdout hedefi için 2024 sonundaki henüz bilinmeyen residual kullanılmaz. 2025 artıklarını interval tuning/calibration'a ekleme.

**Kabul:** Origin/target/train/calibration takvimi machine-readable ve testli; actual join sadece scoring aşamasında; bilgi erişim iddiası dürüst; bütün belirsizlikler yazılı karara bağlanmış. Bu kapı kapanmadan holdout açılmaz.

### D — M11 final holdout ve araştırma sonucu

**İş:** Kilitli champion ve protokolle final değerlendirmeyi çalıştır. Dört modelin kilitli benchmark karşılaştırmaları da raporlanabilir; bunlar champion'u yeniden seçmez. Validation ve holdout çıktıları ayrı tutulur.

**Çıktı:** Origin-level holdout predictions, 42 grup için kapsanan observation sayıları ve sMAPE/MAE/RMSE, interval coverage/width, champion vs Seasonal Naive farkı, snapshot/run metadata, `M11_HOLDOUT_REPORT.md`.

**Kabul:** Model/config hashleri freeze ile eşleşir; holdout tuning/calibration yok; iyi, zayıf ve null sonuçlar birlikte yer alır; başarısız tahminler denominatordan sessizce kaybolmaz; rapor yeniden üretilebilir. Güncel 2026 veri ihtiyacı ayrı production snapshot'ta ele alınır; frozen benchmark dosyaları overwrite edilmez.

### E — M12 için serving artifact ve PostgreSQL katmanı

**İş:** Backtest, final evaluation ve ileriye dönük serving üretimini ayır. Onaylı serving snapshot'ta champion modelleri eğit/kaydet; model loader ve predict arayüzünü oluştur. Tek yeni origin için 14×3=42 forecast kaydı veya açık unavailable status üret. XGBoost olmayan champion için SHAP uydurma; Model Lab'de XGBoost'un ayrı açıklamaları erişilebilir kalsın.

**Şema:** `data_snapshots`, `trade_observations`, `mirror_statistics`, `macro_observations`, `selected_sectors`, `model_runs`, `model_metrics`, `champions`, `forecasts`, `shap_contributions`. Gerekçesiz ek tablolar/features oluşturma. Snapshot/run ve domain key unique constraints; ingestion/upsert idempotent; serving reads için direction/sector/period/run indexleri.

**Sayı sözleşmesi:** Resmi parasal kaynak değerlerinde exact decimal veya integer minor units; integer dönüşümünde kaynak hassasiyeti korunmalı. Model fitting/inference için sayısal float dönüşümü açık model sınırında yapılır; tahminlerin hesaplaması tam sayı aritmetiğine zorlanmaz. UTC-aware işlem timestamp'leri, ayrıca aylık reference period tutulur.

**Kabul:** Migration + küçük seed; reload sonrası aynı tahmin; duplicate import yok; atomic publication ile yarım run görünmez; demo read-only; Supabase service credential istemciye ulaşmaz. Eğitim ve harici ingestion API request içinde çalışmaz. Tam veri için yetkili acquisition, demo için lisansı doğrulanmış küçük snapshot akışı belgelenir.

### F — M12 FastAPI ürün API'si

**Minimum read endpoints:** `/api/v1/overview`, `/trade`, `/forecasts`, `/metrics`, `/sectors`, `/mirror`, `/indicators`, `/methodology`.

**İş:** Typed response modelleri, direction/HS2/horizon/snapshot validation, history pagination ve date bounds, controller→service→repository ayrımı, structured errors, ortam bazlı CORS ve cache. Overview aggregate'larında TOTAL ile sektörler toplanarak double count yapılmaz.

**Response metadata:** source/reporter/flow, observation cutoff, retrieval time, snapshot/run, forecast origin/target, champion, unit ve interval availability. `available`, `unavailable`, `stale` gibi durumlar açık ve tutarlı olmalı; missing değer sıfır yapılmaz. Freshness duvar saati ile source reporting cutoff ayrılarak hesaplanır.

**Kabul:** Contract/integration testleri; invalid horizon/sector; empty data/stale snapshot; DB unavailable; deterministic sample response; OpenAPI. `/health` ile DB/data readiness ayrılır. Public API yazma/training/ingestion endpoint açmaz.

### G — M13 TR/EN/RU dashboard

**İlk uçtan uca dilim:** Overview → direction/sector/horizon seçimi → tarihçe ve tahmin grafiği → 80/95 bands → champion/metric/snapshot açıklaması. Sonra kalan modüller:

| Modül | Kullanıcıya verdiği cevap |
| --- | --- |
| Overview | Hangi iki akım izleniyor, hangi dönem ve snapshot mevcut? |
| Trade Explorer | TOTAL ve seçilmiş sektörlerin aylık gelişimi nasıl? |
| Forecasts | Seçilen seri için nokta tahmin, interval ve model nedir? |
| Sector Analysis | Bu sektör neden seçildi, hangileri hangi gerekçeyle elendi? |
| Model Lab | Dört model validation/holdout'ta nasıl karşılaştırılıyor; XGBoost drivers neler? |
| Mirror Statistics | İki reporter arasındaki fark ve eksik dönemler neler? |
| Economic Indicators | Kullanılan beş gösterge ve kaynak/yayın gecikmeleri neler? |
| Methodology | Target politikası, origin, validation, holdout ve sınırlamalar nasıl tanımlandı? |

**İş:** Gerçek locale switching/routing, ortak typed messages, locale number/date formatting, URL'de paylaşılabilir filters, responsive layout, accessible controls ve grafik için tablo alternatifi. Actual / historical backtest / future serving forecast ayrımı görünür olsun. Negative lower bounds'u sessizce sıfıra kırpma; raw değer ve yorum politikasını göster.

**Kabul:** API ile gerçek bağlantı; TR/EN/RU çekirdek metinlerin tamamı; reload'da seçimler; empty/loading/error/stale durumları; mirror olmayan aylar açık; mobile/desktop E2E. XGBoost nominal/empirical coverage ayrımı ve SARIMAX persistence scenario açıklaması görünür.

### H — M14 araştırma ve portfolyo paketi

**Teslimler:** Güncel İngilizce README; kısa Türkçe proje özeti ve Rusça abstract; architecture/dataflow diagram; data dictionary; kaynak/lisans/retrieval notları; methodology; validation/holdout model results; limitations; reproducibility guide; demo ekranları ve kısa video; sürümlenmiş final research report.

**Research report:** Existing M14 15-bölümlü paper yapısı korunur. Başlık/abstract gerçek sonuçlara göre yazılır; hakem değerlendirmesi olmayan belge “working paper / technical research report” olarak sunulur. Literatür taraması birincil araştırma ve resmi veri kaynaklarıyla yapılır; kaynaklar uydurulmaz.

**Anlatı:** Makine öğrenmesi her zaman kazanıyor iddiası yerine araştırma sorusu → veri sorunu ve v1.1 target kararı → leakage/availability önlemleri → modellerin karşılaştırması → sektör/horizon farkları → holdout → uncertainty ve sınırlar. Rus mirror boşluğuna kanıtsız politik/idarî neden atfetme.

**Ek analiz:** Makronun katkısı ayrı controlled ablation ile daha güçlü ölçülebilir. Yapılacaksa development-only tasarımı M11 öncesinde dondurulur; final holdout görülünce yeni özellik/model seçtirmek için kullanılmaz. Mevcut sonuçlar yalnız SARIMA vs mevcut SARIMAX/XGBoost yapılandırmalarının performansını destekler.

**Portfolyo kartı:** Gerçek katkılar, stack, GitHub, çalışan demo, research report, 2–3 somut bulgu, dürüst sınırlamalar. “Accuracy %” üretme; sMAPE ve göreli improvement açık kullanılsın. README ilk ekranında problem, screenshot, sonuç ve çalıştırma yolunu görsün.

**Kabul:** Başka bir geliştirici offline sample ile credential olmadan ürünü açar; full experiment için gereken source access ayrıca anlatılır. Lisans/redistribution izni doğrulanmamış büyük kaynak paketleri public repo/release'e eklenmez. Sonuç tablosu artifact'lerden otomatik üretilir.

### I — M15 release, deployment ve son kullanıcı kabulü

**İş:** Python unit/integration + long experiment doğrulaması, lint/type checks, frontend build/E2E, migration/seed, secret/dependency checks ve docs consistency CI. Docker/local setup ve production config; FastAPI, Next.js, PostgreSQL/Supabase ortamları; deploy health/readiness, timeout/cache, logging ve rollback/runbook.

**Kabul:** Temiz clone kurulum; sample demo ve full reproduction yolları; tüm MUST maddeleri için test/artifact/ekran kanıtı; 14 seri×3 horizon kullanılabilir veya gerekçeli unavailable; 80/95 interval politikası; model registry/API/UI uyuşması; üç dil; kaynak tarihi/uncertainty görünür; live smoke/UAT ayrı kayıt; kalıcı demo bağlantısı.

**Release:** Semver tag, changelog, artifact manifest/checksums, final rapor ve demo linki. Push/publish/deployment bu plan kapsamında yapılmaz; ayrı uygulama görevinin yayın yetkisiyle yürütülür.

## 6. Öncelik, efor ve bağımlılıklar

| İş paketi | Tahmini odaklı efor | Bağımlılık |
| --- | --- | --- |
| A: M9 review + provenance/test/availability | 3–5 gün | Mevcut değişikliklerin korunması |
| B: M10 champion + baseline intervals | 2–3 gün | A |
| C–D: M11 protokol freeze + holdout | 3–5 gün | B, origin/refit kararının çözülmesi |
| E: serving artifacts + DB | 2–4 gün | Champion ve final deney sözleşmesi |
| F: ürün API'si | 3–4 gün | E |
| G: çok dilli dashboard | 6–9 gün | F contract; görsel taslak daha erken hazırlanabilir |
| H: araştırma/portfolyo paketi | 4–6 gün | Final sonuçlar; kaynak/architecture taslağı daha erken |
| I: release/CI/deploy/UAT | 3–5 gün | E–H |

Toplam **26–41 odaklı çalışma günü** planlama tahminidir; compute bekleme süreleri, tarihsel vintage temini ve erişim/lisans belirsizlikleri dahil değildir. Yarı zamanlı çalışma takviminde daha uzun sürer. İlk A kapısından sonra tahmin güncellenmeli; veri/deney doğruluğu çözülmeden sabit bitiş tarihi vaat edilmemeli.

Kritik yol: A → B → C → D → E → F → G → I. Research draft ve API/UI contract tasarımı sonuçları görmeden yapılabilir; frozen holdout'a erken erişim gerektirmez. Her görev allowed files, non-goals, testler, teslim formatı ve stop condition içermeli.

## 7. Bu incelemenin doğrulama kaydı

- `ruff check .`: geçti.
- `frontend npm run build`: geçti; sadece mevcut tek-sayfa skeleton'ın build/type doğrulamasıdır, ürün kabulü değildir.
- Tam pytest koşusu: 135 test geçti; M7 fixture'ın canonical output'a yazma davranışı tespit edilince çalışma durduruldu. Full-suite GREEN sonucu yok. Mevcut M6/M7 run dosyalarının mtimeleri 1 Ekim 2026 olarak kaldı; bu durdurulan koşuda bunların overwrite edildiğine dair bulgu yok.
- M9 hedefli test koşusu: `pytest tests/unit/test_xgboost.py -q -o addopts=''` → **37 passed**, 184,77 saniye; bir mevcut FastAPI/Starlette test-client deprecation uyarısı. Bu bütün model zincirinin temiz ortam kabulü değildir.
- `git diff --check`: önceden değiştirilmiş `test_features.py` ve `test_seasonal_naive.py` dosyalarında EOF blank-line uyarıları var; bu incelemede mevcut dosyalara müdahale edilmedi.
- Mevcut development artifacts doğrudan okundu; dört modelin 3.934'er non-null tahmini ve 42'şer metrik grubu doğrulandı. M8'in 1.344 fit olması hata değildir: bir seri/origin fit'i birden fazla horizon için paylaşılır.
- Yeni ingestion, kaynak veri refresh'i, M10 seçimi, M11 scoring, DB migration, deployment, commit veya push yapılmadı. Yalnız bu plan dosyası eklendi; test/build araçları geçici cache/build dosyaları üretir.

## 8. Tamamlanmış sayılma sınırı

V1 ancak bağımsız holdout raporu, deterministik champion registry, gerçek artifact inference/serving, PostgreSQL katmanı, bağlı FastAPI/Next.js ürün akışı, TR/EN/RU temel deneyim, yeniden üretilebilir araştırma paketi ve canlı release kabulü birlikte mevcut olduğunda tamamlanmıştır. Sadece model çıktısı, başarılı build veya görsel dashboard bu sınırı karşılamaz.

V1'e deep learning, başka ülkeler, HS6, mobil uygulama, auto-trading, LLM tahminleri, causal claims ve günlük retraining eklenmez. Bunlar bu projenin bitmesini uzatır ve mevcut spec'in dışındadır.

### 14 Ekim 2026 — M10 uygulama kaydı

Kullanıcının doğrudan M10 uygulama isteği kapsamında B1–B5 yürütüldü. Bu ilerleme A2 bakım bulgularını veya M8 run metadata'sındaki validation YAML hash farkını kapatmaz. Tam deney kabulü ve M11 kapısı açık önkoşullar nedeniyle kapalıdır. Seçim mevcut v1.1 protokolündeki 96/94/91 aylık origin, MAE tie-break ve yüzde 2 göreli parsimony kuralını korur; prompttaki beş fold varsayımı uygulanmadı. Doğrulama komutları ve test sonuçları `docs/M10_ACCEPTANCE_REPORT.md` içinde kayıtlıdır.

### 16 Ekim 2026 — Stage A ve M11 teslim doğrulaması

M9 A2 kapandı; httpx2 ile uyarı kaynağında giderildi, global deprecation filtresi kaldırıldı. M8 eski metadata hash'i değiştirilmeden takvim/hedef eşdeğerliği gate kuralı kabul edildi. Kullanıcının açık kapsam düzeltmesiyle 2025 holdout değerlendirmesi retrospective latest-vintage/deterministik lag kuralları altında tamamlandı. Tüm 42 görevde 12 hedef, üç rol, Conformal havuz cutoff, SHAP, ham negatif tahminler ve kayıtlı fallback davranışı raporlandı. İlk tam suite 441 passed / 0 warnings; teslim tarihi hedefli kontrol 31 passed / 0 warnings. Ayrıntılı kanıt `docs/M11_ACCEPTANCE_REPORT.md` içindedir. M12'ye otomatik geçilmedi.

### 16 Ekim 2026 — M12 read-only artifact API

Kullanıcının açık disk-artifact sunum kapsamıyla typed FastAPI API uygulandı. 20 dosyalık serving manifesti, M10/M11 upstream checksum çapraz kontrolü, immutable snapshot, global/task CV/holdout metrikleri, ham/operasyonel fallback tahminleri, 9.072 gerçek SHAP değeri ve mirror veri boşlukları endpoint sözleşmesine taşındı. Hedefli kabul: 70 API testi, 0 uyarı; tam suite kabulü `M12_ACCEPTANCE_REPORT.md` içinde. Yerel Uvicorn TCP smoke geçti; production deploy/M13 browser UAT yapılmadı. E1–E2 model/DB işleri bu kullanıcının M12 isteğinden ayrı kalır.

### 16 Ekim 2026 — M13 kabulü ve M14 portfolyo vitrini

M13 kabul raporu, beş modüllü Next.js arayüzünü, canlı API/doğrulanmış snapshot modlarını, deterministik snapshot hash'ini ve 511 testlik tam suite'i kaydeder. Bu teslim yerel build ve browser smoke kanıtıdır; hosted deployment ve gerçek kullanıcı UAT değildir. M11 kapsamı README ve API/UI boyunca **retrospective latest-vintage holdout evaluation under deterministic lag rules** olarak açıklanır; historical publication-time archive ve macro vintage replay kanıtı değildir.

M14 kapsamında README; doğrulanmış metrikler, metodoloji ve kaynak sınırları, mimari diyagram, API/dashboard kullanım yolu, offline build notu ve Pre-M3/M3–M13 rapor diziniyle yenilendi. M10–M13 durumları milestone/specification kayıtlarında eşitlendi. Daha geniş M14 çıktıları (research report/working paper, Türkçe özet, Rusça abstract, veri sözlüğü ve kaynak/lisans paketlemesi) tamamlanmadığından M14 bütünü açık kalır; M15 release/deploy/UAT ayrıca değerlendirilir.
