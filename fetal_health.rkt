#lang racket

;; Proje: Fetal Sağlık Sınıflandırması (K-NN Algoritması)
;; Amaç: Fetal sağlık durumunu (Normal, Şüpheli, Patolojik) tahmin etmek.

;; ---------------------------------------------------------------------
;; 1. VERİ YAPILARI (DATA STRUCTURES)
;; ---------------------------------------------------------------------

;; Bir veri noktasını temsil eden yapı.
;; features: 21 adet sayısal özellikten oluşan bir liste.
;; label: 1.0 (Normal), 2.0 (Şüpheli) veya 3.0 (Patolojik) değerlerinden biri.
(struct data-point (features label) #:transparent)

;; ---------------------------------------------------------------------
;; 2. CSV OKUMA VE AYRIŞTIRMA (CSV READING & PARSING)
;; ---------------------------------------------------------------------

;; Bir satırı (string) alıp data-point yapısına dönüştürür.
;; Virgülle ayrılmış değerleri parçalar ve sayıya çevirir.
(define (parse-line line)
  (let* ([tokens (string-split line ",")] ;; Virgülle ayır
         [numbers (map string->number tokens)]) ;; Sayıya çevir
    ;; Son eleman etiket, geri kalanlar özelliklerdir.
    ;; take ve drop fonksiyonları ile ayırıyoruz.
    ;; Toplam 22 kolon var: 21 özellik + 1 etiket.
    (let ([feats (take numbers 21)]
          [lbl (last numbers)])
      (data-point feats lbl))))

;; CSV dosyasını okuyup data-point listesi döndürür.
;; Başlık satırını atlar.
(define (load-data filepath)
  (let* ([lines (file->lines filepath)] ;; Tüm satırları oku
         [data-lines (cdr lines)]) ;; İlk satırı (başlık) atla
    (map parse-line data-lines))) ;; Her satırı parse et

;; ---------------------------------------------------------------------
;; 4. NORMALİZASYON (MIN-MAX SCALING)
;; ---------------------------------------------------------------------

;; Özellik listelerinin transpozunu alır (satırları sütuna çevirir)
;; Bu sayede her özellik için min/max hesaplayabiliriz.
(define (transpose matrix)
  (apply map list matrix))

;; Bir listenin min ve max değerlerini bulur.
(define (get-min-max lst)
  (cons (apply min lst) (apply max lst)))

;; Tüm veri setindeki özellikler için min ve max değerlerini hesaplar.
(define (calculate-feature-bounds data)
  (let* ([all-features (map data-point-features data)] ;; Sadece özellik vektörlerini al
         [columns (transpose all-features)])           ;; Sütunlara çevir
    (map get-min-max columns)))                        ;; Her sütun için min-max bul

;; Tek bir değeri normalize eder: (x - min) / (max - min)
;; Eğer max = min ise (sabit özellik), 0 döndürür.
(define (normalize-value val min-val max-val)
  (if (= max-val min-val)
      0.0
      (/ (- val min-val) (- max-val min-val))))

;; Bir özellik vektörünü normalize eder.
(define (normalize-vector features bounds)
  (map (lambda (val bound)
         (normalize-value val (car bound) (cdr bound)))
       features
       bounds))

;; Tüm veri noktalarını normalize eder.
(define (normalize-dataset data bounds)
  (map (lambda (dp)
         (data-point (normalize-vector (data-point-features dp) bounds)
                     (data-point-label dp)))
       data))

;; ---------------------------------------------------------------------
;; 6. K-NN ALGORİTMASI (TEMEL MANTIK)
;; ---------------------------------------------------------------------

;; İki vektör arasındaki Öklid mesafesini hesaplar.
(define (euclidean-distance v1 v2)
  (sqrt (apply + (map (lambda (x y) (sqr (- x y))) v1 v2))))

;; Bir liste içindeki elemanların frekansını bulur (Etiket oylaması için)
;; Örn: '(1 1 2 1 3) -> '((1 . 3) (2 . 1) (3 . 1))
(define (count-occurrences lst)
  (foldl (lambda (x acc)
           (let ([existing (assoc x acc)])
             (if existing
                 (cons (cons x (+ 1 (cdr existing))) (remove existing acc))
                 (cons (cons x 1) acc))))
         '()
         lst))

;; En çok tekrar eden etiketi bulur.
(define (majority-vote labels)
  (let* ([counts (count-occurrences labels)]
         ;; Sayıya göre çoktan aza sırala
         [sorted (sort counts > #:key cdr)]) 
    (car (first sorted)))) ;; En üsttekinin etiketini al

;; K-NN Tahmin Fonksiyonu
;; k: Komşu sayısı
;; train-data: Eğitim veri seti (data-point listesi)
;; query-features: Sınıflandırılacak özellik vektörü
(define (knn-predict k train-data query-features)
  (let* ([distances 
          (map (lambda (dp)
                 (cons (euclidean-distance (data-point-features dp) query-features)
                       (data-point-label dp)))
               train-data)]
         ;; Mesafeye göre küçükten büyüğe sırala
         [sorted-distances (sort distances < #:key car)]
         ;; İlk k kalemi al
         [k-nearest (take sorted-distances k)]
         ;; Sadece etiketleri al
         [k-labels (map cdr k-nearest)])
    ;; Oylama yap ve sonucu dön
    (majority-vote k-labels)))

;; ---------------------------------------------------------------------
;; 7. DEĞERLENDİRME (EVALUATION)
;; ---------------------------------------------------------------------

;; Veri setini karıştırıp (shuffle), eğitim ve test olarak böler.
;; split-ratio: Eğitim verisinin oranı (örn: 0.8)
(define (split-train-test data split-ratio)
  (let* ([shuffled (shuffle data)]
         [n (length shuffled)]
         [train-count (exact-floor (* n split-ratio))]
         [train-set (take shuffled train-count)]
         [test-set (drop shuffled train-count)])
    (values train-set test-set)))

;; Modelin doğruluğunu hesaplar.
(define (calculate-accuracy k train-set test-set)
  (let ([correct-predictions
         (for/sum ([item test-set])
           (let* ([actual (data-point-label item)]
                  [predicted (knn-predict k train-set (data-point-features item))])
             (if (= actual predicted) 1 0)))])
    (/ correct-predictions (length test-set))))

;; ---------------------------------------------------------------------
;; 8. UYGULAMA AKIŞI GÜNCELLEMESİ
;; ---------------------------------------------------------------------

;; Test için veri setini yükle
(define dataset-path "fetal_health.csv")

(define feature-bounds '())
(define train-data '())
(define test-data '())

;; ---------------------------------------------------------------------
;; 9. REMOVED REPL FOR GUI
;; ---------------------------------------------------------------------

(require racket/gui/base)

;; Özellik İsimleri (Giriş alanları için)
(define feature-names
  '("Bazal Kalp Hızı (Baseline Value)" 
    "Hızlanmalar (Accelerations)" 
    "Fetal Hareket (Fetal Movement)" 
    "Rahim Kasılmaları (Uterine Contractions)"
    "Hafif Yavaşlamalar (Light Decelerations)" 
    "Şiddetli Yavaşlamalar (Severe Decelerations)" 
    "Uzun Süreli Yavaşlamalar (Prolongued Decelerations)"
    "Anormal Kısa Dönem Değişkenlik (Abnormal Short Term Variability)" 
    "Kısa Dönem Değişkenlik Ortalaması (Mean Value of Short Term Variability)"
    "Anormal Uzun Dönem Değişkenlik Yüzdesi (Percentage of Time with Abnormal Long Term Variability)"
    "Uzun Dönem Değişkenlik Ortalaması (Mean Value of Long Term Variability)" 
    "Histogram Genişliği (Histogram Width)" 
    "Histogram Min Değeri"
    "Histogram Max Değeri" 
    "Histogram Tepe Sayısı (Number of Peaks)" 
    "Histogram Sıfır Sayısı (Number of Zeroes)"
    "Histogram Mod" 
    "Histogram Ortalama (Mean)" 
    "Histogram Medyan" 
    "Histogram Varyans"
    "Histogram Eğilimi (Tendency)"))


;; ---------------------------------------------------------------------
;; 10. GRAFİK ARAYÜZ (GUI)
;; ---------------------------------------------------------------------

(define (start-gui)
  ;; Ana Pencere
  (define frame (new frame% 
                     [label "Fetal Sağlık Analizi"]
                     [width 500]
                     [height 700]))

  ;; Ana Panel (Dikey)
  (define main-panel (new vertical-panel% [parent frame]))

  ;; Başlık Mesajı
  (new message% [parent main-panel] 
       [label "Lütfen analiz için aşağıdaki 21 parametreyi giriniz:"]
       [auto-resize #t])

  ;; Kaydırılabilir Panel (Giriş Alanları için)
  ;; (scrollable-panel sınıfı Racket GUI'de doğrudan yok, panel% içinde style ile yapılır veya canvas kullanılır
  ;;  ancak basitlik için normal bir panel ve ekran sığmazsa diye dışına scroll bar koymak biraz karmaşık olabilir Racket'ta.
  ;;  Bu yüzden basitçe vertical-panel kullanacağız, eğer sığmazsa layout otomatik ayarlar.)
  ;; *Düzeltme*: Auto-scroll style pane'e eklenebilir.
  (define scroll-panel (new vertical-panel% 
                            [parent main-panel]
                            [style '(auto-vscroll)]))

  ;; Giriş Alanlarını Saklamak İçin Liste
  (define input-fields '())

  ;; Özellik İsimlerine Göre Giriş Alanlarını Oluşturma
  (for ([name feature-names])
    (let ([tf (new text-field% 
                   [parent scroll-panel] 
                   [label name]
                   [init-value ""])])
      (set! input-fields (append input-fields (list tf)))))

  ;; Buton Paneli
  (define btn-panel (new horizontal-panel% 
                         [parent main-panel]
                         [alignment '(center center)]
                         [stretchable-height #f]))

  ;; Hızlı Veri Girişi Butonu Aksiyonu
  (define (on-quick-input-click button event)
    (let ([input-str (get-text-from-user "Hızlı Veri Girişi" 
                                         "Lütfen 21 adet sayısal değeri aralarına boşluk koyarak tek satırda yapıştırınız:" 
                                         frame)])
      (when input-str
        (with-handlers ([exn:fail? (lambda (e) 
                                     (message-box "Hata" "Geçersiz format! Lütfen sadece sayı giriniz." frame '(ok stop)))])
          (let* ([tokens (string-split input-str)]
                 [numbers (map string->number tokens)])
            (if (= (length numbers) 21)
                (for ([tf input-fields]
                      [val numbers])
                  (send tf set-value (number->string val)))
                (message-box "Hata" (format "21 adet değer girilmeli. Girilen: ~a" (length numbers)) frame '(ok stop))))))))

  ;; Analiz Butonu Aksiyonu
  (define (on-analyze-click button event)
    (with-handlers ([exn:fail? (lambda (e) 
                                 (message-box "Hata" 
                                              "Lütfen tüm alanlara geçerli sayısal değerler giriniz!" 
                                              frame 
                                              '(ok stop)))])
      ;; Tüm girdileri al ve sayıya çevir
      (let* ([raw-values (map (lambda (tf) (string->number (send tf get-value))) input-fields)])
        ;; Kontrol: Herhangi biri sayı değilse hata fırlatır (map içinde string->number #f dönerse sorun olabilir)
        (if (member #f raw-values)
            (error "Geçersiz giriş")
            (begin
              ;; Tahmin Yap
              ;; (predict-risk fonksiyonunu normalize ve tahmin için kullanacağız, ancak önce feature-bounds lazım)
              (let* ([normalized-input (normalize-vector raw-values feature-bounds)]
                     [prediction (knn-predict 5 train-data normalized-input)]
                     [result-text 
                      (cond
                        [(= prediction 1.0) "NORMAL\n(Risk Seviyesi Düşük)"]
                        [(= prediction 2.0) "ŞÜPHELİ (SUSPECT)\n(Takip Önerilir)"]
                        [(= prediction 3.0) "PATOLOJİK (PATHOLOGICAL)\n(Acil Müdahale Gerekebilir)"]
                        [else "BİLİNMEYEN"])])
                (message-box "Analiz Sonucu" 
                             (format "Tahmin Edilen Fetal Sağlık Durumu:\n\n~a" result-text) 
                             frame 
                             '(ok))))))))

  ;; Korelasyon Görselini Açma Butonu Aksiyonu
  (define (on-open-matrix-click button event)
    (if (file-exists? "correlation_matrix.png")
        (shell-execute "open" "correlation_matrix.png" "" (current-directory) 'sw_show)
        (message-box "Hata" "Görsel dosyası (correlation_matrix.png) bulunamadı! Lütfen önce generate_matrix.rkt dosyasını çalıştırın." frame '(ok stop))))

  ;; Butonlar
  (new button% [parent btn-panel]
       [label "HIZLI GİRİŞ (YAPIŞTIR)"]
       [min-width 150]
       [min-height 40]
       [callback on-quick-input-click])

  (new button% [parent btn-panel]
       [label "KORELASYON MATRİSİ (GÖRSEL)"]
       [min-width 150]
       [min-height 40]
       [callback on-open-matrix-click])

  (new button% [parent btn-panel]
       [label "ANALİZ ET"]
       [min-width 150]
       [min-height 40]
       [callback on-analyze-click])



  ;; Pencereyi Göster
  (send frame show #t))

;; ---------------------------------------------------------------------
;; 11. MAIN (BAŞLANGIÇ)
;; ---------------------------------------------------------------------

(define (main)
  ;; 1. Veriyi Oku
  (displayln "1. Veri seti yükleniyor...")
  (define raw-data
    (with-handlers ([exn:fail:filesystem? (lambda (e) 
                                            (displayln "Hata: Dosya bulunamadı! Lütfen 'fetal_health.csv' dosyasını aynı dizine koyun.") 
                                            '())])
      (load-data dataset-path)))

  (when (not (null? raw-data))
    (printf "   -> Toplam veri sayısı: ~a\n" (length raw-data))

    ;; 2. Min/Max Değerlerini Hesapla
    (displayln "2. İstatistikler hesaplanıyor (Min-Max)...")
    (set! feature-bounds (calculate-feature-bounds raw-data))

    ;; 3. Veriyi Normalize Et
    (displayln "3. Veri normalize ediliyor...")
    (define normalized-data (normalize-dataset raw-data feature-bounds))
    
    ;; 4. Eğitim ve Test Olarak Ayır (%80 - %20)
    (displayln "4. Veri seti ayrıştırılıyor (Eğitim: %80, Test: %20)...")
    (define-values (tr te) (split-train-test normalized-data 0.8))
    (set! train-data tr)
    (set! test-data te)
    (printf "   -> Eğitim Seti: ~a\n" (length train-data))
    (printf "   -> Test Seti:    ~a\n" (length test-data))

    ;; 5. Modeli Değerlendir (K=5)
    (displayln "5. Model doğruluğu test ediliyor (K=5)...")
    (displayln "   (Bu işlem biraz zaman alabilir, lütfen bekleyin...)")
    (define acc (calculate-accuracy 5 train-data test-data))
    (printf "   -> MODEL DOĞRULUĞU (ACCURACY): ~a%\n" (~r (* acc 100.0) #:precision 2))
    
    ;; 6. GUI Başlat
    (displayln "6. Grafik Arayüz (GUI) başlatılıyor...")
    (start-gui)))

;; Programı çalıştır
(main)


