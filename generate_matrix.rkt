#lang racket

(require plot)
(require racket/math)
(require racket/list)
(require racket/draw)

;; ---------------------------------------------------------------------
;; AYARLAR VE VERİ YÜKLEME
;; ---------------------------------------------------------------------

(define dataset-path "fetal_health.csv")

;; Özellik İsimleri (Kısa halleri, görsellik için)
(define feature-names
  '("Baseline" "Accel" "Movement" "Contract"
    "Light Dec" "Severe Dec" "Prolong Dec"
    "Abnorm STV" "Mean STV"
    "Abnorm LTV" "Mean LTV" 
    "Width" "Min" "Max" 
    "Peaks" "Zeroes"
    "Mode" "Mean" "Median" "Variance" "Tendency"))

(define (parse-line line)
  (let* ([tokens (string-split line ",")]
         [numbers (map string->number tokens)])
    (take numbers 21))) ;; Sadece özellikleri al (ilk 21)

(define (load-data filepath)
  (let* ([lines (file->lines filepath)]
         [data-lines (cdr lines)]) ;; Başlığı atla
    (map parse-line data-lines)))

(define (transpose matrix)
  (apply map list matrix))

(define (sum lst) (apply + lst))

;; Pearson Korelasyonu
(define (pearson-correlation xs ys)
  (let* ([n (length xs)]
         [sum-x (sum xs)]
         [sum-y (sum ys)]
         [sum-x-sq (sum (map sqr xs))]
         [sum-y-sq (sum (map sqr ys))]
         [sum-xy (sum (map * xs ys))]
         [numerator (- (* n sum-xy) (* sum-x sum-y))]
         [denominator (sqrt (* (- (* n sum-x-sq) (sqr sum-x))
                               (- (* n sum-y-sq) (sqr sum-y))))])
    (if (zero? denominator) 0.0 (/ numerator denominator))))

;; Rengi hesapla (-1: Mavi, 0: Beyaz, +1: Kırmızı)
(define (get-color val)
  (cond
    [(> val 0) (list 1.0 (- 1.0 val) (- 1.0 val))] ;; Kırmızı tonları
    [(< val 0) (list (+ 1.0 val) (+ 1.0 val) 1.0)] ;; Mavi tonları
    [else (list 1.0 1.0 1.0)])) ;; Beyaz

;; ---------------------------------------------------------------------
;; MATRİS OLUŞTURMA VE ÇİZİM
;; ---------------------------------------------------------------------

(define (main)
  (displayln "Veri yükleniyor...")
  (define data (load-data dataset-path))
  (define columns (transpose data))
  
  (displayln "Korelasyonlar hesaplanıyor...")
  (define num-features (length columns))
  
  (define rects
    (flatten
     (for/list ([i (in-range num-features)])
       (for/list ([j (in-range num-features)])
         (let* ([col-i (list-ref columns i)]
                [col-j (list-ref columns j)]
                [corr (pearson-correlation col-i col-j)]
                [color-rgb (get-color corr)])
           ;; Dikdörtgen çiz: xmin xmax ymin ymax
           (rectangles (list (vector (ivl j (+ j 1)) (ivl i (+ i 1))))
                       #:color (make-color (exact-floor (* 255 (first color-rgb)))
                                           (exact-floor (* 255 (second color-rgb)))
                                           (exact-floor (* 255 (third color-rgb))))
                       #:line-color "white"
                       #:line-width 1
                       #:alpha 1.0))))))

  (displayln "Görsel çiziliyor ve kaydediliyor...")
  
  (define features-count (length feature-names))
  
  ;; Y ekseni için isimleri ters çevir (Görselde 0 aşağıda, biz en üstte 0 olsun istiyoruz)
  ;; Ancak koordinat düzleminde y'yi ters çevirerek çizeceğiz.
  ;; i=0 (İlk özellik) -> Y=20
  ;; i=20 (Son özellik) -> Y=0
  
  ;; Ticks oluşturucu fonksiyon
  (define (custom-ticks-layout min max)
    (for/list ([i (in-range features-count)])
      (pre-tick (+ i 0.5) #t)))

  ;; X ekseni etiket formatlayıcı (0.5 -> İsim[0])
  (define (x-tick-format min max ticks)
    (for/list ([t ticks])
      (let ([idx (exact-floor (pre-tick-value t))])
        (if (< idx features-count)
            (list-ref feature-names idx)
            ""))))
            
  ;; Y ekseni etiket formatlayıcı (Geriye doğru mapping)
  ;; y=0.5 -> Son eleman (Index 20)
  ;; y=20.5 -> İlk eleman (Index 0)
  (define (y-tick-format min max ticks)
    (for/list ([t ticks])
      (let* ([val (pre-tick-value t)]
             [inverted-idx (- (sub1 features-count) (exact-floor val))])
        (if (and (>= inverted-idx 0) (< inverted-idx features-count))
            (list-ref feature-names inverted-idx)
            ""))))

  (parameterize ([plot-x-ticks (ticks custom-ticks-layout x-tick-format)]
                 [plot-y-ticks (ticks custom-ticks-layout y-tick-format)]
                 [plot-font-size 9]) ;; Genel font boyutu
    (plot-file
     (list
      ;; 1. Dikdörtgenler (Isı Haritası)
      ;; i=satır, j=sütun
      (flatten
       (for/list ([i (in-range num-features)])
         (for/list ([j (in-range num-features)])
           (let* ([col-i (list-ref columns i)]
                  [col-j (list-ref columns j)]
                  [corr (pearson-correlation col-i col-j)]
                  [color-rgb (get-color corr)]
                  ;; Y koordinatını ters çevir: i=0 -> y=20, i=20 -> y=0
                  [y-pos (- (sub1 num-features) i)])
             (rectangles (list (vector (ivl j (+ j 1)) (ivl y-pos (+ y-pos 1))))
                         #:color (make-color (exact-floor (* 255 (first color-rgb)))
                                             (exact-floor (* 255 (second color-rgb)))
                                             (exact-floor (* 255 (third color-rgb))))
                         #:line-color "white"
                         #:line-width 1
                         #:alpha 1.0)))))
      
      ;; 2. Değerler (Point Label)
      (flatten
       (for/list ([i (in-range num-features)])
         (for/list ([j (in-range num-features)])
           (let* ([col-i (list-ref columns i)]
                  [col-j (list-ref columns j)]
                  [corr (pearson-correlation col-i col-j)]
                  ;; Metin pozisyonu (karenin merkezi)
                  [y-pos (- (sub1 num-features) i)])
             (point-label (vector (+ j 0.5) (+ y-pos 0.5)) 
                          (~r corr #:precision 2) 
                          #:anchor 'center
                          #:point-sym 'none ;; Noktayı gizle
                          #:size 8)))))) ;; Yazı boyutu
     
     "correlation_matrix.png"
     #:title "Fetal Sağlık Özellikleri - Korelasyon Matrisi"
     #:x-label #f 
     #:y-label #f 
     #:x-min 0 #:x-max 21
     #:y-min 0 #:y-max 21
     #:width 1400
     #:height 1400))
   
  (displayln "Tamamlandı."))

(main)
