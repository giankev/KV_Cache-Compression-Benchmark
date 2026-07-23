# Note di implementazione

## 1. Algoritmo

La compressione viene applicata alla `DynamicCache` di Hugging Face. Per ogni
layer non escluso e per ogni batch/head KV, si calcola in `float32` il punteggio
`sum(K ** 2)` di ciascun token. La radice quadrata non serve: non cambia
l'ordinamento della norma L2.

- `low_l2` conserva i token con punteggio minore;
- `high_l2` conserva quelli con punteggio maggiore;
- `random` conserva un sottoinsieme riproducibile tramite seed locale.

Il numero di token è `ceil(keep_ratio * sequence_length)`. La variante online
usa invece un budget massimo fisso.

## 2. Shape della cache

Ogni coppia ha shape:

```text
K, V: [batch, num_key_value_heads, sequence, head_dim]
```

K e V devono avere la stessa shape e lo stesso device. La selezione avviene
lungo la dimensione `sequence`.

## 3. Perché si conservano le key a norma L2 minore

Il lavoro di riferimento osserva empiricamente che, nei modelli studiati, key
con norma L2 più bassa sono spesso associate a maggiore attenzione. `low_l2`
usa questa correlazione come euristica economica: non calcola l'attenzione e non
modifica il modello. Non è una garanzia valida per ogni layer o modello.

## 4. Indici condivisi tra K e V

Gli score sono calcolati solo da K. Gli stessi identici indici temporali sono
poi usati per raccogliere sia K sia V. Usare indici diversi romperebbe la
corrispondenza key/value del token originale.

## 5. Ordine temporale

`topk` restituisce gli elementi in ordine di score, non di posizione. Dopo la
selezione gli indici vengono quindi riordinati in senso crescente prima del
`gather`. La cache conserva così l'ordine cronologico dei token selezionati.
Questa è una correzione intenzionale rispetto al semplice ordinamento per norma
presente nel repository di riferimento.

## 6. Lunghezza logica e fisica

La lunghezza **logica** è il numero totale di token già processati. La lunghezza
**fisica** è il numero di token ancora memorizzati in un layer della cache. Dopo
il pruning le due grandezze non coincidono; inoltre i layer 0 e 1, esclusi dalla
compressione, sono fisicamente più lunghi degli altri.

`position_ids` e `cache_position` continuano sempre dalla lunghezza logica
originale. Per batch size 1, senza padding, domanda e risposta vengono elaborate
un token alla volta e senza una attention mask artificiale basata sul layer 0.
Questo è il percorso verificato dallo smoke test; il progetto non introduce un
custom attention layer.

## 7. Bug corretti

- **Posizione dopo pruning:** prima veniva usata indirettamente la lunghezza
  fisica; ora le posizioni logiche sono esplicite.
- **Attention mask dal layer 0:** era incompatibile con layer di lunghezze
  diverse; dopo il pruning non viene costruita una mask all-ones condivisa.
- **Prompt approssimato:** il filler veniva decodificato e ritokenizzato; ora il
  contesto è composto direttamente da token ID ed è lungo esattamente quanto il
  target. Viene salvata anche la posizione effettiva dell'information line.
- **Random non riproducibile:** non si usa più stato globale; numero passkey e
  selezione cache hanno generatori locali derivati dal seed dell'esempio.
- **Memoria passkey:** la memoria è misurata subito dopo il prefill e subito
  dopo il pruning; il raw CSV conserva la percentuale risparmiata.
- **Memoria media online LM:** cache compressa e baseline teorica vengono
  confrontate allo stesso passo temporale, sia come media sia alla fine.
- **ALR mutata in-place:** lunghezze e key del prefill vengono salvate prima del
  decode; il token query aggiunto non entra nel calcolo L2/ALR.

## 8. Benchmark

Il modello principale è `Qwen/Qwen2.5-3B-Instruct`. Il benchmark riprende il
task professor-style di `l2compress/eval_passkey.py`: un solo passkey, garbage
ripetuto, posizione casuale per seed e confronto esatto dei token della
risposta. Il prompt viene costruito direttamente da token ID e ha esattamente
la lunghezza richiesta.

Il runner L2 usa `no_compression`, `low_l2`, `random` e `high_l2` con keep ratio
comune 0.8. Il runner SnapKV usa soltanto `no_compression` e `snapkv`. Entrambi
aggregano l'accuracy sui dieci seed effettivamente eseguiti e interrompono le
configurazioni compresse quando la baseline dello stesso prompt fallisce.

La compressione è eseguita una sola volta, dopo il prefill. Lo stack Kaggle è
fissato in `requirements-kaggle.txt`; `transformers==4.57.6` espone l'API
`DynamicCache.layers` usata dal codice.

## 9. Limiti

- un solo modello principale;
- dieci passkey sintetici per lunghezza di contesto;
- compressione esclusivamente post-prefill nel passkey;
- selezione indipendente per head KV, come nel metodo di riferimento;
- percorso post-pruning limitato a batch size 1 senza padding e forward da un
  token;
- risultati dimostrativi per un progetto universitario, non conclusivi né
  destinati alla pubblicazione.
