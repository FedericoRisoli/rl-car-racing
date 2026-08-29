# Documentazione — Reinforcement Learning per CarRacing

# Reinforcement Learning per CarRacing

## Confronto sperimentale tra PPO e DQN su Gymnasium CarRacing-v2

**Repository:** [FedericoRisoli/rl-car-racing](https://github.com/FedericoRisoli/rl-car-racing)

---

# Abstract

Questo lavoro studia l'impiego di tecniche di **Reinforcement Learning** per il controllo di un veicolo nell'ambiente visuale `CarRacing-v2` di Gymnasium. L'obiettivo è confrontare, in condizioni sperimentali il più possibile omogenee, due algoritmi disponibili in Stable-Baselines3: **Proximal Policy Optimization (PPO)** e **Deep Q-Network (DQN)**.

Entrambi gli agenti ricevono immagini RGB `96×96×3` come osservazione, utilizzano una `CnnPolicy` e operano nello stesso spazio di cinque azioni discrete. Il protocollo sperimentale separa esplicitamente tuning, validation e test finale. Le run definitive utilizzano quattro training seed indipendenti (`10, 11, 12, 13`) per ciascun algoritmo, con un budget di **1.000.000 di environment step per run**. La selezione del modello viene effettuata esclusivamente sulla validation, usando i seed `200–204`, mentre il confronto finale viene eseguito su **30 piste nuove**, generate con i seed `1000–1029`.

Nel test finale PPO raggiunge un reward medio di **750.85**, contro **480.04** di DQN, con una differenza di **+270.82** pari a circa **+56.42%** rispetto al reward medio DQN. La track completion media è **85.11%** per PPO e **58.00%** per DQN, con un vantaggio di **27.11 punti percentuali**. PPO ottiene inoltre una deviazione standard inferiore tra i quattro training seed (`56.72` contro `93.51` sul reward medio per run), risulta migliore in tutti i **4/4 training seed** e, mediando sui training seed, in tutte le **30/30 piste di test**.

I risultati indicano che, **nella configurazione sperimentale studiata**, PPO è risultato più efficace e più consistente di DQN. La conclusione non viene estesa in senso universale ad altri task, configurazioni o spazi delle azioni.

**Parole chiave:** Reinforcement Learning, CarRacing, PPO, DQN, Stable-Baselines3, Gymnasium, guida autonoma, CNN.

---

# 1. Introduzione

Il local trajectory planning e il controllo del movimento costituiscono componenti centrali nei sistemi di guida autonoma. In un problema di Reinforcement Learning un agente non riceve una sequenza di azioni corretta da imitare, ma apprende interagendo con l'ambiente e massimizzando una reward cumulativa.

`CarRacing-v2` rappresenta un banco di prova utile perché combina percezione visuale e controllo sequenziale: la pista viene generata proceduralmente, l'osservazione è un'immagine e l'agente deve imparare a mantenere il veicolo sulla carreggiata e ad avanzare lungo il circuito.

Lo studio confronta due famiglie algoritmiche differenti:

- **DQN**, algoritmo value-based e off-policy;
- **PPO**, algoritmo policy-gradient actor-critic e on-policy.

Gli algoritmi non sono stati implementati da zero: vengono utilizzate le implementazioni di **Stable-Baselines3**. Il contributo progettuale riguarda la costruzione della pipeline sperimentale, la configurazione comune del problema, la gestione dei seed, il tuning controllato, il logging, checkpoint e recovery, la selezione del best model, il test finale su piste non utilizzate in precedenza e l'analisi quantitativa e qualitativa dei risultati.

La domanda sperimentale è quindi:

> **A parità di ambiente, spazio delle azioni, osservazioni e protocollo di valutazione, quale comportamento mostrano PPO e DQN su CarRacing-v2?**
> 

---

# 2. Ambiente e formulazione del problema

## 2.1 Ambiente

L'ambiente utilizzato è:

```python
ENV_ID = "CarRacing-v2"
CONTINUOUS = False
LAP_COMPLETE_PERCENT = 0.95
DOMAIN_RANDOMIZE = False
MAX_EPISODE_STEPS = 1000
```

La configurazione è centralizzata in `src/env.py` e viene riutilizzata sia durante il training sia durante l'evaluation, riducendo il rischio di differenze involontarie tra PPO e DQN.

## 2.2 Osservazione

L'osservazione è un'immagine RGB:

$$
96 \times 96 \times 3
$$

L'agente non riceve direttamente la posizione del veicolo, la traiettoria ideale, il raggio di curvatura o la distanza dal bordo. Le informazioni necessarie alla guida devono essere ricavate dai pixel.

Entrambi gli algoritmi utilizzano `CnnPolicy`. La CNN funge da estrattore di feature visuali:

```
Immagine RGB 96×96×3
        │
        ▼
       CNN
        │
        ▼
Feature visuali
    ┌───┴───┐
    ▼       ▼
   PPO     DQN
```

Stable-Baselines3 gestisce internamente la conversione dal formato immagine `Height × Width × Channels` al formato richiesto da PyTorch tramite `VecTransposeImage` quando necessario.

## 2.3 Spazio delle azioni

È stata utilizzata la modalità discreta di CarRacing:

```python
continuous=False
```

Le azioni disponibili sono:

1. nessun comando;
2. sterza a sinistra;
3. sterza a destra;
4. accelera;
5. frena.

La scelta è necessaria per utilizzare DQN con Stable-Baselines3. PPO potrebbe operare anche con azioni continue, ma utilizzare due action space differenti avrebbe introdotto una variabile sperimentale aggiuntiva. Per questo motivo entrambi gli algoritmi sono stati valutati con **lo stesso spazio delle azioni discreto**.

La discretizzazione rappresenta anche un limite del confronto: il controllo è più grossolano rispetto alla modalità continua e non consente di combinare liberamente sterzo e acceleratore nello stesso timestep.

## 2.4 Reward e terminazione

Il progetto utilizza la reward originale di CarRacing, senza reward shaping personalizzato. In forma semplificata, la reward premia l'avanzamento lungo nuove tile della pista e penalizza il tempo trascorso senza progresso. L'uscita dall'area valida può produrre una forte penalizzazione e una terminazione anticipata.

Ogni episodio è limitato a **1000 timestep**. Durante l'evaluation vengono mantenute distinte le informazioni `terminated` e `truncated`, in modo da distinguere condizioni terminali dell'ambiente da interruzioni dovute al limite temporale.

---

# 3. Algoritmi

## 3.1 Deep Q-Network

DQN è un algoritmo **value-based**. La rete neurale approssima la funzione:

$$
Q(s,a)
$$

che stima il ritorno futuro atteso associato all'azione `a` nello stato `s`. La scelta greedy è:

$$
a^* = \arg\max_a Q(s,a)
$$

DQN è **off-policy** e utilizza un replay buffer contenente transizioni del tipo:

$$
(s_t, a_t, r_t, s_{t+1})
$$

La configurazione finale principale è:

```
learning_rate              = 0.0001
buffer_size                = 10_000
learning_starts            = 500
batch_size                 = 32
tau                        = 1.0
gamma                      = 0.99
train_freq                 = 4 step
gradient_steps             = 1
target_update_interval     = 10_000
exploration_initial_eps    = 1.0
exploration_final_eps      = 0.05
exploration_fraction       = 0.5
max_grad_norm              = 10.0
optimize_memory_usage      = False
```

Il replay buffer è stato mantenuto a `10_000` per limitare il consumo di memoria: le osservazioni sono immagini RGB e un buffer molto grande avrebbe avuto un costo elevato sui computer disponibili.

### Esplorazione epsilon-greedy

Con `exploration_fraction=0.5`, epsilon diminuisce da `1.0` a `0.05` durante il primo 50% del training. Questo parametro **non significa che il 50% delle azioni sia casuale per tutta la run**.

## 3.2 Proximal Policy Optimization

PPO è un algoritmo **on-policy actor-critic**. L'actor rappresenta la policy:

$$
\pi_\theta(a|s)
$$

mentre il critic approssima il valore dello stato:

$$
V(s)
$$

PPO limita aggiornamenti troppo grandi tramite il clipping del rapporto tra la nuova policy e quella utilizzata per raccogliere i rollout.

La configurazione utilizzata mantiene sostanzialmente gli hyperparameter standard di Stable-Baselines3:

```
learning_rate       = 0.0003
n_steps             = 2048
batch_size          = 64
n_epochs            = 10
gamma               = 0.99
gae_lambda          = 0.95
clip_range          = 0.2
normalize_advantage = True
ent_coef            = 0.0
vf_coef             = 0.5
max_grad_norm       = 0.5
use_sde             = False
```

Poiché PPO raccoglie rollout di `2048` step, il numero effettivo di timestep può superare leggermente il target indicato. Questo è un effetto normale della struttura a rollout dell'algoritmo.

---

# 4. Ambiente software e hardware

Le versioni software sono state allineate tra le macchine utilizzate:

| Componente | Versione |
| --- | --- |
| Python | 3.11.9 |
| NumPy | 1.26.4 |
| Gymnasium | 0.29.0 |
| Stable-Baselines3 | 2.3.2 |
| PyTorch | 2.3.1+cpu |

Il device è stato fissato a:

```python
device="cpu"
```

I training sono stati eseguiti su due computer con hardware differente:

- AMD Ryzen 5 5600H, 8 GB RAM;
- Intel Core i5-1135G7, 8 GB RAM.

Per questo motivo il **wall-clock time non viene utilizzato come misura principale di confronto**. L'asse sperimentale comune è il numero di environment step.

---

# 5. Pipeline sperimentale

## 5.1 Separazione tra tuning, validation e test

La separazione dei seed è stata definita per evitare di riutilizzare nel test finale piste che avevano già influenzato le decisioni progettuali.

| Fase | Seed | Funzione |
| --- | --- | --- |
| Pilot / tuning | training 0–4 | scelte preliminari e tuning DQN |
| Validation usata durante il tuning | 100–104 | supporto alle scelte preliminari |
| Training finale | 10, 11, 12, 13 | 4 run indipendenti per algoritmo |
| Validation finale | 200–204 | selezione del best checkpoint |
| Test finale | 1000–1029 | 30 piste non usate per tuning o model selection |

Le piste `100–104`, essendo state utilizzate durante il tuning, non vengono considerate parte del test finale.

## 5.2 Tuning di DQN

Il tuning principale ha riguardato `exploration_fraction`, confrontando `0.1` e `0.5` su cinque training seed appaiati.

| Training seed | fraction 0.1 | fraction 0.5 | Migliore |
| --- | --- | --- | --- |
| 0 | 132.33 | 92.14 | 0.1 |
| 1 | 146.09 | 275.76 | 0.5 |
| 2 | 62.96 | 30.81 | 0.1 |
| 3 | 13.66 | 72.87 | 0.5 |
| 4 | 91.93 | 109.41 | 0.5 |

Riepilogo:

- media dei best reward: `89.39` per `0.1`, `116.20` per `0.5`;
- mediana: `91.93` per `0.1`, `92.14` per `0.5`;
- vittorie seed-per-seed: `2/5` per `0.1`, `3/5` per `0.5`.

È stato quindi fissato:

```
exploration_fraction = 0.5
```

La scelta è stata congelata prima del protocollo finale per evitare un tuning indefinito sui medesimi dati di validation.

## 5.3 Training finale

Il protocollo definitivo comprende:

```
PPO × seed 10,11,12,13
DQN × seed 10,11,12,13
```

per un totale di **8 run finali**.

Budget nominale per run:

```
1.000.000 environment step
```

Durante ciascuna run viene effettuata una validation ogni:

```
25.000 step
```

sui seed:

```
200 201 202 203 204
```

La validation viene eseguita in modo deterministico.

## 5.4 Selezione del best checkpoint

Il modello destinato al test finale non è necessariamente quello all'ultimo timestep. Per ogni run viene selezionato il checkpoint con **mean validation reward più alto** sui seed `200–204`.

Questa scelta è importante perché le learning curve non sono monotone: continuare il training non garantisce che la policy finale sia la migliore osservata durante la run.

`best_model_info.json` registra le informazioni associate al checkpoint selezionato, tra cui timestep, mean reward, deviazione standard e validation seed.

## 5.5 Checkpoint, recovery e resume

`train.py` implementa meccanismi distinti per:

- checkpoint di analisi;
- best model;
- recovery DQN;
- resume di una run interrotta.

Per DQN il recovery salva anche il replay buffer, poiché riprendere soltanto i pesi della rete perderebbe una parte fondamentale dello stato di apprendimento off-policy.

---

# 6. Evaluation finale

## 6.1 Protocollo di test

Ogni best model viene valutato su tutte le 30 piste:

```
1000, 1001, ..., 1029
```

con policy deterministica.

Il test finale comprende quindi:

$$
2 \; algoritmi \times 4 \; training\ seed \times 30 \; piste = 240 \; episodi
$$

Sono stati verificati:

- **8 run finali**;
- **240 episodi di test**;
- **30 episodi per ciascun algoritmo/training seed**;
- nessun duplicato `algorithm × training_seed × evaluation_seed`;
- `strict_mode = true` nell'aggregazione;
- **0 warning** nei controlli di integrità.

## 6.2 Metriche registrate

Per ogni episodio vengono memorizzate metriche quali:

- `episode_reward`;
- `episode_length`;
- `completed`;
- `visited_tiles`;
- `total_tiles`;
- `track_completion`;
- `termination_reason`;
- `terminated`;
- `truncated`.

Il reward rappresenta la metrica principale ottimizzata dagli algoritmi. La track completion viene affiancata per fornire una lettura più intuitiva della percentuale di pista percorsa.

Reward e track completion sono fortemente correlati nel task CarRacing e non vengono quindi interpretati come due evidenze statisticamente indipendenti.

---

# 7. Aggregazione e analisi dei dati

Gli output delle evaluation vengono aggregati tramite `src/aggregate_results.py`. Lo script esegue controlli di consistenza sul protocollo e produce:

```
aggregated_results/
├── aggregation_report.json
├── combined_test_episodes.csv
├── learning_curve_algorithm.csv
├── learning_curve_per_run.csv
├── test_algorithm_summary.csv
├── test_paired_seed_comparison.csv
├── test_per_run_summary.csv
├── test_per_track_summary.csv
└── training_run_summary.csv
```

L'unità sperimentale principale è il **training seed indipendente**. Per questo motivo ciascuna run viene prima riassunta sulle sue 30 piste e solo successivamente le quattro run vengono aggregate a livello di algoritmo.

I 120 episodi per algoritmo vengono comunque utilizzati per statistiche descrittive e confronti pista-per-pista, ma non vengono presentati come 120 training indipendenti.

L'analisi descrittiva finale è implementata in `src/statistical_analysis.py` e utilizza misure semplici e direttamente interpretabili:

- media;
- mediana;
- deviazione standard;
- minimo e massimo;
- differenza PPO-DQN;
- confronto per training seed;
- confronto per test track;
- conteggio delle vittorie;
- completion rate.

---

# 8. Risultati

## 8.1 Prestazioni complessive sul test finale

| Metrica | PPO | DQN |
| --- | --- | --- |
| Training seed | 4 | 4 |
| Episodi di test | 120 | 120 |
| Reward medio | 750.85 | 480.04 |
| Std reward tra training seed | 56.72 | 93.51 |
| Track completion media | 85.11% | 58.00% |
| Std track completion tra training seed | 5.66 pp | 9.35 pp |
| Completion rate formale | 5.83% (7/120) | 0.00% (0/120) |
| Episode length medio | 986.23 | 1000.00 |

La differenza di reward è:

$$
750.85 - 480.04 = 270.82
$$

che corrisponde a circa:

$$
\frac{270.82}{480.04} \times 100 \approx 56.42\%
$$

La differenza di track completion è:

$$
85.11\% - 58.00\% = 27.11 \; punti\ percentuali
$$

## 8.2 Risultati per training seed

| Training seed | DQN reward medio | PPO reward medio | Migliore |
| --- | --- | --- | --- |
| 10 | 465.43 | 683.24 | PPO |
| 11 | 426.25 | 772.62 | PPO |
| 12 | 616.18 | 731.59 | PPO |
| 13 | 412.29 | 815.95 | PPO |

PPO è migliore in **4 training seed su 4**.

Un risultato particolarmente evidente è che:

```
peggior PPO = 683.24
miglior DQN = 616.18
```

I range delle quattro medie per-run non si sovrappongono nel campione osservato.

La deviazione standard tra training seed è inferiore per PPO:

```
PPO = 56.72
DQN = 93.51
```

Questo indica che, nel protocollo studiato, le prestazioni dei best model PPO risultano più consistenti rispetto al cambiamento del training seed.

## 8.3 Risultati per pista

Per ciascun test seed `1000–1029` è stata calcolata la performance media sui quattro training seed.

Risultato:

```
PPO migliore: 30 / 30 piste
DQN migliore: 0 / 30 piste
```

Il vantaggio PPO non è quindi concentrato soltanto su un piccolo numero di circuiti particolarmente favorevoli.

Scendendo al livello dei singoli confronti `training seed × test track`:

```
PPO migliore: 106 / 120
DQN migliore: 14 / 120
Pareggi:       0 / 120
```

Questo conteggio è utilizzato come descrizione della diffusione del vantaggio e non come se i 120 episodi rappresentassero 120 training indipendenti.

## 8.4 Completion rate

PPO completa formalmente il giro in:

```
7 / 120 episodi = 5.83%
```

DQN in:

```
0 / 120 episodi = 0.00%
```

Il completion rate formale è basso anche per PPO, ma non deve essere interpretato isolatamente. PPO percorre mediamente **85.11%** della pista, mentre DQN si ferma mediamente a **58.00%**.

La metrica `completed` rappresenta il criterio formale dell'ambiente, mentre `track_completion` misura direttamente la porzione di circuito visitata. Le due misure sono quindi complementari ma non equivalenti.

---

# 9. Dinamica del training

Le learning curve sono costruite con:

- asse X = environment step;
- asse Y = mean validation reward;
- linea = media sui quattro training seed;
- fascia = deviazione standard tra training seed.

PPO raggiunge valori di validation più alti e mostra una crescita più rapida. DQN presenta invece maggiore dispersione e una marcata regressione nella fase finale.

Le ultime cinque evaluation medie DQN sono:

| Step | Mean validation reward | Std tra training seed |
| --- | --- | --- |
| 900.000 | 202.63 | 170.19 |
| 925.000 | 205.66 | 171.77 |
| 950.000 | 159.30 | 202.50 |
| 975.000 | 150.56 | 132.62 |
| 1.000.000 | 2.11 | 43.46 |

A `1M` tutti e quattro i modelli DQN risultano deboli nell'ultima evaluation: il valore minimo è circa `-57.20` e il massimo circa `40.15`.

Questo risultato **non viene interpretato automaticamente come catastrophic forgetting**. È sufficiente affermare che la performance di validation DQN regredisce in modo netto nella parte finale della run.

Il fenomeno conferma l'utilità della selezione del **best checkpoint**: il final test viene eseguito sul modello che aveva ottenuto il miglior valore medio di validation, non semplicemente sul modello all'ultimo timestep.

---

# 10. Visualizzazione dei risultati

`src/plot_results.py` legge esclusivamente gli output consolidati in `aggregated_results/` e produce i grafici utilizzati nell'analisi:

```
plots/
├── learning_curve.png
├── final_test_reward_by_seed.png
├── final_test_track_completion_by_seed.png
└── final_test_reward_per_track.png
```

I grafici principali mostrano:

- andamento della validation durante il training;
- reward finale dei quattro modelli PPO e DQN;
- track completion per training seed;
- confronto pista-per-pista sui 30 test seed.

Nel grafico per pista i test seed rappresentano circuiti indipendenti e non una sequenza temporale; per questo l'interpretazione corretta è un confronto appaiato pista-per-pista.

---

# 11. Analisi qualitativa tramite video

Accanto alle metriche quantitative è stato registrato un confronto visuale PPO-DQN sulla stessa pista.

Configurazione scelta:

```
training seed = 10
test track    = 1005
```

La pista `1005` è stata scelta perché abbastanza rappresentativa delle prestazioni complessive e non come caso estremo favorevole a un singolo algoritmo.

Risultati del confronto registrato:

```
DQN seed 10, track 1005 → reward 482.82
PPO seed 10, track 1005 → reward 654.60
```

I due video sono stati affiancati in un unico filmato con sovraimpressione dell'algoritmo, del training seed, del test track e del reward.

Il video ha funzione **qualitativa**: permette di osservare concretamente la differenza nel comportamento di guida, ma non sostituisce l'evidenza quantitativa ottenuta dalle 30 piste e dai quattro training seed.

---

# 12. Discussione

## 12.1 Interpretazione del confronto PPO-DQN

Nel protocollo adottato PPO supera DQN secondo tutte le principali letture aggregate: reward medio, track completion, confronto tra training seed e confronto tra piste.

Il risultato è particolarmente consistente perché PPO è superiore:

- in tutti i quattro training seed finali;
- in tutte le 30 piste quando si media sui training seed;
- in 106 dei 120 confronti diretti seed×pista.

Inoltre PPO presenta una deviazione standard inferiore tra le quattro run, suggerendo una minore sensibilità al training seed nella configurazione studiata.

Non è possibile attribuire il vantaggio a una singola causa con i dati raccolti. Alcuni elementi che possono aver inciso sono:

- la natura on-policy e actor-critic di PPO;
- la maggiore instabilità osservata nelle curve DQN;
- il replay buffer DQN limitato a `10_000` transizioni per vincoli di memoria;
- la difficoltà di apprendere un controllo visuale da immagini con un algoritmo value-based nel budget considerato;
- la discretizzazione comune delle azioni, necessaria per DQN ma non necessariamente ottimale per PPO.

Questi punti devono essere considerati **possibili fattori**, non dimostrazioni causali.

## 12.2 Reward e track completion

La forte relazione tra reward e porzione di pista percorsa spiega perché i due indicatori raccontano una storia simile. La track completion viene mantenuta perché rende intuitivo il significato del reward e permette di descrivere la qualità della guida in termini di avanzamento sul circuito.

## 12.3 Effetto della selezione del checkpoint

Il crollo della validation DQN verso `1M` mostra che un training più lungo non implica necessariamente una policy migliore all'ultimo timestep. Il protocollo basato sul best checkpoint riduce il rischio di confrontare gli algoritmi usando modelli scelti arbitrariamente in base alla durata del training.

## 12.4 Generalizzazione

Le piste del test finale `1000–1029` non sono state usate nel tuning né nella selezione dei best model. La superiorità di PPO su queste piste indica quindi una migliore performance su circuiti nuovi **all'interno della stessa distribuzione di CarRacing-v2 utilizzata nel progetto**.

Non si tratta di una prova di generalizzazione a domini visivi differenti, perché `domain_randomize=False` e l'aspetto grafico dell'ambiente rimane quello standard.

---

# 13. Limiti dello studio

I principali limiti sono:

1. **Quattro training seed finali per algoritmo.** Sono sufficienti per osservare la variabilità, ma rappresentano un campione limitato della stochasticità possibile.
2. **Spazio delle azioni discreto.** È stato scelto per rendere confrontabili PPO e DQN, ma non rappresenta il controllo più ricco disponibile in CarRacing.
3. **Replay buffer DQN limitato a 10.000.** La scelta è stata guidata dai vincoli di memoria dei computer utilizzati.
4. **PPO non sottoposto a tuning esteso.** Sono stati mantenuti prevalentemente i parametri standard Stable-Baselines3.
5. **Tuning DQN limitato.** È stato studiato in modo controllato soprattutto `exploration_fraction`.
6. **CPU e hardware differenti.** Per questo motivo il wall-clock time non è utilizzato come criterio di efficacia algoritmica.
7. **Domain randomization disattivata.** Il test misura la capacità di generalizzare a nuove piste, non a variazioni grafiche del dominio.
8. **Assenza di confronto con altri algoritmi.** Lo studio riguarda esclusivamente PPO e DQN standard.
9. **Statistiche prevalentemente descrittive.** Con quattro training seed si è scelto di evitare conclusioni inferenziali eccessivamente forti.

Questi limiti delimitano la portata delle conclusioni senza invalidare il confronto interno, che rimane controllato rispetto ad ambiente, seed di test, budget e procedura di selezione.

---

# 14. Riproducibilità e struttura del software

La repository è organizzata in moduli con responsabilità distinte:

```
src/
├── env.py
├── train.py
├── evaluate.py
├── aggregate_results.py
├── plot_results.py
├── statistical_analysis.py
└── random_agent.py
```

Funzioni principali:

- `env.py`: configurazione dell'ambiente;
- `train.py`: training, validation, checkpoint, best model, recovery e resume;
- `evaluate.py`: evaluation deterministica e registrazione video;
- `aggregate_results.py`: controlli di integrità e aggregazione;
- `plot_results.py`: grafici finali;
- `statistical_analysis.py`: statistiche descrittive;
- `random_agent.py`: test preliminare dell'ambiente.

Comando tipo per il training finale:

```powershell
python src\train.py --algo ppo --timesteps 1000000 --seed 10 --run-type final --eval-freq 25000 --eval-seeds 200 201 202 203 204
```

```powershell
python src\train.py --algo dqn --timesteps 1000000 --seed 10 --run-type final --eval-freq 25000 --eval-seeds 200 201 202 203 204
```

Aggregazione:

```powershell
python src\aggregate_results.py
```

Grafici:

```powershell
python src\plot_results.py
```

Analisi descrittiva:

```powershell
python src\statistical_analysis.py
```

La pipeline complessiva è:

```
Configurazione comune
        │
        ├───────────────┐
        ▼               ▼
       PPO             DQN
        │               │
        ▼               ▼
training multi-seed finale
        │
        ▼
validation 200–204
        │
        ▼
selezione best checkpoint
        │
        ▼
test finale 1000–1029
        │
        ▼
aggregazione controllata
        │
        ▼
grafici + statistiche descrittive
        │
        ▼
analisi quantitativa e qualitativa
```

---

# 15. Conclusioni

Il progetto ha realizzato e validato una pipeline completa per il confronto tra **PPO** e **DQN** nel task visuale `CarRacing-v2`.

Il protocollo sperimentale ha mantenuto comuni ambiente, osservazioni, spazio delle azioni, budget di training e piste di evaluation, separando in modo esplicito tuning, validation e test finale. Sono stati utilizzati quattro training seed indipendenti per algoritmo e 30 piste nuove per il test finale.

I risultati mostrano un vantaggio netto di PPO nella configurazione studiata:

```
Reward medio
PPO = 750.85
DQN = 480.04
Differenza = +270.82 (+56.42%)

Track completion media
PPO = 85.11%
DQN = 58.00%
Differenza = +27.11 punti percentuali

Variabilità reward tra training seed
PPO std = 56.72
DQN std = 93.51

Confronti
PPO migliore su 4/4 training seed
PPO migliore su 30/30 piste mediate sui training seed
PPO migliore in 106/120 confronti seed × pista
```

Nel contesto sperimentale considerato, PPO ha quindi mostrato **maggiore efficacia e maggiore consistenza** rispetto a DQN. La conclusione è supportata sia dai risultati quantitativi del test set sia dall'analisi qualitativa del comportamento di guida, ma rimane circoscritta alla specifica formulazione del problema adottata in questo studio.