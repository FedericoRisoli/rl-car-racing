# Reinforcement Learning per CarRacing

## Documentazione tecnica e metodologia sperimentale

---

# Capitolo 1 — Introduzione

## 1.1 Obiettivo del progetto

Il progetto riguarda l'applicazione di tecniche di **Reinforcement Learning** al problema di guida proposto dall'ambiente `CarRacing` di Gymnasium.

L'obiettivo assegnato è confrontare almeno due algoritmi di Reinforcement Learning applicati allo stesso problema. Sono stati scelti **Proximal Policy Optimization (PPO)** e **Deep Q-Network (DQN)**.

Il lavoro non è stato impostato con l'obiettivo di costruire da zero l'implementazione matematica dei due algoritmi. Per questa parte vengono utilizzate le implementazioni disponibili nella libreria **Stable-Baselines3**. Il lavoro svolto riguarda invece soprattutto la costruzione dell'esperimento: configurazione dell'ambiente, scelta dello spazio delle azioni, gestione degli addestramenti, riproducibilità, salvataggio dei modelli e definizione di un metodo di valutazione comune.

Il punto centrale del progetto è quindi il confronto tra PPO e DQN nelle stesse condizioni sperimentali.

In particolare, si vuole osservare:

* se i due algoritmi riescono effettivamente ad apprendere a percorrere la pista;
* quanto rapidamente migliorano durante il training;
* quanto sono stabili tra addestramenti differenti;
* quanto riescono a generalizzare su piste generate con seed diversi;
* quali tipi di fallimento si presentano più frequentemente.

Il confronto non vuole stabilire quale algoritmo sia migliore in senso assoluto. Le conclusioni saranno necessariamente riferite alla configurazione di CarRacing utilizzata nel progetto.

---

## 1.2 Organizzazione generale

Il progetto è stato mantenuto volutamente abbastanza semplice dal punto di vista della struttura del codice.

I file principali sono:

```text
src/
├── env.py
├── train.py
├── evaluate.py
└── random_agent.py
```

Le responsabilità sono separate nel seguente modo:

* `env.py` contiene la configurazione dell'ambiente;
* `train.py` gestisce il training di PPO e DQN;
* `evaluate.py` si occupa della valutazione dei modelli;
* `random_agent.py` è stato utilizzato inizialmente per verificare che CarRacing funzionasse correttamente.

La configurazione dell'ambiente è centralizzata in `env.py` per evitare di avere parametri diversi tra training ed evaluation.

Con il crescere del progetto sono state aggiunte anche directory dedicate a modelli, risultati e checkpoint.

Una struttura indicativa è:

```text
rl-car-racing/
│
├── src/
│   ├── env.py
│   ├── train.py
│   ├── evaluate.py
│   └── random_agent.py
│
├── models/
├── results/
├── logs/
├── checkpoints/
│
├── requirements-base.txt
├── requirements-lock.txt
└── README.md
```

---

# Capitolo 2 — Ambiente CarRacing

## 2.1 Descrizione dell'ambiente

L'ambiente scelto è:

```python
CarRacing-v2
```

CarRacing appartiene agli ambienti Box2D di Gymnasium.

A ogni episodio viene generato un circuito e viene posizionata una macchina sulla linea di partenza. L'agente deve percorrere la pista cercando di visitare il maggior numero possibile di segmenti senza uscire dall'area valida.

A differenza di problemi di Reinforcement Learning più semplici, lo stato non viene fornito sotto forma di poche variabili numeriche.

L'osservazione consiste principalmente in un'immagine RGB:

[
96 \times 96 \times 3
]

L'agente vede quindi direttamente la scena di gioco.

Non gli vengono passate esplicitamente informazioni come:

* distanza dal bordo;
* raggio della curva;
* traiettoria ottimale;
* coordinate della macchina;
* direzione da seguire.

Queste informazioni devono essere dedotte implicitamente dall'immagine.

Questo rende CarRacing interessante perché combina un problema di controllo con un problema di percezione visuale.

---

## 2.2 Configurazione utilizzata

La configurazione comune è definita in `env.py`.

I parametri principali sono:

```python
ENV_ID = "CarRacing-v2"

CONTINUOUS = False

DOMAIN_RANDOMIZE = False

LAP_COMPLETE_PERCENT = 0.95

MAX_EPISODE_STEPS = 1000
```

Tutti i programmi che devono utilizzare CarRacing creano l'ambiente passando attraverso la stessa funzione `make_env()`.

Il vantaggio è soprattutto pratico: eventuali modifiche alla configurazione vengono effettuate in un solo punto e training ed evaluation continuano a utilizzare la stessa definizione del problema.

---

# Capitolo 3 — Spazio delle azioni

## 3.1 Azioni continue e discrete

CarRacing può essere eseguito in due modalità.

Nella modalità continua l'azione è composta da tre valori:

[
[\text{steering},\text{gas},\text{brake}]
]

Il controllo permette quindi, ad esempio, di accelerare e sterzare contemporaneamente con intensità differenti.

Nel progetto è stata invece utilizzata la modalità discreta:

```python
continuous=False
```

Le azioni disponibili diventano cinque:

| Valore | Azione            |
| -----: | ----------------- |
|      0 | nessun comando    |
|      1 | sterza a sinistra |
|      2 | sterza a destra   |
|      3 | accelera          |
|      4 | frena             |

---

## 3.2 Motivo della discretizzazione

Questa scelta nasce direttamente dagli algoritmi che si vogliono confrontare.

PPO può lavorare sia con spazi di azione continui sia con spazi discreti.

DQN, nell'implementazione utilizzata da Stable-Baselines3, richiede invece uno spazio delle azioni discreto.

Una possibilità sarebbe stata lasciare PPO nella modalità continua e utilizzare DQN con la modalità discreta.

In questo modo, però, avremmo confrontato contemporaneamente due cose:

1. due algoritmi differenti;
2. due rappresentazioni differenti delle azioni.

Se PPO avesse ottenuto risultati migliori, sarebbe stato difficile capire quanto del vantaggio dipendesse dall'algoritmo e quanto dal fatto di avere un controllo più preciso della macchina.

Per questo è stato scelto lo stesso spazio discreto per entrambi.

Il confronto diventa quindi più controllato:

[
\text{stesso ambiente}
+
\text{stesse osservazioni}
+
\text{stesse azioni}
]

mentre cambia principalmente l'algoritmo utilizzato per imparare.

La discretizzazione introduce comunque una limitazione.

Con questa configurazione non è possibile eseguire nello stesso timestep, ad esempio:

```text
accelerare + sterzare a sinistra
```

Il controllo è quindi meno preciso rispetto alla versione continua originale di CarRacing.

Questo è un compromesso accettato per rendere possibile il confronto PPO-DQN nelle stesse condizioni.

---

# Capitolo 4 — Osservazioni e rete neurale

## 4.1 Perché utilizzare una CNN

Dal momento che l'input dell'agente è un'immagine, per entrambi gli algoritmi viene utilizzata:

```python
CnnPolicy
```

La parte iniziale della rete è quindi una **Convolutional Neural Network**.

Una rete convoluzionale è adatta a questo problema perché mantiene conto delle relazioni spaziali presenti nell'immagine.

Durante il training non vengono specificate manualmente caratteristiche come i bordi della strada o la direzione della curva. La rete deve imparare autonomamente feature utili a partire dai pixel.

In maniera semplificata il processo può essere rappresentato così:

```text
Immagine 96×96×3
        │
        ▼
Convolutional Neural Network
        │
        ▼
Rappresentazione delle feature
        │
        ▼
PPO oppure DQN
```

PPO e DQN condividono quindi lo stesso tipo di ingresso, anche se utilizzano le feature in maniera diversa.

---

## 4.2 Preprocessing delle immagini

Gymnasium restituisce normalmente le immagini nella forma:

```text
Height × Width × Channels
```

nel nostro caso:

```text
96 × 96 × 3
```

Le reti convoluzionali implementate attraverso PyTorch lavorano invece generalmente nella forma:

```text
Channels × Height × Width
```

quindi:

```text
3 × 96 × 96
```

Nelle parti del progetto che utilizzano un `VecEnv` viene quindi utilizzato `VecTransposeImage`, che esegue questa conversione.

Le immagini hanno valori dei pixel nell'intervallo 0–255. La policy CNN di Stable-Baselines3 gestisce internamente anche la loro normalizzazione.

---

# Capitolo 5 — Reward e terminazione

## 5.1 Reward di CarRacing

Nel progetto non è stata modificata la funzione di reward originale dell'ambiente.

CarRacing assegna una piccola penalità per ogni frame:

[
-0.1
]

e assegna un reward positivo quando viene visitata una nuova tile della pista.

Il contributo positivo dipende dal numero totale di tile:

[
\frac{1000}{N}
]

dove (N) è il numero totale di tile del circuito.

Il reward incentiva quindi l'agente a:

1. percorrere nuova pista;
2. farlo senza perdere troppo tempo.

Un agente che rimane fermo accumula progressivamente reward negativo.

Un agente che avanza lungo la pista riceve invece reward positivi visitando nuove tile.

Il reward contiene quindi già una componente legata sia all'avanzamento sia alla velocità con cui viene ottenuto.

---

## 5.2 Limite massimo dell'episodio

Nel progetto ogni episodio può durare al massimo:

```python
MAX_EPISODE_STEPS = 1000
```

Il limite serve soprattutto a impedire che agenti poco addestrati rimangano indefinitamente nell'ambiente senza completare il circuito.

Gymnasium distingue due condizioni:

```text
terminated
truncated
```

`terminated` indica che l'ambiente ha raggiunto una propria condizione terminale.

`truncated` indica invece che l'episodio è stato interrotto da un limite esterno, nel nostro caso principalmente il `TimeLimit`.

Questa distinzione viene mantenuta anche durante l'evaluation.

---

## 5.3 Completamento del giro

Il parametro:

```python
LAP_COMPLETE_PERCENT = 0.95
```

indica che almeno il 95% della pista deve essere stato visitato affinché il ritorno nella zona iniziale possa essere considerato come completamento del giro.

Non è richiesto quindi che il 100% delle tile venga necessariamente attraversato.

La soglia permette di tollerare piccoli segmenti non visitati senza considerare fallito un giro che è stato sostanzialmente completato.

---

# Capitolo 6 — Domain randomization

CarRacing supporta anche la variazione casuale di alcune caratteristiche visive dell'ambiente.

Nel progetto questa funzionalità è disabilitata:

```python
DOMAIN_RANDOMIZE = False
```

La scelta è stata fatta principalmente per mantenere il problema iniziale più controllato.

La rete deve già imparare contemporaneamente a interpretare la pista e a guidare. Aggiungere variazioni importanti dell'aspetto grafico avrebbe aumentato ulteriormente la difficoltà e reso meno immediata l'interpretazione dei risultati.

La domanda principale del progetto rimane il confronto tra PPO e DQN, non lo studio della robustezza a differenti domini visivi.

Questo significa però che i risultati finali dovranno essere letti tenendo presente questo limite.

Un eventuale sviluppo futuro potrebbe ripetere gli esperimenti con:

```python
domain_randomize=True
```

per osservare quanto le policy siano robuste rispetto a variazioni grafiche.

---

# Capitolo 7 — Algoritmi utilizzati

## 7.1 Stable-Baselines3

PPO e DQN vengono utilizzati attraverso **Stable-Baselines3**.

Il progetto non contiene quindi un'implementazione manuale dei due algoritmi.

La libreria si occupa delle operazioni interne come:

* gestione delle reti neurali;
* calcolo delle loss;
* backpropagation;
* aggiornamento degli optimizer;
* rollout;
* replay buffer;
* target network.

Il nostro codice definisce invece come gli algoritmi vengono inseriti all'interno dell'esperimento.

In particolare vengono gestiti:

* ambiente;
* policy;
* seed;
* numero di timestep;
* salvataggio dei modelli;
* checkpoint;
* logging;
* evaluation;
* organizzazione dei risultati.

L'utilizzo di una libreria permette inoltre di lavorare su implementazioni consolidate e documentate, evitando che eventuali errori nella riscrittura di PPO o DQN influenzino il confronto.

---

# Capitolo 8 — Deep Q-Network

## 8.1 Principio di funzionamento

DQN appartiene alla famiglia degli algoritmi **value-based**.

L'obiettivo è approssimare la funzione:

[
Q(s,a)
]

che rappresenta il ritorno futuro atteso associato all'esecuzione dell'azione (a) nello stato (s).

Nel nostro caso lo stato è un'immagine e lo spazio delle azioni contiene cinque elementi.

Per ogni osservazione, la rete produce quindi cinque Q-value:

[
Q(s,a_0),Q(s,a_1),Q(s,a_2),Q(s,a_3),Q(s,a_4)
]

L'azione considerata migliore è quella con valore massimo:

[
a^*=\arg\max_a Q(s,a)
]

---

## 8.2 Experience replay

DQN conserva le transizioni generate durante l'interazione con l'ambiente.

Una transizione può essere rappresentata come:

[
(s_t,a_t,r_t,s_{t+1})
]

Questi dati vengono inseriti in un **replay buffer**.

Durante il training vengono campionati minibatch di esperienze dal buffer invece di utilizzare esclusivamente le transizioni appena generate.

Il replay permette di riutilizzare le esperienze e riduce la correlazione temporale tra i campioni utilizzati per aggiornare la rete.

Nel progetto è stato impostato:

```python
buffer_size=10_000
```

La dimensione è inferiore al valore standard utilizzabile da DQN.

La ragione principale è il costo in memoria delle osservazioni visuali. Ogni transizione contiene immagini 96×96 RGB e un replay buffer molto grande avrebbe quindi richiesto una quantità considerevole di RAM.

Il valore 10.000 rappresenta per ora un compromesso tra disponibilità di esperienze e memoria utilizzata.

Non viene considerato un valore ottimizzato definitivamente per CarRacing.

---

## 8.3 Inizio del training

È stato inoltre impostato:

```python
learning_starts=500
```

Prima di iniziare gli aggiornamenti vengono quindi raccolte almeno 500 transizioni.

L'obiettivo è evitare che la rete inizi ad apprendere utilizzando un buffer quasi vuoto, contenente solo pochissime esperienze iniziali.

---

## 8.4 Target network

DQN utilizza una seconda rete chiamata **target network**.

Se la stessa rete venisse utilizzata contemporaneamente per calcolare il valore corrente e il valore target, il bersaglio dell'apprendimento cambierebbe continuamente insieme alla rete stessa.

La target network viene aggiornata più lentamente e rende quindi più stabile il processo di apprendimento.

---

## 8.5 Esplorazione epsilon-greedy

DQN deve trovare un equilibrio tra:

```text
exploration
exploitation
```

Se scegliesse fin dall'inizio sempre l'azione con il Q-value maggiore, il comportamento dipenderebbe da stime ancora casuali e molte alternative potrebbero non essere mai esplorate.

Durante il training viene quindi utilizzata una strategia epsilon-greedy.

Con probabilità (\epsilon) viene eseguita un'azione casuale; altrimenti viene utilizzata l'azione con Q-value maggiore.

Con il procedere del training l'esplorazione viene progressivamente ridotta.

---

# Capitolo 9 — Proximal Policy Optimization

## 9.1 Principio di funzionamento

PPO appartiene invece alla famiglia degli algoritmi **policy-gradient**.

L'algoritmo cerca direttamente una policy:

[
\pi_\theta(a|s)
]

che descrive la probabilità di scegliere un'azione dato lo stato corrente.

A differenza di DQN, quindi, non viene costruita principalmente una funzione che assegna un valore a ogni azione.

L'obiettivo è modificare direttamente la policy in modo da aumentare la probabilità delle azioni che hanno prodotto risultati positivi.

---

## 9.2 Struttura actor-critic

PPO utilizza una struttura **actor-critic**.

L'actor rappresenta la policy:

[
\pi(a|s)
]

Il critic cerca invece di stimare:

[
V(s)
]

cioè il valore dello stato.

La struttura può essere visualizzata come:

```text
                 ┌── Actor ──→ probabilità delle azioni
                 │
Immagine → CNN ──┤
                 │
                 └── Critic ─→ valore dello stato
```

Il critic viene utilizzato durante il training per stimare quanto un'azione sia stata migliore o peggiore rispetto a ciò che ci si aspettava dallo stato corrente.

---

## 9.3 Algoritmo on-policy

PPO è **on-policy**.

Questo significa che le esperienze utilizzate per gli aggiornamenti vengono raccolte con la policy corrente.

Dopo un certo numero di interazioni viene costruito un rollout, vengono calcolate le quantità necessarie all'aggiornamento e la policy viene modificata.

In seguito vengono raccolti nuovi dati con la nuova policy.

Non viene quindi mantenuto un grande archivio storico delle transizioni equivalente al replay buffer di DQN.

---

## 9.4 Clipping

Un problema dei metodi policy-gradient è che aggiornamenti troppo grandi possono modificare drasticamente una policy che stava già funzionando bene.

PPO affronta il problema confrontando la probabilità assegnata a un'azione dalla nuova policy con quella assegnata dalla vecchia policy.

Il rapporto è:

[
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
]

La funzione obiettivo applica un clipping a questo rapporto.

Con un `clip_range` pari a 0.2, aggiornamenti che porterebbero il rapporto troppo lontano da 1 vengono limitati nella funzione obiettivo.

L'idea non è impedire alla policy di cambiare, ma evitare modifiche eccessive in un singolo aggiornamento.

---

# Capitolo 10 — Configurazione del training

## 10.1 Creazione dei modelli

La configurazione PPO è attualmente simile a:

```python
model = PPO(
    "CnnPolicy",
    env,
    seed=args.seed,
    device="cpu",
    verbose=1,
)
```

Per DQN:

```python
model = DQN(
    "CnnPolicy",
    env,
    seed=args.seed,
    device="cpu",
    verbose=1,
    buffer_size=10_000,
    learning_starts=500,
)
```

Per PPO si è deciso inizialmente di lasciare invariata la maggior parte degli hyperparameter standard della libreria.

Questa configurazione viene utilizzata come baseline.

Prima di effettuare tuning esteso è utile capire infatti come si comportano gli algoritmi con una configurazione semplice e riproducibile.

---

## 10.2 Utilizzo della CPU

Entrambi gli algoritmi vengono eseguiti con:

```python
device="cpu"
```

La decisione è legata principalmente all'ambiente di sviluppo.

Il progetto viene eseguito da due persone su computer differenti e non è disponibile necessariamente lo stesso tipo di GPU.

Forzare l'esecuzione su CPU rende quindi più semplice mantenere una configurazione comune.

Lo svantaggio evidente è un training più lento, soprattutto perché l'elaborazione delle immagini attraverso CNN è computazionalmente costosa.

Per il progetto è stato comunque considerato più importante mantenere un setup semplice e replicabile tra le due macchine.

---

# Capitolo 11 — Timestep, episodi e run

## 11.1 Significato di timestep

Il numero passato a:

```python
model.learn(total_timesteps=...)
```

indica il numero di interazioni con l'ambiente.

Un timestep può essere schematizzato come:

```text
osservazione
     ↓
scelta dell'azione
     ↓
env.step(action)
     ↓
reward
     ↓
nuova osservazione
```

Un training da:

```text
1.000.000 timestep
```

non corrisponde quindi a un milione di episodi.

Ogni episodio contiene un numero variabile di timestep, fino al massimo configurato di 1000.

---

## 11.2 Smoke, pilot e final

Le run sono state divise in tre categorie:

```text
smoke
pilot
final
```

La distinzione serve a mantenere separati test tecnici ed esperimenti utilizzati per l'analisi.

### Smoke

Gli smoke test hanno un numero di timestep molto basso.

Servono per controllare che:

* il training parta;
* il modello venga creato correttamente;
* il file venga salvato;
* il modello possa essere ricaricato;
* l'evaluation funzioni;
* i risultati vengano scritti su file.

Non vengono utilizzati per stabilire quale algoritmo sia migliore.

Con poche migliaia di timestep un agente visuale su CarRacing non ha avuto tempo sufficiente per sviluppare una policy significativa.

### Pilot

Le run `pilot` vengono utilizzate dopo aver verificato il funzionamento della pipeline.

Sono più lunghe e permettono di capire se il modello sta effettivamente iniziando ad apprendere.

Servono anche a stimare:

* tempi di training;
* quantità di memoria richiesta;
* frequenza dei checkpoint;
* frequenza delle evaluation;
* durata realistica degli esperimenti finali.

### Final

Le run `final` sono quelle utilizzate per il confronto conclusivo.

Una volta iniziati questi training, la configurazione dovrebbe essere mantenuta stabile per tutti gli esperimenti confrontati.

---

# Capitolo 12 — Naming e organizzazione degli esperimenti

Ogni training viene identificato con un nome che riporta le informazioni principali.

Per esempio:

```text
ppo_final_1M_seed_0
```

corrisponde a:

```text
algoritmo       PPO
tipo run        final
timestep        1.000.000
training seed   0
```

Analogamente:

```text
dqn_pilot_100k_seed_2
```

identifica una pilot run DQN da 100.000 timestep con seed 2.

Una convenzione di questo tipo diventa importante quando iniziano a essere presenti diversi modelli, checkpoint, CSV e video.

Il nome della run dovrebbe idealmente propagarsi anche agli output di evaluation, in modo che ogni risultato sia riconducibile direttamente al modello da cui è stato prodotto.

---

# Capitolo 13 — Seed e riproducibilità

## 13.1 Perché utilizzare più seed

Gli esperimenti di Reinforcement Learning sono fortemente influenzati dalla casualità.

Tra gli elementi che possono cambiare tra due run troviamo:

* inizializzazione dei pesi;
* esplorazione;
* ordine delle esperienze;
* campionamento dei minibatch;
* generazione delle piste.

Per questo motivo un singolo training non è sufficiente per confrontare due algoritmi.

Supponiamo di ottenere:

```text
PPO seed 0 → reward 700
DQN seed 0 → reward 300
```

Questo risultato da solo non consente di concludere che PPO sia superiore.

È possibile che il primo training sia stato particolarmente favorevole e il secondo particolarmente sfavorevole.

Il confronto finale dovrà quindi includere più training indipendenti.

Per esempio:

```text
PPO
├── seed 0
├── seed 1
├── seed 2
└── seed 3

DQN
├── seed 0
├── seed 1
├── seed 2
└── seed 3
```

In questo modo diventa possibile osservare non soltanto la media delle prestazioni, ma anche la loro variabilità.

---

## 13.2 Training seed ed evaluation seed

È utile distinguere due concetti.

Il **training seed** identifica la casualità associata all'addestramento.

L'**evaluation seed** viene invece utilizzato quando viene generata una pista per testare il modello.

Per esempio:

```text
PPO training seed 0
```

può essere valutato su:

```text
eval seed 100
eval seed 101
eval seed 102
eval seed 103
eval seed 104
```

Questo permette di testare lo stesso agente su più circuiti differenti.

---

# Capitolo 14 — Metodo di valutazione

## 14.1 Ambiente separato

L'evaluation viene effettuata su un ambiente separato rispetto a quello utilizzato per il training.

L'ambiente ha la stessa configurazione, ma viene creato separatamente.

Questo evita di utilizzare direttamente le statistiche prodotte durante l'addestramento come misura finale delle prestazioni.

Training ed evaluation hanno infatti obiettivi differenti.

Durante il training interessa esplorare e aggiornare la policy.

Durante l'evaluation interessa misurare il comportamento di una policy già appresa.

---

## 14.2 Stesse piste per PPO e DQN

Uno dei punti principali del protocollo è che entrambi gli algoritmi vengono valutati utilizzando gli stessi evaluation seed.

Per esempio:

```text
seed 100 → PPO e DQN
seed 101 → PPO e DQN
seed 102 → PPO e DQN
```

Se utilizzassimo piste casuali differenti, una differenza di reward potrebbe essere dovuta semplicemente alla diversa difficoltà del circuito.

Utilizzare gli stessi seed consente invece di effettuare un confronto diretto sulla stessa pista.

Questo permette anche analisi successive pista per pista.

---

## 14.3 Evaluation deterministica

Durante l'evaluation viene utilizzato:

```python
deterministic=True
```

Durante il training la casualità è utile perché permette all'agente di esplorare.

Durante la valutazione interessa invece osservare la strategia che il modello ha appreso.

La modalità deterministica riduce quindi la variabilità dovuta alla scelta casuale delle azioni.

---

# Capitolo 15 — Evaluation durante il training

In `train.py` è presente una callback dedicata alla valutazione periodica:

```python
FixedSeedEvalCallback
```

La callback viene eseguita ogni certo numero di timestep.

Per esempio:

```text
0
│
├── 10k → evaluation
├── 20k → evaluation
├── 30k → evaluation
├── 40k → evaluation
│
└── ...
```

Le piste utilizzate rimangono le stesse tra un'evaluation e la successiva.

In questo modo è possibile osservare l'evoluzione della policy senza che il cambiamento della pista renda difficile interpretare la curva.

L'evaluation periodica dovrebbe consentire di studiare aspetti come:

* velocità iniziale di apprendimento;
* plateau;
* instabilità;
* eventuali regressioni;
* differenze di sample efficiency tra PPO e DQN.

---

# Capitolo 16 — Checkpoint

I checkpoint permettono di salvare lo stato del modello a intervalli regolari durante un training lungo.

Hanno due utilità differenti.

La prima è pratica: un training molto lungo può essere interrotto per diversi motivi e salvare periodicamente il lavoro riduce il rischio di perdere l'intera run.

La seconda è sperimentale.

Salvando modelli, ad esempio, a:

```text
100k
200k
300k
500k
1M
```

è possibile valutare l'agente in diversi momenti del training e ricostruire più facilmente l'evoluzione dell'apprendimento.

Per PPO il salvataggio del modello contiene le informazioni principali necessarie per mantenere la policy.

DQN richiede maggiore attenzione perché utilizza anche un replay buffer.

Se l'obiettivo è soltanto valutare un checkpoint, il modello salvato è sufficiente.

Se invece si vuole interrompere e successivamente riprendere fedelmente il training DQN, è utile mantenere anche il replay buffer.

La gestione definitiva di questa parte verrà documentata insieme all'implementazione finale dei checkpoint.

---

# Capitolo 17 — Metriche

## 17.1 Reward episodico

Per ogni episodio viene registrata la somma dei reward:

[
R=\sum_{t=0}^{T}r_t
]

È la metrica direttamente ottimizzata dagli algoritmi e rimane quindi fondamentale.

Non è però l'unica metrica considerata.

---

## 17.2 Track completion

Viene registrato il numero di tile visitate rispetto al numero totale di tile della pista:

[
\text{track completion}
=======================

\frac{\text{visited tiles}}
{\text{total tiles}}
]

Questo valore indica in maniera immediata quanta parte del circuito è stata percorsa.

Un agente potrebbe, ad esempio, non completare alcun giro ma riuscire regolarmente a percorrere l'80–90% della pista.

Questa informazione andrebbe persa guardando soltanto il completion rate.

---

## 17.3 Completion rate

Per un insieme di episodi viene calcolato:

[
\text{completion rate}
======================

\frac{\text{episodi completati}}
{\text{episodi totali}}
]

Questa metrica rappresenta direttamente la capacità dell'agente di portare a termine il compito.

---

## 17.4 Motivo della terminazione

La fine di un episodio viene classificata in categorie come:

```text
lap_completed
out_of_bounds
time_limit
unknown
```

Questo dato permette di capire meglio perché un modello fallisce.

Due agenti con lo stesso completion rate possono avere problemi molto differenti.

Un agente potrebbe uscire frequentemente dalla pista perché guida in modo aggressivo.

Un altro potrebbe rimanere sulla pista ma procedere troppo lentamente fino al raggiungimento del limite dei 1000 timestep.

Guardando solo il reward o il completion rate questa differenza sarebbe meno evidente.

---

## 17.5 Lunghezza dell'episodio

Viene registrato anche il numero di timestep.

La metrica non viene interpretata isolatamente.

Un episodio breve potrebbe indicare:

* completamento molto rapido;
* uscita immediata dalla pista.

Un episodio lungo potrebbe invece rappresentare:

* buona percorrenza;
* agente bloccato;
* guida estremamente lenta.

Per questo motivo la durata viene analizzata insieme alle altre metriche.

---

# Capitolo 18 — Statistiche aggregate

I risultati dei singoli episodi vengono successivamente aggregati.

Tra le statistiche calcolate troviamo:

```text
mean reward
standard deviation
median reward
minimum reward
maximum reward
completion rate
mean episode length
mean track completion
```

La media permette di descrivere il comportamento generale del modello.

La deviazione standard indica quanto i risultati cambino tra un episodio e l'altro.

Questo è importante perché in Reinforcement Learning due modelli con lo stesso reward medio possono avere stabilità molto differenti.

Per esempio:

```text
A = 490, 500, 510, 495

B = 100, 900, 150, 850
```

Le medie possono essere relativamente simili, ma il comportamento del primo modello è molto più regolare.

La mediana viene mantenuta perché è meno influenzata da eventuali episodi estremamente positivi o negativi.

---

# Capitolo 19 — Salvataggio dei risultati

L'evaluation produce due livelli di informazione.

Il primo contiene i dati episodio per episodio.

Esempio:

```text
evaluation_seed
episode_reward
episode_length
completed
termination_reason
visited_tiles
total_tiles
track_completion
```

Il secondo contiene il summary statistico della run.

La scelta di mantenere anche i dati grezzi permette di effettuare analisi successive senza dover eseguire nuovamente il modello.

Dai CSV degli episodi sarà possibile, per esempio:

* confrontare PPO e DQN pista per pista;
* costruire boxplot;
* calcolare intervalli di confidenza;
* identificare piste particolarmente difficili;
* analizzare separatamente i diversi tipi di fallimento.

---

# Capitolo 20 — Valutazione qualitativa

Oltre ai risultati numerici, `evaluate.py` permette di visualizzare gli episodi o registrarli in video.

La valutazione visiva non viene utilizzata come sostituto delle metriche.

Può però essere utile per interpretarle.

Per esempio, osservando direttamente l'agente è possibile distinguere tra:

* oscillazioni continue;
* guida molto prudente;
* forte accelerazione seguita da uscite di pista;
* difficoltà nelle curve strette;
* incapacità di recuperare dopo una traiettoria sbagliata.

Per la presentazione finale sarà utile registrare PPO e DQN sulla stessa pista.

I video permetteranno di collegare i risultati numerici al comportamento reale della macchina.

---

# Capitolo 21 — Validation e test

L'evaluation periodica introduce una distinzione importante tra **validation** e **test finale**.

Se durante il training utilizziamo continuamente alcuni seed, ad esempio:

```text
100
101
102
103
104
```

e utilizziamo questi risultati per decidere quale checkpoint conservare, quelle piste stanno influenzando indirettamente la scelta del modello.

Non dovrebbero quindi essere utilizzate anche come unico test finale.

Una struttura più corretta sarebbe:

```text
TRAINING
   │
   ▼
VALIDATION
seed 100–104
   │
   ▼
scelta del checkpoint
   │
   ▼
TEST FINALE
seed 200–219
```

I seed del test finale non dovrebbero essere utilizzati durante lo sviluppo o per scegliere il modello migliore.

Questa separazione rende più credibile la misura della capacità di generalizzazione.

---

# Capitolo 22 — Protocollo sperimentale finale

Una volta conclusa la fase di sviluppo, il confronto dovrebbe essere effettuato mantenendo invariata la configurazione.

Per entrambi gli algoritmi dovranno essere utilizzati:

| Variabile            | PPO            | DQN            |
| -------------------- | -------------- | -------------- |
| Ambiente             | uguale         | uguale         |
| Osservazione         | RGB 96×96      | RGB 96×96      |
| Action space         | discreto       | discreto       |
| Numero azioni        | 5              | 5              |
| TimeLimit            | 1000           | 1000           |
| Domain randomization | False          | False          |
| Evaluation seed      | uguali         | uguali         |
| Evaluation           | deterministica | deterministica |
| Budget di training   | confrontabile  | confrontabile  |

Il confronto dovrebbe essere ripetuto utilizzando più seed di training.

Uno schema possibile è:

```text
                     CarRacing
                         │
               configurazione comune
                         │
            ┌────────────┴────────────┐
            │                         │
           PPO                       DQN
            │                         │
      seed differenti          seed differenti
            │                         │
            └────────────┬────────────┘
                         │
                 validation comune
                         │
                scelta checkpoint
                         │
                   test comune
                         │
                 analisi risultati
```

---

# Capitolo 23 — Criteri di confronto

Il confronto conclusivo non verrà effettuato utilizzando una singola misura.

Saranno considerate almeno tre dimensioni.

## Prestazione finale

Attraverso:

* reward medio;
* reward mediano;
* completion rate;
* track completion media.

## Stabilità

Attraverso:

* deviazione standard degli episodi;
* variabilità tra training seed.

## Sample efficiency

Attraverso le evaluation intermedie.

In questo caso la domanda non sarà soltanto:

> quale algoritmo ottiene il risultato finale migliore?

ma anche:

> quanto training è stato necessario per raggiungerlo?

Un algoritmo potrebbe infatti ottenere prestazioni finali simili a un altro ma richiedere un numero molto maggiore di interazioni con l'ambiente.

---

# Capitolo 24 — Stato attuale del progetto

Al momento è funzionante l'intera pipeline di base:

* ambiente CarRacing;
* spazio delle azioni discreto;
* agente casuale;
* PPO;
* DQN;
* policy CNN;
* training parametrico;
* seed;
* salvataggio dei modelli;
* evaluation periodica;
* evaluation finale;
* metriche per episodio;
* summary statistico;
* registrazione video.

Sono stati inoltre completati diversi smoke test, che hanno verificato il corretto funzionamento della pipeline.

I risultati degli smoke test non vengono utilizzati per confrontare gli algoritmi, perché la durata dell'addestramento è troppo ridotta.

Le attività ancora in corso o previste sono:

1. completamento della gestione dei checkpoint;
2. test del resume del training;
3. definizione definitiva dei seed di validation e test;
4. pilot run;
5. scelta del budget dei training finali;
6. training multi-seed;
7. valutazione finale;
8. produzione dei grafici;
9. confronto PPO-DQN.

---

# Capitolo 25 — Limiti del lavoro

Il progetto contiene alcune scelte che limitano la generalità dei risultati.

La prima è la discretizzazione dello spazio delle azioni.

CarRacing nasce anche come problema di controllo continuo e ridurre il controllo a cinque azioni semplifica alcuni aspetti, ma rende anche meno preciso il comportamento del veicolo.

La scelta è stata necessaria per poter utilizzare DQN e mantenere lo stesso action space per PPO.

Un'altra limitazione riguarda la domain randomization, che è stata disabilitata.

I modelli vengono quindi addestrati e valutati in un dominio visuale relativamente stabile.

Non viene studiata direttamente la robustezza a variazioni dell'aspetto dell'ambiente.

Esiste inoltre il problema generale del costo computazionale.

Utilizzare osservazioni visuali significa che entrambe le reti devono elaborare immagini attraverso una CNN e training sufficientemente lunghi richiedono un numero elevato di interazioni.

Il progetto viene inoltre eseguito su CPU per mantenere una configurazione comune tra le macchine utilizzate nello sviluppo.

Le conclusioni dovranno quindi essere formulate in maniera coerente con questi vincoli.

Non sarà possibile affermare:

> PPO è migliore di DQN.

Una conclusione corretta sarà invece del tipo:

> Nella configurazione di CarRacing utilizzata nel progetto e con il protocollo sperimentale adottato, PPO ha ottenuto risultati mediamente migliori di DQN secondo le metriche considerate.

Oppure, naturalmente, il contrario se sarà quello che mostreranno i dati.

---

# Capitolo 26 — Sviluppi possibili

Una volta completato il confronto principale, il progetto potrebbe essere esteso in diverse direzioni.

Una prima possibilità sarebbe confrontare PPO in ambiente discreto e continuo, per misurare quanto la discretizzazione limiti effettivamente le prestazioni.

Un'altra estensione potrebbe riguardare la domain randomization, valutando la robustezza delle policy a modifiche dell'aspetto visivo.

Si potrebbe inoltre studiare l'effetto degli hyperparameter, in particolare:

* dimensione del replay buffer;
* learning rate;
* exploration schedule di DQN;
* rollout size di PPO;
* batch size;
* numero di epoche PPO.

Un'ulteriore possibilità sarebbe introdurre algoritmi come:

* Double DQN;
* Dueling DQN;
* SAC;
* TD3.

Queste estensioni non sono però necessarie per rispondere alla domanda principale del progetto, che rimane il confronto tra PPO e DQN.

---

# Capitolo 27 — Conclusione metodologica

Il lavoro svolto fino a questo punto ha portato alla costruzione di una pipeline comune per l'addestramento e la valutazione di PPO e DQN sull'ambiente CarRacing.

La parte più importante per il confronto non è semplicemente riuscire ad addestrare i due modelli, ma assicurarsi che il confronto venga effettuato nelle stesse condizioni.

Per questo motivo sono stati mantenuti comuni:

* ambiente;
* osservazioni;
* action space;
* criterio di terminazione;
* evaluation seed;
* metriche;
* metodo di valutazione.

L'utilizzo di più training seed servirà invece a distinguere l'effetto dell'algoritmo dalla normale variabilità presente nel Reinforcement Learning.

I checkpoint e le evaluation periodiche permetteranno infine di analizzare non soltanto il risultato finale, ma anche l'evoluzione dell'apprendimento durante il training.

La fase successiva del progetto consiste quindi nel completare l'infrastruttura di checkpoint e passare progressivamente dagli smoke test agli esperimenti pilota e infine ai training utilizzati per il confronto finale.
