# Reinforcement Learning per CarRacing

## Documentazione tecnica e metodologia sperimentale

---

# 1. Introduzione

## 1.1 Obiettivo del progetto

Il progetto riguarda l'applicazione di tecniche di **Reinforcement Learning** al problema di guida proposto dall'ambiente `CarRacing` di Gymnasium.

L'obiettivo è confrontare due algoritmi di Reinforcement Learning applicati allo stesso problema:

* **Proximal Policy Optimization (PPO)**
* **Deep Q-Network (DQN)**

Il progetto non prevede l'implementazione manuale dei due algoritmi a partire dalle rispettive formulazioni matematiche. Per questa parte vengono utilizzate le implementazioni messe a disposizione dalla libreria **Stable-Baselines3**.

Il lavoro si concentra invece sulla costruzione dell'intero esperimento:

* configurazione dell'ambiente;
* scelta dello spazio delle azioni;
* configurazione degli algoritmi;
* gestione dei seed;
* organizzazione delle run;
* salvataggio dei modelli;
* checkpoint;
* valutazione periodica;
* valutazione finale;
* raccolta delle metriche;
* confronto tra i risultati.

Il punto centrale del progetto non è quindi soltanto riuscire ad addestrare due agenti, ma fare in modo che PPO e DQN vengano confrontati nelle condizioni più simili possibile.

In particolare, si vuole osservare:

* se entrambi gli algoritmi riescono ad apprendere a percorrere la pista;
* quanto rapidamente migliorano durante il training;
* quanto sono stabili tra addestramenti differenti;
* come si comportano su piste diverse da quelle incontrate durante una singola run;
* quali tipi di fallimento si presentano più frequentemente;
* quale algoritmo risulta più efficace all'interno della configurazione scelta.

L'obiettivo non è stabilire quale algoritmo sia migliore in senso assoluto. Le conclusioni saranno necessariamente riferite alla specifica configurazione di CarRacing utilizzata nel progetto.

---

## 1.2 Struttura generale del progetto

Il codice è stato organizzato cercando di separare le diverse responsabilità.

I file principali presenti nella directory `src` sono:

```text
src/
├── env.py
├── train.py
├── evaluate.py
└── random_agent.py
```

Le responsabilità sono distribuite nel seguente modo:

* `env.py` contiene la configurazione comune dell'ambiente;
* `train.py` gestisce l'addestramento di PPO e DQN;
* `evaluate.py` gestisce la valutazione dei modelli;
* `random_agent.py` è stato utilizzato inizialmente per verificare il corretto funzionamento dell'ambiente.

La configurazione dell'ambiente è mantenuta in un unico file per evitare che training ed evaluation utilizzino accidentalmente parametri differenti.

Una struttura generale della repository può essere rappresentata in questo modo:

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
├── docs/
│   └── documentazione_progetto.md
│
├── requirements-base.txt
├── requirements-lock.txt
└── README.md
```

Alcune directory possono essere generate durante l'esecuzione e non necessariamente devono essere versionate su Git.

---

# 2. Ambiente CarRacing

## 2.1 Descrizione dell'ambiente

L'ambiente utilizzato è:

```python
CarRacing-v2
```

CarRacing appartiene agli ambienti basati su Box2D messi a disposizione da Gymnasium.

A ogni episodio viene generato un circuito e un veicolo viene posizionato sulla pista. L'agente deve imparare a controllare la macchina cercando di percorrere il circuito senza uscire dall'area valida.

Una caratteristica importante dell'ambiente è il tipo di osservazione fornita all'agente.

Lo stato non è rappresentato da poche variabili numeriche come:

* posizione del veicolo;
* distanza dal bordo;
* angolo della curva;
* velocità ottimale;
* traiettoria ideale.

L'agente riceve invece un'immagine RGB della scena.

La dimensione dell'osservazione è:

$$
96 \times 96 \times 3
$$

Quindi, a ogni timestep, il modello riceve un'immagine di 96×96 pixel con tre canali RGB.

Questo rende il problema più complesso rispetto a un ambiente con stato numerico, perché l'agente deve imparare contemporaneamente a:

1. interpretare visivamente l'ambiente;
2. capire la forma della pista;
3. riconoscere la propria posizione;
4. scegliere l'azione di controllo corretta.

CarRacing combina quindi un problema di percezione visuale con un problema di controllo.

---

## 2.2 Configurazione utilizzata

La configurazione comune dell'ambiente viene definita in `env.py`.

I parametri principali sono:

```python
ENV_ID = "CarRacing-v2"

CONTINUOUS = False

DOMAIN_RANDOMIZE = False

LAP_COMPLETE_PERCENT = 0.95

MAX_EPISODE_STEPS = 1000
```

L'ambiente viene creato attraverso una funzione comune, in modo che training, evaluation e test preliminari utilizzino la stessa configurazione.

Questa scelta permette anche di modificare facilmente il comportamento dell'ambiente senza dover intervenire in più file.

---

# 3. Spazio delle azioni

## 3.1 Modalità continua e discreta

CarRacing può essere utilizzato con uno spazio delle azioni continuo oppure discreto.

Nella modalità continua, il controllo del veicolo può essere espresso attraverso tre valori:

$$
[\text{steering}, \text{gas}, \text{brake}]
$$

In questo caso è possibile controllare simultaneamente sterzo, accelerazione e frenata.

Nel progetto è stata scelta invece la modalità discreta:

```python
continuous=False
```

Lo spazio delle azioni contiene cinque possibilità:

| Valore | Azione            |
| -----: | ----------------- |
|      0 | Nessun comando    |
|      1 | Sterza a sinistra |
|      2 | Sterza a destra   |
|      3 | Accelera          |
|      4 | Frena             |

---

## 3.2 Motivo della discretizzazione

La scelta deriva principalmente dagli algoritmi che si vogliono confrontare.

PPO può essere applicato sia a spazi di azione continui sia a spazi discreti.

DQN, nella versione disponibile in Stable-Baselines3, lavora invece con spazi di azione discreti.

Una possibile alternativa sarebbe stata:

```text
PPO → spazio continuo
DQN → spazio discreto
```

Questo confronto sarebbe però meno controllato.

Se PPO avesse ottenuto risultati migliori, sarebbe stato difficile capire se il vantaggio dipendesse realmente dall'algoritmo oppure dal fatto di poter utilizzare un controllo più preciso del veicolo.

Per ridurre questa differenza è stato quindi deciso di utilizzare lo stesso spazio discreto per entrambi gli algoritmi.

Il confronto viene così effettuato mantenendo comuni:

```text
stesso ambiente
+
stesse osservazioni
+
stesso spazio delle azioni
```

e modificando principalmente l'algoritmo utilizzato per l'apprendimento.

---

## 3.3 Limiti della discretizzazione

La scelta dello spazio discreto introduce comunque una limitazione importante.

Con questa configurazione viene selezionata una sola azione per timestep.

Non è quindi possibile eseguire direttamente un comando come:

```text
sterza a sinistra + accelera
```

nello stesso istante.

La versione continua di CarRacing permette un controllo molto più preciso.

La discretizzazione rappresenta quindi un compromesso: riduce la qualità potenziale del controllo, ma permette di confrontare PPO e DQN all'interno dello stesso spazio delle azioni.

---

# 4. Osservazioni visuali e rete neurale

## 4.1 Utilizzo di una CNN

Dal momento che l'osservazione dell'ambiente è un'immagine, per entrambi gli algoritmi viene utilizzata una policy basata su rete convoluzionale:

```python
CnnPolicy
```

Una rete neurale completamente connessa potrebbe teoricamente ricevere tutti i pixel dell'immagine come un unico vettore.

Questo approccio sarebbe però poco adatto a sfruttare la struttura spaziale dell'immagine.

Le **Convolutional Neural Network**, invece, sono progettate proprio per elaborare dati visuali.

Durante il training la rete può imparare autonomamente a riconoscere caratteristiche come:

* bordi della strada;
* presenza del prato;
* direzione della pista;
* curve;
* orientamento del veicolo;
* posizione relativa della macchina.

Il processo può essere rappresentato in maniera semplificata come:

```text
Immagine RGB 96×96×3
        │
        ▼
Convolutional Neural Network
        │
        ▼
Feature visuali
        │
        ▼
PPO oppure DQN
```

La parte convoluzionale trasforma quindi l'immagine in una rappresentazione numerica più compatta, che viene successivamente utilizzata dall'algoritmo di Reinforcement Learning.

---

## 4.2 Formato delle immagini

Gymnasium utilizza normalmente immagini nella forma:

```text
Height × Width × Channels
```

Nel nostro caso:

```text
96 × 96 × 3
```

PyTorch utilizza invece generalmente il formato:

```text
Channels × Height × Width
```

quindi:

```text
3 × 96 × 96
```

Quando necessario viene utilizzato `VecTransposeImage`, che si occupa di convertire il formato dell'immagine.

In questo modo la rete convoluzionale riceve i dati nella struttura corretta.

---

# 5. Reward e terminazione degli episodi

## 5.1 Reward

Nel progetto non è stata definita una reward function personalizzata.

Viene utilizzata quella originale dell'ambiente CarRacing.

In maniera semplificata, l'ambiente assegna:

* una piccola penalità per ogni frame;
* reward positivo quando vengono visitate nuove tile della pista.

La penalità temporale è:

$$
-0.1
$$

per ogni frame.

Quando viene visitata una nuova tile, il reward positivo dipende dal numero totale di tile della pista:

$$
\frac{1000}{N}
$$

dove $N$ rappresenta il numero totale di tile.

Questo significa che l'agente viene incentivato a:

1. percorrere nuove parti della pista;
2. farlo nel minor numero possibile di timestep.

Un agente che rimane fermo accumula progressivamente reward negativo.

Un agente che avanza lungo il circuito riceve invece reward positivi visitando nuove tile.

---

## 5.2 Durata massima dell'episodio

Nel progetto è stato impostato:

```python
MAX_EPISODE_STEPS = 1000
```

Un episodio può quindi durare al massimo 1000 timestep.

Questo limite è particolarmente utile nelle prime fasi del training, quando gli agenti possono rimanere bloccati o continuare a muoversi senza riuscire a completare la pista.

Senza un limite massimo, alcuni episodi potrebbero durare molto a lungo senza produrre informazioni particolarmente utili.

---

## 5.3 `terminated` e `truncated`

Gymnasium distingue due modi differenti di terminare un episodio.

Il primo è:

```python
terminated
```

e indica che l'ambiente ha raggiunto una propria condizione terminale.

Il secondo è:

```python
truncated
```

e indica invece che l'episodio è stato interrotto da una condizione esterna, come il superamento del numero massimo di timestep.

Questa distinzione è importante durante la valutazione.

Per esempio, un episodio può terminare perché:

* il veicolo ha completato correttamente il giro;
* il veicolo è uscito dall'area consentita;
* è stato raggiunto il limite di 1000 timestep.

Questi casi rappresentano comportamenti molto differenti e vengono quindi analizzati separatamente.

---

# 6. Completamento della pista

## 6.1 `lap_complete_percent`

Il progetto utilizza:

```python
LAP_COMPLETE_PERCENT = 0.95
```

Il valore indica che almeno il 95% della pista deve essere stato visitato affinché il ritorno alla zona iniziale possa essere considerato come completamento del giro.

La soglia evita di richiedere che ogni singola tile venga attraversata perfettamente.

Un giro sostanzialmente completato può quindi essere riconosciuto anche se una piccola parte della pista non è stata marcata come visitata.

---

# 7. Domain randomization

CarRacing permette di introdurre variazioni casuali nell'aspetto dell'ambiente.

Nel progetto questa funzionalità è disabilitata:

```python
DOMAIN_RANDOMIZE = False
```

La decisione è legata principalmente alla necessità di mantenere l'esperimento iniziale relativamente controllato.

L'agente deve già imparare a:

* interpretare immagini;
* riconoscere la pista;
* controllare il veicolo;
* scegliere le azioni corrette.

Aggiungere anche variazioni importanti dell'aspetto grafico renderebbe il problema ancora più complesso.

Poiché l'obiettivo principale è il confronto tra PPO e DQN, si è preferito inizialmente mantenere stabile il dominio visuale.

Questo rappresenta però un limite del progetto.

Un agente addestrato in questa configurazione non viene esplicitamente allenato a essere robusto rispetto a variazioni cromatiche o grafiche.

Una possibile estensione futura potrebbe quindi consistere nel ripetere parte degli esperimenti con:

```python
domain_randomize=True
```

---

# 8. Stable-Baselines3

## 8.1 Utilizzo della libreria

Gli algoritmi PPO e DQN vengono utilizzati attraverso **Stable-Baselines3**.

Il progetto non implementa quindi manualmente tutte le operazioni matematiche necessarie all'apprendimento.

La libreria gestisce internamente aspetti come:

* reti neurali;
* funzioni di loss;
* backpropagation;
* optimizer;
* rollout;
* replay buffer;
* target network;
* aggiornamento dei parametri.

Il codice del progetto si occupa invece di costruire l'esperimento attorno agli algoritmi.

In particolare vengono definite:

* configurazione dell'ambiente;
* policy;
* seed;
* numero di timestep;
* gestione delle run;
* salvataggio;
* checkpoint;
* evaluation;
* logging;
* raccolta dei risultati.

L'utilizzo di una libreria consolidata permette inoltre di ridurre il rischio che errori nell'implementazione degli algoritmi influenzino il confronto.

---

# 9. Deep Q-Network

## 9.1 Principio di funzionamento

DQN appartiene alla famiglia degli algoritmi **value-based**.

L'obiettivo è apprendere una funzione:

$$
Q(s,a)
$$

che stima il ritorno futuro atteso quando, trovandosi nello stato $s$, viene scelta l'azione $a$.

In un ambiente molto semplice questa funzione potrebbe essere rappresentata attraverso una tabella.

Nel caso di CarRacing questo non è possibile, perché lo stato è costituito da un'immagine.

DQN utilizza quindi una rete neurale per approssimare la funzione Q.

Dato uno stato, la rete restituisce un valore per ogni possibile azione:

$$
Q(s,a_0), Q(s,a_1), Q(s,a_2), Q(s,a_3), Q(s,a_4)
$$

Quando l'agente sfrutta ciò che ha appreso, sceglie l'azione associata al Q-value maggiore:

$$
a^* = \arg\max_a Q(s,a)
$$

---

## 9.2 Replay buffer

DQN utilizza un **experience replay buffer**.

Durante l'interazione con l'ambiente vengono prodotte transizioni del tipo:

$$
(s_t,a_t,r_t,s_{t+1})
$$

Queste transizioni vengono memorizzate e successivamente utilizzate per addestrare la rete.

Il replay buffer permette di:

* riutilizzare esperienze passate;
* ridurre la correlazione tra campioni consecutivi;
* addestrare la rete utilizzando minibatch di esperienze diverse.

Nel progetto è stato scelto:

```python
buffer_size = 10_000
```

La dimensione è relativamente ridotta rispetto a configurazioni DQN utilizzate in altri problemi.

La motivazione principale è il costo in memoria delle osservazioni.

Ogni stato è un'immagine RGB di dimensione 96×96, quindi conservare una quantità molto elevata di transizioni richiederebbe molta memoria RAM.

Il valore 10.000 rappresenta quindi un compromesso pratico tra quantità di esperienza memorizzata e risorse disponibili.

Non viene considerato necessariamente il valore ottimale per CarRacing.

---

## 9.3 `learning_starts`

È stato inoltre impostato:

```python
learning_starts = 500
```

DQN non inizia quindi immediatamente ad aggiornare la rete.

Prima vengono raccolte almeno 500 transizioni.

Questo permette al replay buffer di contenere già una quantità minima di esperienze prima dell'inizio degli aggiornamenti.

Se l'apprendimento iniziasse dopo pochissimi step, i primi minibatch sarebbero costituiti da un insieme molto limitato e altamente correlato di esperienze.

---

## 9.4 Target network

DQN utilizza anche una seconda rete chiamata **target network**.

Il problema nasce dal fatto che il valore che la rete deve apprendere dipende a sua volta da stime prodotte dalla stessa funzione Q.

Se la stessa rete venisse modificata continuamente sia per produrre le stime sia per inseguirle come target, l'apprendimento potrebbe diventare instabile.

Per questo viene utilizzata una copia della rete principale, aggiornata più lentamente.

La target network fornisce quindi un riferimento più stabile durante l'apprendimento.

---

## 9.5 Esplorazione epsilon-greedy

Durante il training DQN non sceglie sempre l'azione con Q-value massimo.

Deve infatti esplorare anche azioni che, sulla base delle conoscenze attuali, sembrano peggiori.

La strategia utilizzata è di tipo **epsilon-greedy**.

Con probabilità $\epsilon$ viene scelta un'azione casuale.

Negli altri casi viene scelta:

$$
\arg\max_a Q(s,a)
$$

All'inizio del training la componente di esplorazione è più importante.

Con il proseguire dell'apprendimento, il valore di $\epsilon$ viene ridotto e l'agente tende a utilizzare maggiormente le azioni considerate migliori.

---

# 10. Proximal Policy Optimization

## 10.1 Principio di funzionamento

PPO appartiene alla famiglia degli algoritmi **policy-gradient**.

Invece di imparare principalmente il valore delle singole azioni, PPO cerca direttamente di apprendere una policy:

$$
\pi_\theta(a|s)
$$

La policy descrive la probabilità di scegliere l'azione $a$ quando l'agente si trova nello stato $s$.

L'obiettivo dell'apprendimento è quindi modificare i parametri $\theta$ in modo da aumentare la probabilità delle azioni che producono risultati migliori.

---

## 10.2 Struttura actor-critic

PPO utilizza una struttura **actor-critic**.

L'actor rappresenta la policy:

$$
\pi(a|s)
$$

e decide quale azione eseguire.

Il critic cerca invece di stimare il valore dello stato:

$$
V(s)
$$

La struttura può essere schematizzata come:

```text
                 ┌── Actor ──→ distribuzione sulle azioni
                 │
Immagine → CNN ──┤
                 │
                 └── Critic ─→ valore dello stato
```

Il critic aiuta a stimare quanto le azioni osservate durante il training siano state migliori o peggiori rispetto alle aspettative.

---

## 10.3 Natura on-policy

PPO è un algoritmo **on-policy**.

Le esperienze utilizzate per aggiornare la policy vengono raccolte utilizzando la policy corrente.

Il processo può essere rappresentato come:

```text
Policy corrente
      │
      ▼
Interazione con ambiente
      │
      ▼
Raccolta rollout
      │
      ▼
Calcolo delle quantità necessarie
      │
      ▼
Aggiornamento PPO
      │
      ▼
Nuova policy
```

A differenza di DQN, non viene mantenuto un grande replay buffer contenente esperienze molto vecchie.

Le esperienze vengono raccolte e utilizzate durante una fase di aggiornamento, dopodiché vengono raccolti nuovi dati con la policy aggiornata.

---

## 10.4 Clipping

Una delle caratteristiche principali di PPO è il meccanismo di clipping.

Negli algoritmi policy-gradient un aggiornamento troppo grande può modificare drasticamente la policy.

Questo può portare a perdere rapidamente un comportamento che stava funzionando bene.

PPO confronta la probabilità assegnata a una determinata azione dalla nuova policy con quella assegnata dalla vecchia policy.

Il rapporto è:

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

La funzione obiettivo limita il vantaggio di aggiornamenti troppo grandi.

Con un valore tipico:

```text
clip_range = 0.2
```

gli aggiornamenti vengono mantenuti in una zona relativamente vicina alla policy precedente.

L'obiettivo non è impedire alla policy di cambiare, ma evitare modifiche eccessive in un singolo aggiornamento.

---

# 11. Configurazione degli algoritmi

## 11.1 PPO

La configurazione di PPO nel progetto è simile a:

```python
model = PPO(
    "CnnPolicy",
    env,
    seed=args.seed,
    device="cpu",
    verbose=1,
)
```

In questa fase si è scelto di utilizzare principalmente i valori standard degli hyperparameter di Stable-Baselines3.

La configurazione serve quindi come baseline iniziale.

Questo permette di verificare prima il comportamento generale dell'algoritmo, evitando di introdurre immediatamente un tuning molto complesso.

---

## 11.2 DQN

La configurazione DQN è simile a:

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

Rispetto a PPO sono stati modificati alcuni parametri per rendere il training compatibile con le risorse disponibili, in particolare la dimensione del replay buffer.

---

# 12. Utilizzo della CPU

Entrambi gli algoritmi vengono eseguiti con:

```python
device="cpu"
```

La scelta è stata fatta principalmente per mantenere un ambiente di esecuzione simile sulle diverse macchine utilizzate per lo sviluppo.

I due sviluppatori possono avere hardware grafico differente e utilizzare la GPU avrebbe richiesto configurazioni specifiche a seconda del computer.

L'utilizzo della CPU semplifica la riproducibilità del setup.

Lo svantaggio è una velocità di training inferiore, soprattutto perché entrambi gli algoritmi utilizzano reti convoluzionali per elaborare le immagini.

Per il progetto è stato comunque considerato preferibile mantenere una configurazione semplice e uniforme.

---

# 13. Timestep ed episodi

## 13.1 Significato di timestep

Il numero passato a:

```python
model.learn(total_timesteps=...)
```

indica il numero di interazioni tra agente e ambiente.

Un timestep corrisponde in maniera semplificata a:

```text
osservazione
     │
     ▼
scelta dell'azione
     │
     ▼
env.step(action)
     │
     ▼
reward
     │
     ▼
nuova osservazione
```

Un training da:

```text
1.000.000 timestep
```

non corrisponde quindi a un milione di episodi.

Un singolo episodio contiene infatti molti timestep.

Nel nostro caso può contenerne al massimo 1000.

---

# 14. Tipologie di run

Per distinguere le diverse fasi dello sviluppo sono state introdotte tre categorie:

```text
smoke
pilot
final
```

---

## 14.1 Smoke test

Gli **smoke test** utilizzano un numero molto basso di timestep.

Non hanno lo scopo di misurare realmente le prestazioni dell'algoritmo.

Servono a verificare che tutta la pipeline funzioni correttamente.

In particolare permettono di controllare:

* creazione dell'ambiente;
* inizializzazione del modello;
* esecuzione del training;
* salvataggio del modello;
* caricamento del modello;
* evaluation;
* generazione dei CSV;
* eventuale registrazione video.

I risultati ottenuti durante gli smoke test non vengono quindi utilizzati per concludere che PPO sia migliore di DQN o viceversa.

Con poche migliaia di timestep l'agente non ha avuto tempo sufficiente per apprendere un comportamento significativo.

---

## 14.2 Pilot run

Le **pilot run** costituiscono una fase intermedia.

Vengono utilizzate quando la pipeline è già stata verificata attraverso gli smoke test.

Gli obiettivi principali sono:

* verificare che il modello inizi effettivamente ad apprendere;
* osservare l'andamento iniziale delle curve;
* stimare il tempo necessario per training più lunghi;
* verificare l'utilizzo della memoria;
* testare i checkpoint;
* definire la frequenza delle evaluation;
* individuare eventuali problemi prima delle run finali.

Le pilot run sono quindi esperimenti di sviluppo, non necessariamente parte del confronto conclusivo.

---

## 14.3 Final run

Le **final run** saranno utilizzate per il confronto finale tra PPO e DQN.

Una volta definita la configurazione definitiva, i parametri dovrebbero essere mantenuti invariati per tutte le run considerate nel confronto.

Questo evita di modificare continuamente il setup sulla base dei risultati osservati e rende più pulito il protocollo sperimentale.

---

# 15. Naming delle run

Ogni training viene identificato attraverso un nome che contiene le principali informazioni della run.

Per esempio:

```text
ppo_final_1M_seed_0
```

indica:

```text
algoritmo       PPO
tipo            final
timestep        1.000.000
training seed   0
```

Un altro esempio:

```text
dqn_pilot_100k_seed_2
```

indica una run DQN di tipo pilot, addestrata per 100.000 timestep con seed 2.

Questo sistema di naming diventa importante quando aumentano:

* modelli;
* checkpoint;
* risultati;
* log;
* video.

Ogni file dovrebbe poter essere ricondotto facilmente alla run che lo ha generato.

---

# 16. Seed e riproducibilità

## 16.1 Variabilità nel Reinforcement Learning

Gli esperimenti di Reinforcement Learning presentano numerose fonti di casualità.

Tra queste troviamo:

* inizializzazione dei pesi della rete;
* azioni esplorative;
* generazione della pista;
* campionamento dei minibatch;
* ordine delle esperienze;
* comportamento iniziale dell'agente.

Per questo motivo due training dello stesso algoritmo, con gli stessi parametri, possono produrre risultati differenti.

Un singolo training non è sufficiente per confrontare correttamente due algoritmi.

Per esempio:

```text
PPO seed 0 → reward elevato
DQN seed 0 → reward basso
```

non permette da solo di concludere che PPO sia migliore.

Potrebbe trattarsi semplicemente di una run particolarmente favorevole per PPO e sfavorevole per DQN.

---

## 16.2 Più training seed

Il confronto finale dovrebbe quindi includere diversi training indipendenti.

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

L'uso di più seed permette di analizzare:

* prestazione media;
* variabilità tra run;
* stabilità dell'algoritmo.

---

# 17. Training seed ed evaluation seed

Nel progetto vengono distinti due tipi di seed.

Il **training seed** identifica la casualità associata all'addestramento.

L'**evaluation seed** viene invece utilizzato per generare la pista durante la valutazione.

Un modello:

```text
PPO training seed 0
```

può quindi essere valutato su più piste:

```text
evaluation seed 100
evaluation seed 101
evaluation seed 102
evaluation seed 103
evaluation seed 104
```

Questo permette di evitare di giudicare un modello sulla base di una singola pista.

---

# 18. Metodo di valutazione

## 18.1 Ambiente separato

L'ambiente utilizzato durante l'evaluation viene creato separatamente rispetto all'ambiente di training.

La configurazione di base rimane la stessa, ma i due ambienti hanno funzioni differenti.

Durante il training interessa:

* raccogliere esperienza;
* esplorare;
* aggiornare il modello.

Durante l'evaluation interessa invece:

* misurare il comportamento appreso;
* ridurre le fonti di casualità;
* raccogliere metriche confrontabili.

Separare i due processi rende quindi più chiara l'interpretazione dei risultati.

---

## 18.2 Stesse piste per PPO e DQN

Una scelta importante del protocollo è quella di valutare entrambi gli algoritmi sugli stessi evaluation seed.

Per esempio:

```text
seed 100 → PPO e DQN
seed 101 → PPO e DQN
seed 102 → PPO e DQN
seed 103 → PPO e DQN
```

Questo significa che i modelli percorrono le stesse piste.

Se PPO e DQN venissero valutati su circuiti casuali differenti, una differenza di reward potrebbe dipendere semplicemente dalla difficoltà delle piste.

Utilizzare gli stessi seed riduce questo problema e permette anche un confronto diretto pista per pista.

---

# 19. Evaluation deterministica

Durante l'evaluation viene utilizzato:

```python
deterministic=True
```

Durante il training la casualità è utile perché permette di esplorare l'ambiente.

Durante la valutazione si vuole invece osservare il comportamento appreso dal modello.

La modalità deterministica riduce quindi una fonte di variabilità.

In questo modo, dato lo stesso stato, il modello tende a produrre la stessa azione.

---

# 20. Evaluation periodica durante il training

Durante l'addestramento viene eseguita periodicamente una valutazione del modello.

Nel progetto questa logica viene gestita attraverso una callback personalizzata:

```python
FixedSeedEvalCallback
```

La callback permette di valutare il modello dopo un certo numero di timestep.

Un esempio di andamento è:

```text
training
│
├── 10k  → evaluation
├── 20k  → evaluation
├── 30k  → evaluation
├── 40k  → evaluation
├── 50k  → evaluation
│
└── ...
```

Le piste utilizzate rimangono le stesse tra le varie valutazioni.

Questo rende più semplice interpretare l'evoluzione delle prestazioni.

Se le piste cambiassero completamente a ogni evaluation, sarebbe più difficile distinguere un reale miglioramento da una semplice differenza nella difficoltà del circuito.

---

## 20.1 Utilità dell'evaluation periodica

Le evaluation durante il training permettono di osservare:

* velocità di apprendimento;
* plateau;
* regressioni;
* instabilità;
* differenze tra PPO e DQN;
* sample efficiency.

Per esempio, due algoritmi potrebbero raggiungere lo stesso risultato finale ma con velocità molto differenti.

Uno potrebbe raggiungere buone prestazioni dopo 200.000 timestep, mentre l'altro potrebbe richiederne 800.000.

Questa informazione andrebbe persa osservando soltanto il modello finale.

---

# 21. Wrapper dell'ambiente di evaluation

L'ambiente di evaluation viene inserito all'interno di alcuni wrapper.

Una struttura semplificata è:

```text
CarRacing
    │
    ▼
Monitor
    │
    ▼
DummyVecEnv
    │
    ▼
VecTransposeImage
```

---

## 21.1 Monitor

`Monitor` viene utilizzato per raccogliere statistiche sugli episodi.

Permette di registrare in maniera standard informazioni come reward e lunghezza episodica.

---

## 21.2 DummyVecEnv

Stable-Baselines3 utilizza internamente un'interfaccia basata su ambienti vettorializzati.

`DummyVecEnv` permette di utilizzare un singolo ambiente attraverso questa interfaccia.

Nel nostro caso non significa quindi che vengano eseguiti diversi ambienti in parallelo.

L'ambiente rimane uno solo.

---

## 21.3 VecTransposeImage

`VecTransposeImage` converte il formato delle immagini da:

```text
Height × Width × Channels
```

a:

```text
Channels × Height × Width
```

per renderlo compatibile con il formato utilizzato dalla rete convoluzionale.

---

# 22. Checkpoint

I checkpoint permettono di salvare periodicamente il modello durante un training lungo.

Questa funzionalità ha due obiettivi principali.

Il primo è pratico.

Se un training molto lungo viene interrotto, non si vuole necessariamente ricominciare da zero.

Il secondo è sperimentale.

Salvare il modello a diversi timestep permette di osservare l'evoluzione dell'apprendimento.

Per esempio:

```text
100k
200k
300k
400k
500k
...
```

I checkpoint possono essere successivamente valutati per costruire curve che mostrano come cambiano le prestazioni durante il training.

---

## 22.1 Checkpoint e DQN

Nel caso di DQN bisogna considerare anche il replay buffer.

Se un checkpoint viene utilizzato soltanto per valutare il modello, è sufficiente poter ricaricare la rete.

Se invece l'obiettivo è riprendere realmente il training dal punto in cui era stato interrotto, è importante preservare anche le informazioni necessarie alla continuazione dell'apprendimento.

DQN utilizza infatti il replay buffer come parte fondamentale del proprio stato di training.

La gestione definitiva del resume verrà documentata una volta conclusa l'implementazione del sistema di checkpoint.

---

# 23. Metriche di valutazione

Il reward è una misura fondamentale, ma non descrive completamente il comportamento dell'agente.

Per questo motivo `evaluate.py` raccoglie diverse metriche.

Tra le principali:

* episode reward;
* episode length;
* visited tiles;
* total tiles;
* track completion;
* completamento del giro;
* termination reason;
* `terminated`;
* `truncated`.

---

# 24. Episode reward

Il reward totale di un episodio è:

$$
R = \sum_{t=0}^{T} r_t
$$

Questa è la metrica direttamente associata all'obiettivo definito dall'ambiente.

Un reward maggiore indica generalmente un comportamento migliore.

Non è però sempre sufficiente per capire cosa sia successo durante la guida.

Per questo viene interpretato insieme alle altre metriche.

---

# 25. Track completion

La percentuale di pista percorsa viene calcolata come:

$$
\text{track completion} =
\frac{\text{visited tiles}}
{\text{total tiles}}
$$

Per esempio:

```text
visited_tiles = 240
total_tiles = 300
```

produce:

$$
\frac{240}{300}=0.8
$$

quindi una track completion dell'80%.

Questa metrica è utile perché permette di misurare i progressi anche quando il modello non completa interamente il giro.

Un agente che arriva regolarmente al 90% della pista è infatti molto diverso da uno che riesce a percorrerne soltanto il 10%.

---

# 26. Completion rate

Per un insieme di episodi viene calcolato:

$$
\text{completion rate} =
\frac{\text{episodi completati}}
{\text{episodi totali}}
$$

Per esempio, se un modello completa 16 piste su 20:

$$
\text{completion rate} = \frac{16}{20} = 0.8
$$

quindi:

```text
80%
```

Il completion rate rappresenta direttamente la capacità del modello di completare il compito.

---

# 27. Determinazione del completamento

Non è sufficiente utilizzare:

```python
terminated == True
```

come sinonimo di giro completato.

Un episodio può infatti terminare anche per ragioni negative.

Per questo durante l'evaluation vengono utilizzate anche informazioni relative allo stato della pista.

Tra queste possono rientrare:

* numero di tile visitate;
* numero totale di tile;
* informazioni relative al completamento del giro.

Questo permette di distinguere correttamente una terminazione dovuta al successo da una terminazione dovuta al fallimento.

---

# 28. Termination reason

Ogni episodio viene classificato attraverso una causa di terminazione.

Le categorie utilizzate sono:

```text
lap_completed
out_of_bounds
time_limit
unknown
```

Questa informazione permette di capire meglio il comportamento del modello.

Per esempio, due agenti potrebbero avere entrambi un completion rate molto basso.

Il primo potrebbe terminare frequentemente con:

```text
out_of_bounds
```

Questo potrebbe indicare che l'agente guida in maniera aggressiva ma perde spesso il controllo.

Il secondo potrebbe invece terminare principalmente con:

```text
time_limit
```

Questo potrebbe indicare una guida molto lenta oppure un agente che rimane bloccato.

Le due situazioni sono molto differenti, anche se il completion rate finale è simile.

---

# 29. Episode length

Per ogni episodio viene registrato anche il numero di timestep.

Questa metrica non viene interpretata isolatamente.

Un episodio breve potrebbe rappresentare:

* un giro completato molto velocemente;
* un'uscita immediata dalla pista.

Un episodio molto lungo potrebbe invece rappresentare:

* un agente che percorre quasi tutta la pista;
* una guida molto lenta;
* un agente bloccato.

Per questo la durata viene sempre interpretata insieme a:

* reward;
* track completion;
* completion;
* termination reason.

---

# 30. Statistiche aggregate

I risultati dei singoli episodi vengono successivamente sintetizzati attraverso diverse statistiche.

Tra quelle considerate:

* reward medio;
* deviazione standard del reward;
* reward mediano;
* reward minimo;
* reward massimo;
* completion rate;
* durata media degli episodi;
* deviazione standard della durata;
* track completion media.

---

## 30.1 Media

La media permette di descrivere il comportamento medio del modello.

Per il reward:

$$
\bar{R} =
\frac{1}{N}
\sum_{i=1}^{N}R_i
$$

---

## 30.2 Deviazione standard

La deviazione standard permette invece di misurare la variabilità dei risultati.

Due modelli possono avere una media simile ma comportamenti molto differenti.

Per esempio:

```text
Modello A
490
500
510
495
```

e:

```text
Modello B
100
900
150
850
```

possono avere medie relativamente simili, ma il primo è molto più stabile.

La deviazione standard permette di rappresentare questa differenza.

---

## 30.3 Mediana

La mediana viene conservata perché è meno sensibile ai valori estremi.

Un singolo episodio eccezionalmente buono o cattivo può modificare sensibilmente la media.

La mediana permette quindi di avere una seconda misura della prestazione tipica del modello.

---

# 31. Salvataggio dei risultati

L'evaluation produce due livelli di informazioni.

Il primo contiene i dati episodio per episodio.

Per esempio:

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

Il secondo contiene invece un summary statistico.

Mantenere entrambi i livelli è utile perché consente di effettuare nuove analisi in seguito senza dover rieseguire il modello.

I dati grezzi potranno essere utilizzati per:

* confrontare PPO e DQN pista per pista;
* costruire grafici;
* calcolare nuove statistiche;
* individuare outlier;
* analizzare le cause di fallimento;
* confrontare diversi checkpoint.

---

# 32. Valutazione qualitativa

Oltre alla valutazione numerica, `evaluate.py` permette di visualizzare gli episodi o registrarli in video.

Il video non sostituisce le metriche quantitative.

Serve però a interpretare meglio il comportamento.

Per esempio, osservando direttamente il modello è possibile distinguere tra:

* oscillazioni continue;
* guida lenta ma stabile;
* accelerazione eccessiva;
* difficoltà nelle curve;
* difficoltà nel recuperare una traiettoria sbagliata;
* uscita frequente dal circuito.

Per la presentazione finale sarà particolarmente utile mostrare PPO e DQN sulla stessa pista.

In questo modo i risultati numerici potranno essere affiancati da un confronto visivo immediato.

---

# 33. Validation e test finale

L'evaluation periodica durante il training introduce una distinzione importante tra **validation** e **test finale**.

Supponiamo di utilizzare durante il training i seed:

```text
100
101
102
103
104
```

Se questi seed vengono utilizzati continuamente per osservare le prestazioni e scegliere il checkpoint migliore, stanno influenzando indirettamente le decisioni sul modello.

Non rappresentano quindi più un test completamente indipendente.

È preferibile separarli dai seed utilizzati per la valutazione conclusiva.

Una possibile struttura è:

```text
TRAINING
   │
   ▼
VALIDATION
seed 100-104
   │
   ▼
scelta del modello/checkpoint
   │
   ▼
TEST FINALE
seed 200-219
```

I valori numerici sono soltanto un esempio.

L'aspetto importante è che i seed del test finale non vengano utilizzati per scegliere il modello.

---

# 34. Protocollo sperimentale finale

Il confronto finale dovrà mantenere costanti il maggior numero possibile di variabili.

Una configurazione ideale può essere rappresentata così:

| Variabile            | PPO            | DQN            |
| -------------------- | -------------- | -------------- |
| Ambiente             | CarRacing-v2   | CarRacing-v2   |
| Osservazione         | RGB 96×96      | RGB 96×96      |
| Action space         | Discreto       | Discreto       |
| Numero azioni        | 5              | 5              |
| Max episode steps    | 1000           | 1000           |
| Domain randomization | False          | False          |
| Evaluation seed      | Uguali         | Uguali         |
| Evaluation           | Deterministica | Deterministica |
| Budget training      | Confrontabile  | Confrontabile  |

Lo schema generale sarà:

```text
                     CarRacing
                         │
                         ▼
               Configurazione comune
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
            PPO                     DQN
             │                       │
             ▼                       ▼
      training seed multipli   training seed multipli
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                 validation comune
                         │
                         ▼
                scelta del checkpoint
                         │
                         ▼
                    test comune
                         │
                         ▼
                 analisi risultati
```

---

# 35. Dimensioni del confronto

Il confronto finale non verrà ridotto a una singola metrica.

Saranno considerate diverse dimensioni.

---

## 35.1 Prestazione finale

Saranno analizzate metriche come:

* mean reward;
* median reward;
* completion rate;
* mean track completion.

---

## 35.2 Stabilità

La stabilità sarà valutata attraverso:

* deviazione standard;
* variabilità tra gli episodi;
* variabilità tra training seed differenti.

---

## 35.3 Sample efficiency

Le evaluation intermedie permetteranno di osservare quanto velocemente i due algoritmi apprendono.

La domanda non sarà quindi soltanto:

> Quale algoritmo raggiunge la prestazione finale migliore?

ma anche:

> Quanto training è necessario per raggiungere quella prestazione?

Un algoritmo potrebbe raggiungere un risultato simile a un altro utilizzando molte meno interazioni con l'ambiente.

---

## 35.4 Comportamento qualitativo

La valutazione numerica sarà completata da video degli agenti.

Questo permetterà di osservare direttamente differenze difficili da rappresentare attraverso un singolo valore numerico.

---

# 36. Stato attuale del progetto

La pipeline principale è attualmente funzionante.

Sono già state implementate o verificate:

* configurazione centralizzata di CarRacing;
* spazio delle azioni discreto;
* random agent;
* PPO;
* DQN;
* `CnnPolicy`;
* training parametrico;
* gestione dei seed;
* distinzione tra smoke, pilot e final;
* salvataggio dei modelli;
* evaluation separata dal training;
* evaluation periodica;
* evaluation su seed fissi;
* evaluation deterministica;
* metriche per episodio;
* summary statistico;
* track completion;
* completion rate;
* termination reason;
* registrazione video;
* prime smoke run.

Le attività ancora in corso o previste comprendono:

1. completamento del sistema di checkpoint;
2. verifica completa del resume;
3. definizione definitiva dei seed di validation;
4. definizione dei seed di test;
5. esecuzione delle pilot run;
6. scelta del budget finale di training;
7. esecuzione di training multi-seed;
8. evaluation finale;
9. aggregazione dei risultati;
10. produzione dei grafici;
11. confronto PPO-DQN;
12. conclusioni finali.

---

# 37. Limiti del progetto

Il confronto presenta diversi limiti che devono essere tenuti in considerazione nell'interpretazione dei risultati.

---

## 37.1 Action space discreto

CarRacing nasce anche come problema di controllo continuo.

Ridurre il controllo a cinque azioni rende il sistema meno preciso.

La scelta è stata introdotta principalmente per poter confrontare DQN e PPO nello stesso spazio delle azioni.

I risultati non descrivono quindi necessariamente la prestazione massima ottenibile da PPO su CarRacing.

---

## 37.2 Domain randomization disabilitata

L'ambiente viene utilizzato senza randomizzazione del dominio visuale.

Il progetto non misura quindi direttamente la robustezza degli agenti rispetto a variazioni grafiche significative.

---

## 37.3 Risorse computazionali

Gli algoritmi lavorano direttamente su immagini e utilizzano reti convoluzionali.

Il costo computazionale è quindi elevato rispetto a problemi con osservazioni numeriche.

Inoltre i training vengono eseguiti su CPU, principalmente per mantenere una configurazione simile sulle macchine utilizzate nello sviluppo.

Questo limita la velocità con cui possono essere eseguiti esperimenti molto lunghi.

---

## 37.4 Hyperparameter

Non viene eseguito inizialmente un tuning molto esteso.

In particolare PPO utilizza principalmente i valori standard di Stable-Baselines3, mentre per DQN sono stati modificati alcuni parametri soprattutto per motivi di memoria.

Le conclusioni saranno quindi legate anche alle configurazioni specifiche utilizzate.

---

# 38. Interpretazione corretta dei risultati

Le conclusioni non dovranno essere formulate in termini assoluti.

Non sarà corretto scrivere:

> PPO è migliore di DQN.

Una formulazione più corretta sarà:

> Nella configurazione di CarRacing utilizzata nel progetto e con il protocollo sperimentale adottato, PPO ha ottenuto prestazioni mediamente superiori a DQN secondo le metriche considerate.

Oppure il contrario, se saranno i dati a mostrarlo.

La differenza è importante perché il risultato dipenderà da:

* ambiente;
* discretizzazione;
* rete;
* hyperparameter;
* budget di training;
* seed;
* protocollo di evaluation.

---

# 39. Possibili sviluppi futuri

Una volta concluso il confronto principale, il progetto potrebbe essere esteso in diverse direzioni.

Una prima possibilità sarebbe confrontare PPO nelle due configurazioni:

```text
PPO discreto
vs
PPO continuo
```

Questo permetterebbe di misurare quanto la discretizzazione influenzi realmente le prestazioni.

Un'altra possibilità sarebbe attivare la domain randomization e analizzare la robustezza dei modelli.

Si potrebbe inoltre studiare l'effetto di diversi hyperparameter, per esempio:

* dimensione del replay buffer;
* learning rate;
* exploration schedule di DQN;
* batch size;
* rollout size di PPO;
* numero di epoche di PPO.

Altre estensioni potrebbero includere algoritmi come:

* Double DQN;
* Dueling DQN;
* SAC;
* TD3.

Queste estensioni non sono però necessarie per rispondere alla domanda principale del progetto.

---

# 40. Conclusione metodologica

Il lavoro svolto fino a questo punto ha portato alla costruzione di una pipeline comune per l'addestramento e la valutazione di PPO e DQN sull'ambiente CarRacing.

Il punto più importante del progetto non consiste soltanto nell'ottenere due modelli funzionanti.

Il confronto deve essere costruito in modo da mantenere il più possibile le stesse condizioni sperimentali.

Per questo motivo sono stati mantenuti comuni:

* ambiente;
* osservazioni;
* spazio delle azioni;
* limite degli episodi;
* configurazione visuale;
* evaluation seed;
* metriche;
* metodo di valutazione.

L'utilizzo di più training seed permetterà inoltre di tenere conto della normale variabilità presente negli esperimenti di Reinforcement Learning.

I checkpoint e le evaluation periodiche permetteranno di studiare non soltanto il risultato finale, ma anche l'evoluzione dell'apprendimento durante il training.

La fase successiva consiste quindi nel completare il sistema di checkpoint, effettuare le pilot run e successivamente passare agli esperimenti utilizzati per il confronto finale.
