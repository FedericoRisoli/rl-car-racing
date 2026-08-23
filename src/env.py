# Il file ENV.PY rappresenta il file di configurazione e creazione dell'ambiente di Reinforcement Learning.
import gymnasium as gym

# ID dell'ambiente utilizzato nel progetto.
ENV_ID = "CarRacing-v2"

# Imposta lo spazio delle azioni come discreto. In questo modo, l'agente può scegliere fra 5 azioni:
# 0 = nessuna azione, 1 = sterza a sinistra, 2 = sterza a destra, 3 = accelera, 4 = frena.
CONTINUOUS = False

# Percentuale minima della pista che dev'essere visitata affinchè, tornando nella zona iniziale, il giro
# possa essere considerato completato.
LAP_COMPLETE_PERCENT = 0.95

# Disbilita la randomizzazione dei colori dell'ambiente.
DOMAIN_RANDOMIZE = False

# Numero massimo di step consentiti in un singolo episodio.
MAX_EPISODE_STEPS = 1000

# Metodo che crea e restituisce l'ambiente CarRacing utilizzato nel progetto.
# Il parametro render_mode viene lasciato variabile perché dipende dallo scopo con cui viene creato l'ambiente:
def make_env(render_mode=None):
    return gym.make(
        ENV_ID,
        continuous=CONTINUOUS,
        lap_complete_percent=LAP_COMPLETE_PERCENT,
        domain_randomize=DOMAIN_RANDOMIZE,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
    )