# Dictionnaire des features — `feat-grid` (SeizeIT2)

Ce document décrit chaque colonne des fichiers `feat-grid-*.csv` produits par
[`merge_feat_grid.py`](merge_feat_grid.py) (voir [README.md](README.md) pour le
pipeline complet). Chaque ligne correspond à **1 seconde** de la grille temporelle
d'un run.

Trois familles de colonnes :

1. **Colonnes de contexte** — temps, identifiants, label de crise.
2. **`hrv_*`** — variabilité de la fréquence cardiaque (VFC), calculée par la
   librairie [`hrvanalysis`](https://github.com/Aura-healthcare/hrvanalysis) sur
   les intervalles RR extraits de l'ECG (voir
   [`compute_hrvanalysis_features.py`](../src/usecase/compute_hrvanalysis_features.py)).
3. **`acc_EEG_SD_ACC_*`** — statistiques temporelles et spectrales calculées sur
   le signal de l'accéléromètre intégré au boîtier d'acquisition EEG (canal
   `ACC`, capteur `EEG_SD`), de type TSFEL, sur les axes `x`, `y`, `z` et la
   norme du vecteur (`norme_1`).

---

## 1. Colonnes de contexte

| Colonne | Description |
|---|---|
| `elapsed_s` | Temps écoulé, en secondes, depuis le début du run (colonne d'index de la grille, un point par seconde). |
| `timestamp` | Horodatage absolu reconstruit à partir de l'ancre temporelle HRV (première valeur `timestamp` du fichier `feats_*_ecg_fast.csv`) + `elapsed_s`. |
| `patient_id` | Identifiant BIDS du patient (ex : `sub-001`), extrait du `run_id`. |
| `run_id` | Identifiant complet du run BIDS : `sub-XXX_ses-YY_task-szMonitoring_run-ZZ`. |
| `label` | `1` si la seconde tombe dans la fenêtre `[onset, onset+duration]` d'un événement de crise (`eventType` commençant par `sz`), sinon `0`. |
| `seizure_type` | Type exact de la crise en cours (ex : `sz_foc_ia_nm`), issu de la colonne `eventType` du `events.tsv` BIDS ; vide (`NA`) hors crise. |
| `hrv_available` | `True` si une fenêtre HRV valide a été trouvée pour cette seconde (jointure `merge_asof`, tolérance ±500 ms), sinon `False`. |
| `acc_available` | `True` si une fenêtre ACC valide a été trouvée pour cette seconde (jointure `merge_asof`, tolérance ±2 500 ms), sinon `False`. |

---

## 2. Features HRV (`hrv_*`)

Calculées par fenêtre **glissante de 1 s**, sur des intervalles RR nettoyés
(suppression des outliers, interpolation, retrait des battements ectopiques,
méthode Malik). Selon la quantité d'historique disponible à l'instant courant,
trois familles de features sont calculées sur des fenêtres d'observation de
tailles croissantes :

- **temporelles** (`short_window` = 10 s)
- **non linéaires** (`medium_window` = 60 s)
- **fréquentielles** (`large_window` = 150 s)

C'est pourquoi une même ligne peut avoir certaines colonnes `hrv_*` renseignées
et d'autres à `NaN` (pas encore assez d'historique pour la fenêtre requise).

### 2.1 Index / repérage

| Colonne | Description |
|---|---|
| `hrv_interval_index` | Numéro de la fenêtre glissante de 1 s (0, 1, 2, …) depuis le début du run. |
| `hrv_interval_start_time` | Début de cette fenêtre, en millisecondes depuis le début du run (= `elapsed_s * 1000` avant fusion). |

### 2.2 Domaine temporel (fenêtre courte, 10 s)

Statistiques calculées directement sur la série des intervalles NN (RR normaux,
en millisecondes).

| Colonne | Description |
|---|---|
| `hrv_mean_nni` | Moyenne des intervalles NN (ms). |
| `hrv_sdnn` | Écart-type des intervalles NN (ms) — variabilité globale. |
| `hrv_sdsd` | Écart-type des différences successives entre intervalles NN adjacents (ms). |
| `hrv_nni_50` | Nombre de paires d'intervalles NN successifs différant de plus de 50 ms. |
| `hrv_pnni_50` | `nni_50` exprimé en pourcentage du nombre total d'intervalles NN — proxy de l'activité parasympathique. |
| `hrv_nni_20` | Nombre de paires d'intervalles NN successifs différant de plus de 20 ms. |
| `hrv_pnni_20` | `nni_20` exprimé en pourcentage du nombre total d'intervalles NN. |
| `hrv_rmssd` | Racine carrée de la moyenne des carrés des différences successives entre intervalles NN (ms) — reflète l'activité vagale à court terme. |
| `hrv_median_nni` | Médiane des intervalles NN (ms). |
| `hrv_range_nni` | Étendue (max − min) des intervalles NN (ms). |
| `hrv_cvsd` | Coefficient de variation des différences successives (`rmssd / mean_nni`). |
| `hrv_cvnni` | Coefficient de variation des intervalles NN (`sdnn / mean_nni`). |
| `hrv_mean_hr` | Fréquence cardiaque moyenne (bpm), dérivée des intervalles NN. |
| `hrv_max_hr` | Fréquence cardiaque maximale sur la fenêtre (bpm). |
| `hrv_min_hr` | Fréquence cardiaque minimale sur la fenêtre (bpm). |
| `hrv_std_hr` | Écart-type de la fréquence cardiaque instantanée sur la fenêtre (bpm). |

### 2.3 Domaine fréquentiel (fenêtre longue, 150 s)

Densité spectrale de puissance de la série des intervalles NN (méthode de Welch
dans `hrvanalysis`), décomposée en bandes.

| Colonne | Description |
|---|---|
| `hrv_lf` | Puissance dans la bande basse fréquence (LF, ~0.04–0.15 Hz) — influencée par les systèmes sympathique et parasympathique. |
| `hrv_hf` | Puissance dans la bande haute fréquence (HF, ~0.15–0.4 Hz) — reflète principalement l'activité parasympathique (couplée à la respiration). |
| `hrv_vlf` | Puissance dans la bande très basse fréquence (VLF, ~0.003–0.04 Hz). |
| `hrv_lf_hf_ratio` | Ratio `lf / hf` — indicateur usuel (mais discuté) de l'équilibre sympatho-vagal. |

### 2.4 Non linéaire / géométrique (fenêtre moyenne, 60 s)

| Colonne | Description |
|---|---|
| `hrv_csi` | Cardiac Sympathetic Index — dérivé du nuage de Poincaré (`sd2/sd1` corrigé), associé au tonus sympathique. |
| `hrv_cvi` | Cardiac Vagal Index — dérivé du nuage de Poincaré, associé au tonus vagal. |
| `hrv_Modified_csi` | Variante modifiée du CSI (`sd2² / sd1`). |
| `hrv_sampen` | Sample Entropy des intervalles NN — mesure la complexité/irrégularité du signal (plus haut = plus imprévisible). |
| `hrv_sd1` | Écart-type du nuage de Poincaré perpendiculairement à la diagonale d'identité — variabilité à court terme, corrélé à `rmssd`. |
| `hrv_sd2` | Écart-type du nuage de Poincaré le long de la diagonale d'identité — variabilité à long terme. |
| `hrv_ratio_sd2_sd1` | Ratio `sd2 / sd1` — équilibre entre variabilité longue et courte durée. |

---

## 3. Features accélérométriques (`acc_EEG_SD_ACC_*`)

Calculées par fenêtre glissante sur le signal brut de l'accéléromètre du
boîtier EEG. Le même jeu de ~30 features est répété **à l'identique** pour
chacun des 4 canaux suivants (préfixe `acc_EEG_SD_ACC_<axe>_...`) :

- `x`, `y`, `z` : les trois axes bruts de l'accéléromètre ;
- `norme_1` : la norme du vecteur d'accélération (magnitude 3D, combinant x/y/z),
  indépendante de l'orientation du capteur.

Le tableau ci-dessous décrit le suffixe de feature commun à ces 4 axes.

### 3.1 Statistiques temporelles

| Suffixe | Description |
|---|---|
| `_mean` | Moyenne du signal sur la fenêtre. |
| `_std` | Écart-type du signal. |
| `_max` | Valeur maximale du signal. |
| `_pk_pk_distance` | Distance pic-à-pic : `max − min` du signal sur la fenêtre. |
| `_distance` | Longueur de la courbe du signal (somme des distances euclidiennes entre points successifs). |
| `_mean_abs_diff` | Moyenne des valeurs absolues des différences entre échantillons successifs. |
| `_sum_abs_diff` | Somme des valeurs absolues des différences entre échantillons successifs. |
| `_slope` | Pente de la régression linéaire du signal sur la fenêtre (tendance). |
| `_abs_energy` | Énergie absolue du signal : somme des carrés des échantillons. |
| `_zero_crossing` | Nombre de passages par zéro du signal (autour de sa moyenne). |
| `_autocorr` | Autocorrélation du signal (mesure de répétitivité / périodicité). |
| `_entropy` | Entropie du signal dans le domaine temporel (mesure de désordre/imprévisibilité). |

### 3.2 Statistiques spectrales

Calculées à partir du spectre de puissance (FFT) du signal sur la fenêtre.

| Suffixe | Description |
|---|---|
| `_spectral_entropy` | Entropie de la distribution de puissance spectrale — mesure la complexité/dispersion du spectre. |
| `_wavelet_entropy` | Entropie calculée sur une décomposition en ondelettes du signal. |
| `_spectral_centroid` | Barycentre du spectre de puissance (Hz) — fréquence « moyenne » pondérée par l'énergie. |
| `_median_frequency` | Fréquence en dessous de laquelle se trouve 50 % de l'énergie totale du spectre. |
| `_max_frequency` | Fréquence contenant le maximum d'énergie du spectre. |
| `_max_power_spectrum` | Valeur de puissance maximale du spectre. |
| `_fundamental_frequency` | Fréquence fondamentale (composante périodique dominante) du signal. |
| `_spectral_roll_on` | Fréquence en dessous de laquelle se trouve 5 % de l'énergie cumulée du spectre. |
| `_spectral_roll_off` | Fréquence en dessous de laquelle se trouve 95 % de l'énergie cumulée du spectre. |
| `_power_bandwidth` | Largeur de bande entre `spectral_roll_on` et `spectral_roll_off`. |
| `_spectral_variation` | Variation du spectre de puissance entre fenêtres successives (nouveauté spectrale). |
| `_spectral_slope` | Pente de la décroissance de l'amplitude spectrale en fonction de la fréquence. |
| `_spectral_decrease` | Taux de décroissance normalisé de l'amplitude spectrale avec la fréquence. |
| `_spectral_kurtosis` | Aplatissement (kurtosis) de la distribution de puissance spectrale — présence de pics marqués vs spectre plat. |
| `_spectral_distance` | Distance cumulée entre le spectre observé et une ligne de tendance de référence (mesure d'irrégularité spectrale). |
| `_human_range_energy` | Proportion de l'énergie du signal contenue dans la bande de fréquence typique du mouvement humain (~0.6–2.5 Hz). |

### 3.3 Bandes de fréquence fixes (`F_a_b`)

Énergie du spectre intégrée sur des bandes de 1 Hz de large, de 0 à 12 Hz
(indépendamment de l'axe x/y/z/norme — une seule série de colonnes) :

| Colonne | Description |
|---|---|
| `acc_EEG_SD_ACC_F_0_1` … `acc_EEG_SD_ACC_F_11_12` | Énergie spectrale dans la bande `[n, n+1[` Hz (12 bandes, de 0–1 Hz à 11–12 Hz). |

### 3.4 Bandes de fréquence personnalisées (`F_custom_a_b`)

Mêmes principe, mais sur des bandes plus larges, calquées sur des bandes
d'intérêt habituellement utilisées en EEG (delta/thêta/alpha/bêta) et
réutilisées ici pour caractériser le mouvement :

| Colonne | Description |
|---|---|
| `acc_EEG_SD_ACC_F_custom_0_5` | Énergie spectrale dans la bande 0–5 Hz. |
| `acc_EEG_SD_ACC_F_custom_5_8` | Énergie spectrale dans la bande 5–8 Hz. |
| `acc_EEG_SD_ACC_F_custom_8_12` | Énergie spectrale dans la bande 8–12 Hz. |
| `acc_EEG_SD_ACC_F_custom_12_25` | Énergie spectrale dans la bande 12–25 Hz. |

---

## Notes

- Toutes les colonnes `hrv_*` et `acc_EEG_SD_ACC_*` peuvent être `NaN` quand la
  fenêtre correspondante n'a pas pu être calculée (pas assez d'historique en
  début de run, capteur absent, etc.) — se référer à `hrv_available` /
  `acc_available` pour filtrer les lignes exploitables.
- Le fichier `feat-grid-intersect_*.csv` ne garde que les lignes où
  `hrv_available` **et** `acc_available` valent `True` ; `feat-grid-union_*.csv`
  garde les lignes où **l'un des deux** est disponible.
