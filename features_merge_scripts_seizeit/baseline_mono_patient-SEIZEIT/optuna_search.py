"""
optuna_search.py — Hyperparameter search over config.py using Optuna.

Runs the full launch_train.py pipeline (embedding pretraining + classifier
fine-tuning) once per trial, mutating the shared config dicts in place, and
reports a test-set metric back to Optuna.

Usage:
    pip install optuna
    python optuna_search.py --n-trials 30
    python optuna_search.py --n-trials 100 --metric f1 --embedding-epochs 20 --classifier-epochs 15

    # Resumable / parallelizable study backed by a SQLite file:
    python optuna_search.py --study-name seizure-search --storage sqlite:///optuna_seizure.db --n-trials 50

    # Compare loss types: runs one full Optuna study per loss_type (same
    # trial budget each) and ranks them by best test metric:
    python optuna_search.py --compare-losses --n-trials 20 --embedding-epochs 20 --classifier-epochs 15
    python optuna_search.py --compare-losses --losses supcon batch_hard_triplet --n-trials 20

After the search, the best hyperparameters are printed and written to
<output-dir>/best_params.json (or <output-dir>/<loss_type>/best_params.json
and <output-dir>/loss_comparison.json in --compare-losses mode). Re-apply
them by hand in config.py before a final full-length training run.
"""
import argparse  # parsing des arguments en ligne de commande (--n-trials, --metric, etc.)
import json      # sauvegarde des meilleurs hyperparamètres dans un fichier .json
import logging   # logs (mêmes logs que ceux produits par launch_train.py, car même module logging)
import os        # gestion des chemins et création de dossiers
import shutil    # suppression récursive de dossiers (option --keep-only-best)

import optuna    # librairie de recherche d'hyperparamètres (Bayesian optimization par défaut)

import config        # module config.py — on l'importe en tant que module (pas "from config import ...")
                      # pour pouvoir modifier ses dicts en place (voir suggest_hyperparameters)
import launch_train   # notre pipeline d'entraînement existant (Phase 1 embedding + Phase 2 classifieur)


def suggest_hyperparameters(trial, base_loss_type):
    """Mutate the shared config dicts in place.

    Other modules imported these dicts by reference (`from config import
    MODEL_CONFIG`), so keys must be assigned in place (config.MODEL_CONFIG['x']
    = ...) — reassigning config.MODEL_CONFIG itself would only rebind the name
    inside this module and leave everyone else pointing at the old dict.
    """
    # trial = un essai Optuna. Chaque appel à trial.suggest_xxx(nom, bornes...) :
    #   - tire une valeur (aléatoire au début, puis guidée par l'algorithme bayésien
    #     une fois que plusieurs essais ont été observés)
    #   - enregistre cette valeur sous le nom donné, pour que study.best_params
    #     puisse ensuite l'afficher/la sauvegarder

    # --- model ---
    # suggest_categorical : tire une valeur parmi une liste discrète de choix
    config.MODEL_CONFIG['embedding_dim'] = trial.suggest_categorical('embedding_dim', [16, 32, 64, 128])
    # suggest_int : tire un entier dans [1, 4] inclus
    config.MODEL_CONFIG['num_residual_blocks'] = trial.suggest_int('num_residual_blocks', 1, 4)
    # suggest_float : tire un flottant dans l'intervalle donné (distribution uniforme par défaut)
    config.MODEL_CONFIG['embedding_dropout'] = trial.suggest_float('embedding_dropout', 0.1, 0.5)
    config.MODEL_CONFIG['classifier_dropout'] = trial.suggest_float('classifier_dropout', 0.1, 0.5)

    # --- embedding training ---
    # log=True : tire la valeur sur une échelle logarithmique (adapté aux learning rates,
    # qui ont un impact similaire qu'on soit à 1e-4 ou 2e-4, mais pas entre 1e-4 et 1e-2)
    config.EMBEDDING_TRAINING_CONFIG['learning_rate'] = trial.suggest_float('embedding_lr', 1e-4, 1e-2, log=True)
    config.EMBEDDING_TRAINING_CONFIG['scheduler_step_size'] = trial.suggest_int('embedding_scheduler_step_size', 5, 20)
    config.EMBEDDING_TRAINING_CONFIG['scheduler_gamma'] = trial.suggest_float('embedding_scheduler_gamma', 0.1, 0.9)

    # Certains hyperparamètres ne concernent que certaines loss (temperature/pk_k pour
    # supcon, margin pour triplet). base_loss_type est fixé une fois pour toute l'étude
    # (c'est celui déjà présent dans config.py), donc chaque essai explore un seul type
    # de loss — on ne mélange pas les espaces de recherche entre losses différentes.
    if base_loss_type == 'supcon':
        config.EMBEDDING_TRAINING_CONFIG['temperature'] = trial.suggest_float('temperature', 0.03, 0.5, log=True)
        config.EMBEDDING_TRAINING_CONFIG['pk_k'] = trial.suggest_categorical('pk_k', [16, 32, 64])
    elif base_loss_type in ('triplet', 'batch_hard_triplet'):
        config.EMBEDDING_TRAINING_CONFIG['margin'] = trial.suggest_float('margin', 0.2, 2.0)

    # --- classifier training ---
    config.CLASSIFIER_TRAINING_CONFIG['learning_rate'] = trial.suggest_float('classifier_lr', 1e-5, 1e-2, log=True)
    # suggest_categorical sur un booléen : Optuna teste aussi bien freeze que fine-tuning complet
    config.CLASSIFIER_TRAINING_CONFIG['freeze_embeddings'] = trial.suggest_categorical('freeze_embeddings', [True, False])

    # --- data ---
    config.DATA_CONFIG['undersampling_ratio'] = trial.suggest_categorical('undersampling_ratio', [10, 20, 50, 100])


def make_objective(args, output_dir):
    # base_loss_type est lu une seule fois, avant le début des essais (pas à chaque essai) :
    # c'est la loss définie dans config.py au moment du lancement du script, elle ne change
    # pas pendant la recherche (voir suggest_hyperparameters ci-dessus).
    base_loss_type = config.EMBEDDING_TRAINING_CONFIG['loss_type']

    # make_objective retourne une fonction "objective" fermée sur args/output_dir/base_loss_type
    # (closure) : c'est la signature attendue par study.optimize(objective, ...) — Optuna appelle
    # objective(trial) pour chaque essai, un seul argument.
    def objective(trial):
        # 1. Tire un jeu d'hyperparamètres pour cet essai et les écrit dans les dicts de config.py
        suggest_hyperparameters(trial, base_loss_type)

        # 2. Chaque essai a son propre dossier de résultats et de checkpoints (numéroté),
        # pour ne pas écraser les résultats des autres essais et pouvoir les comparer après coup.
        trial_dir = os.path.join(output_dir, f"trial_{trial.number:04d}")
        config.DATA_CONFIG['results_dir'] = trial_dir
        config.DATA_CONFIG['checkpoint_dir'] = os.path.join(trial_dir, 'checkpoints')

        # 3. Nombre d'epochs réduit pendant la recherche (paramétrable en ligne de commande)
        # pour que chaque essai soit rapide — on ne veut pas payer 50+30 epochs complètes
        # pour chacun des dizaines d'essais de la recherche.
        config.EMBEDDING_TRAINING_CONFIG['epochs'] = args.embedding_epochs
        config.CLASSIFIER_TRAINING_CONFIG['epochs'] = args.classifier_epochs
        # UMAP désactivé par défaut (coûteux et inutile pendant la recherche, seulement utile
        # une fois qu'on a trouvé les meilleurs hyperparamètres et qu'on relance un run complet)
        config.EVAL_CONFIG['generate_umap'] = args.generate_umap

        try:
            # 4. Lance le pipeline complet (Phase 1 embedding + Phase 2 classifieur + éval)
            # exactement comme `python launch_train.py`, mais en récupérant les métriques
            # de test au lieu de simplement les logger.
            test_metrics = launch_train.main()
        except Exception:
            # Si un essai plante (ex: combinaison d'hyperparamètres invalide), on ne fait pas
            # échouer toute la recherche : on logue l'erreur complète et on "prune" (élague)
            # cet essai — Optuna l'ignore et continue avec le suivant.
            logging.exception(f"Trial {trial.number} raised an exception, pruning")
            raise optuna.TrialPruned()

        if test_metrics is None:
            raise optuna.TrialPruned()

        # 5. Récupère la métrique choisie (--metric, roc_auc par défaut) dans le dict retourné
        # par launch_train.main() (voir eval_classifier_head dans eval.py).
        value = test_metrics.get(args.metric)
        if value is None:
            # roc_auc peut être None si le split de test ne contient qu'une seule classe
            # (cas pathologique mentionné dans launch_train.py) — dans ce cas on élague
            # aussi l'essai plutôt que de faire planter la recherche.
            logging.warning(
                f"Trial {trial.number}: metric '{args.metric}' unavailable "
                f"(likely a single class on the test split) — pruning."
            )
            raise optuna.TrialPruned()

        # 6. On garde les 3 métriques (même celles non optimisées) comme "user attributes"
        # du trial, pour pouvoir les consulter après coup (ex: dans best_params.json)
        # même si on n'a optimisé que sur roc_auc.
        trial.set_user_attr('accuracy', test_metrics.get('accuracy'))
        trial.set_user_attr('f1', test_metrics.get('f1'))
        trial.set_user_attr('roc_auc', test_metrics.get('roc_auc'))

        # 7. La valeur retournée est ce qu'Optuna cherche à maximiser (direction='maximize'
        # dans study = optuna.create_study(...) plus bas).
        return value

    return objective


def run_study(args, output_dir, study_name, storage):
    """Lance une étude Optuna complète (avec la loss_type actuellement définie
    dans config.EMBEDDING_TRAINING_CONFIG['loss_type']) et retourne l'étude
    terminée. Factorisé pour être appelé une fois en mode normal, ou une fois
    par loss en mode --compare-losses.
    """
    os.makedirs(output_dir, exist_ok=True)

    # optuna.create_study crée (ou recharge, si storage est fourni et load_if_exists=True)
    # l'objet qui pilote la recherche : il garde en mémoire tous les essais passés et décide
    # des prochains hyperparamètres à tester selon l'algorithme choisi (TPE par défaut).
    # - storage=None → étude en mémoire uniquement, perdue si le script s'arrête
    # - storage="sqlite:///fichier.db" → étude persistée sur disque, permet de reprendre
    #   la recherche plus tard ou de la paralléliser (plusieurs process pointant vers le
    #   même fichier .db)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',  # on cherche à maximiser la métrique (roc_auc/f1/accuracy)
        load_if_exists=storage is not None,
    )
    # study.optimize lance la boucle principale : pour chaque essai, appelle
    # objective(trial), jusqu'à n_trials essais ou jusqu'à expiration de timeout (secondes),
    # selon la limite atteinte en premier.
    study.optimize(
        make_objective(args, output_dir),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    return study


def print_and_save_best(study, args, output_dir, label=None):
    """Affiche le meilleur essai d'une étude et sauvegarde ses hyperparamètres
    dans best_params.json. `label` (ex: le loss_type) préfixe l'affichage
    quand on est appelé plusieurs fois depuis --compare-losses.
    """
    prefix = f"[{label}] " if label else ""

    print("\n" + "=" * 80)
    print(f"{prefix}Best trial: #{study.best_trial.number}  {args.metric}={study.best_value:.4f}")
    print(f"{prefix}Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    # Sauvegarde des meilleurs hyperparamètres dans un .json, pour pouvoir les reporter
    # à la main dans config.py ensuite (voir docstring en tête de fichier).
    best_path = os.path.join(output_dir, 'best_params.json')
    with open(best_path, 'w') as f:
        json.dump({
            'loss_type': label,
            'metric': args.metric,
            'value': study.best_value,
            'params': study.best_params,
            'trial_number': study.best_trial.number,
            'user_attrs': study.best_trial.user_attrs,  # les 3 métriques même non optimisées (cf. étape 6 ci-dessus)
        }, f, indent=4)
    print(f"{prefix}Best params saved to {best_path}")

    # Option de nettoyage : chaque essai produit un dossier complet (checkpoints,
    # embedding_model.pth, figures d'évaluation...), ce qui peut représenter beaucoup
    # d'espace disque sur --n-trials élevé. --keep-only-best supprime tous les dossiers
    # sauf celui du meilleur essai, une fois la recherche terminée.
    if args.keep_only_best:
        best_dir = os.path.join(output_dir, f"trial_{study.best_trial.number:04d}")
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if path != best_dir and os.path.isdir(path) and name.startswith('trial_'):
                shutil.rmtree(path)
        print(f"{prefix}Removed all trial folders except {best_dir}")

    return best_path


def main():
    # Déclaration des arguments en ligne de commande (voir docstring en tête de fichier
    # pour des exemples d'utilisation).
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for config.py")
    parser.add_argument('--n-trials', type=int, default=30,
                         help="Nombre d'essais par étude (par loss en mode --compare-losses)")
    parser.add_argument('--timeout', type=int, default=None, help="Overall search budget in seconds")
    parser.add_argument('--metric', choices=['roc_auc', 'f1', 'accuracy'], default='roc_auc')
    parser.add_argument('--embedding-epochs', type=int, default=20,
                         help="Phase-1 epochs per trial (lower than a full run for faster search)")
    parser.add_argument('--classifier-epochs', type=int, default=15,
                         help="Phase-2 epochs per trial")
    parser.add_argument('--generate-umap', action='store_true',
                         help="Generate UMAP plots for every trial (slow, off by default)")
    parser.add_argument('--output-dir', default='./optuna_results')  # dossier racine (contient un sous-dossier par essai)
    parser.add_argument('--study-name', default='seizure-contrastive-search')
    parser.add_argument('--storage', default=None,
                         help="e.g. sqlite:///optuna_seizure.db to persist/resume the study across runs")
    parser.add_argument('--keep-only-best', action='store_true',
                         help="Delete every trial's result folder except the best one once the search finishes")
    # --compare-losses : au lieu d'une seule étude sur la loss déjà choisie dans config.py,
    # on lance une étude Optuna complète et indépendante pour chaque loss_type de --losses,
    # avec le même budget d'essais, puis on les classe par meilleur score.
    parser.add_argument('--compare-losses', action='store_true',
                         help="Lance une étude Optuna séparée pour chaque loss_type (voir --losses) "
                              "et affiche un classement final par meilleur score")
    parser.add_argument('--losses', nargs='+',
                         choices=['contrastive', 'triplet', 'batch_hard_triplet', 'supcon'],
                         default=['contrastive', 'triplet', 'batch_hard_triplet', 'supcon'],
                         help="Loss types à comparer (utilisé seulement avec --compare-losses)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.compare_losses:
        # Mode normal : une seule étude, sur la loss déjà définie dans config.py.
        study = run_study(args, args.output_dir, args.study_name, args.storage)
        print_and_save_best(study, args, args.output_dir)
        return

    # --- Mode --compare-losses ---
    # On boucle sur chaque loss_type demandé : à chaque itération, on écrase
    # config.EMBEDDING_TRAINING_CONFIG['loss_type'] (lu par make_objective/run_study
    # via suggest_hyperparameters), puis on lance une étude Optuna complète et
    # indépendante pour cette loss, dans son propre sous-dossier.
    results = {}
    for loss_type in args.losses:
        print(f"\n{'#' * 80}\n# Comparing loss_type = {loss_type}\n{'#' * 80}")
        config.EMBEDDING_TRAINING_CONFIG['loss_type'] = loss_type

        loss_output_dir = os.path.join(args.output_dir, loss_type)
        # Un study_name distinct par loss : si --storage pointe vers le même fichier
        # sqlite pour toutes les losses, chaque étude reste bien séparée dedans.
        study_name = f"{args.study_name}-{loss_type}"

        study = run_study(args, loss_output_dir, study_name, args.storage)
        print_and_save_best(study, args, loss_output_dir, label=loss_type)

        results[loss_type] = {
            'value': study.best_value,
            'best_trial': study.best_trial.number,
            'params': study.best_params,
        }

    # Fusionne avec un loss_comparison.json existant plutôt que de l'écraser :
    # permet de relancer --compare-losses avec seulement une ou deux nouvelles
    # losses (ex: --losses contrastive) sans perdre les résultats déjà obtenus
    # pour les autres. Les losses relancées ici remplacent leur ancienne entrée.
    comparison_path = os.path.join(args.output_dir, 'loss_comparison.json')
    merged = {}
    if os.path.exists(comparison_path):
        with open(comparison_path) as f:
            previous = json.load(f)
        if previous.get('metric') != args.metric:
            logging.warning(
                f"loss_comparison.json was previously ranked on '{previous.get('metric')}', "
                f"merging in '{args.metric}' results — values across losses are no longer comparable."
            )
        for entry in previous.get('ranking', []):
            merged[entry['loss_type']] = {k: v for k, v in entry.items() if k != 'loss_type'}
    merged.update(results)

    # Classement final, du meilleur au moins bon selon la métrique choisie.
    ranking = sorted(merged.items(), key=lambda kv: kv[1]['value'], reverse=True)

    print("\n" + "=" * 80)
    print(f"LOSS COMPARISON (metric={args.metric})")
    print("=" * 80)
    for rank, (loss_type, info) in enumerate(ranking, start=1):
        print(f"  {rank}. {loss_type:<20} {args.metric}={info['value']:.4f}  (trial #{info['best_trial']})")
    print("=" * 80)

    with open(comparison_path, 'w') as f:
        json.dump({
            'metric': args.metric,
            'ranking': [{'loss_type': lt, **info} for lt, info in ranking],
        }, f, indent=4)
    print(f"Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
