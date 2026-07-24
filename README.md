# Installation


## Installation avec pixi (recommandé)

Installer **pixi**, le gestionnaire de projets utilisé ici, voir [la doc de pixi pour les instructions](https://pixi.prefix.dev/latest/installation/), c'est très rapide.

```
git clone git@github.com:roughnemezis/dem-tools.git
cd dem-tools
pixi install # installe un environnement virtuel pixi pour le projet
```



## Installation avec pip

`pip install -e .` depuis le dossier du projet devrait faire l'affaire


# Jouer avec le code

- `pixi run marimo edit scripts/algo_marimo.py` lance un notebook marimo de démo de l'algo de calcul de masques d'ombrages
- sinon le module `dem_tools.shadows` contient le code de l'algo de calcul d'ombrages (pour l'instant une seule échéance)



# Personnaliser la configuration

Les exemples donnés dans le dépôt (sous forme de scripts ou de notebooks marimo)
vont chercher leur données dans le dossier 
`/rd/cenfic3/cenobs/home/merzisenh/shared_data/dem_tools`
par défaut.

Si on n'est pas sur le réseau filaire Meteo-France, les données seront cherchées dans `$HOME/.dem-tools-data`.
Faire une copie du dossier de cenfic3 dans ce dossier si nécessaire.

