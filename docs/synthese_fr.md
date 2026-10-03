# Prévision de consommation électrique par compteur : synthèse

*Public : responsable énergie / exploitation. Une page.*

## La question
Peut-on prévoir, la veille pour le lendemain et heure par heure, la consommation de chaque
bâtiment, et repérer automatiquement les compteurs défaillants et les consommations anormales ?

## Les données
76 compteurs électriques sur deux campus (Londres et Dublin), relevés horaires 2016–2017
(1,5 million de mesures), croisés avec la météo locale (température, point de rosée, vent).

## Ce que nous avons trouvé

**1. La prévision est fiable et nettement meilleure que la méthode habituelle.**
La combinaison de deux modèles (gradient boosting et réseau de neurones récurrent) se trompe
**35 % de moins** que la référence « même heure la semaine dernière ». L'écart d'erreur typique
est de 11 %. Elle fait mieux que chaque modèle seul, de façon statistiquement confirmée sur
**63 compteurs sur 76**, et n'est moins bonne sur aucun (test de Diebold-Mariano, corrigé
pour tests multiples).

**Et avec une marge d'incertitude fiable.** Chaque prévision est livrée avec une fourchette
P10–P90. Après calibration, elle contient bien **79 % des valeurs réelles** (objectif : 80 %).
On peut donc dire par exemple : « demain à 14 h, 80 % de chances entre X et Y kWh ». C'est
directement utilisable pour les achats d'énergie.

**2. La météo compte peu ici, et c'est une information en soi.**
Seuls **7 bâtiments sur 76** réagissent nettement à la température. Ces campus se chauffent au
gaz, pas à l'électricité. Pour les 4 bâtiments chauffés électriquement, la consommation
augmente jusqu'à **20 % par degré en moins** sous ~17 °C. Pour un parc avec chauffage électrique
et climatisation, comme en zone méditerranéenne, l'effet météo serait bien plus fort. Le
modèle le prendra en compte bâtiment par bâtiment.

**3. 6,4 % des mesures sont inexploitables** (compteur bloqué, coupure à zéro, pic isolé),
soit 919 incidents de comptage. 12 compteurs sont trop dégradés pour être modélisés.
Cette liste est à transmettre à l'équipe comptage.

**4. Les anomalies de consommation sont triées pour être actionnables.**
Sur 2 253 écarts détectés, 1 555 touchent tout un site le même jour (épisode neigeux de
décembre 2017, samedis ouverts, jours de fermeture) : ce sont des effets calendaires ou
météo, pas des problèmes de bâtiment. Restent **258 anomalies prioritaires sur 36 bâtiments**,
chacune représentant au moins 20 % d'une journée normale. Exemple : un bâtiment dont la
consommation chute des deux tiers le 20 septembre 2017 et ne remonte pas, sans doute un
sous-compteur recâblé ou une aile fermée. À vérifier sur site.

## Recommandations
1. Utiliser la prévision J+1 et sa fourchette pour le pilotage et les achats d'énergie.
   Réentraînement mensuel automatique, avec alerte si le modèle ne bat plus la référence.
2. Intégrer le calendrier propre à chaque site (examens, fermetures) : c'est la principale
   source d'erreur restante.
3. Remplacer la météo observée par des prévisions météo J+1 (Météo-France) pour la mise en
   production.
4. Traiter chaque mois la liste des anomalies prioritaires et des compteurs défaillants.

*Outils : Databricks (préparation des données à l'échelle), Dataiku (modélisation, suivi,
tableau de bord, scénario de réentraînement).*
