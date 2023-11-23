   # Apprentissage Automatique
![](https://i.makeagif.com/media/10-01-2022/2R4KL8.gif)

la machine 👆 (crampter)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L’apprentissage automatique est un concept dans le domaine de l’**intelligence artificielle** permettant de simuler un apprentissage pour une machine. Le principe consiste à analyser une **grande quantité de données** pour imiter un comportement de **réflexion** en réaction à un problème ou un besoin. Les données peuvent être acquises dans les banques de données déjà faites ou par des tests effectués par le programme.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;La technologie permet de découvrir des **patterns répétitifs** et proposer des prédictions en se basant sur des **statistiques**. Comparé aux méthodes d'analyse classiques, le machine learning est de plus en plus **efficace** plus il ya de données.

## Histoire de l’apprentissage automatique

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L'apprentissage automatique, aussi moderne qu'il en a l'air, n'est pas si récent que ça. Le célèbre mathématicien **Alan Turing** est en grande partie à l'origine de cette pratique avec sa "machine universelle" en 1936 et son concept de "**test de Turing**" en 1950, qui ont posé les bases de l'apprentissage automatique.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;En 1943, une simulation du fonctionnement des **neurones** est reproduite à l'aide d'un circuit électrique, c'est ce qui a composé la base du concept des **réseaux neuronaux** dans l'apprentissage machine  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L'informaticien américain **Arthur Samuel** est le premier à utiliser le terme "machine learning" avec un programme apprennant à jouer aux dames

## Liens avec le Big Data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Le **Big Bata** est une partie très importante du machine learning. C'est le concept de **stockage** d'une quantité gargantuesque de **données** dans une même banque de données, **facile d'accès**. Une sorte de regroupement de **toute l'information disponible sur internet**.
Le machine learning a besoin d'analyser une très grande quantité de données rapidement, le Big Data permet d'**automatiser le traitement des données**.

## Les différents types d'apprentiassages

- Apprentissage suppervisé
- Apprentissage non suppervisé
- Apprentissage semi suppervisé
- Apprentissage par renforcement

### Apprentissage suppervisé :
![](https://cdn.discordapp.com/attachments/1031448426442932245/1172475323015778314/sageyonce.gif?ex=656073bc&is=654dfebc&hm=bdfc3fd256db964e8c2b5a5b0cc1c780a55c20c6a2a50fdcaee050efb5fee00d&)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dans ce mode d'apprentissage nous donnons à la machine un set de données étiquetées, ce sont des données qui comportent des informations afin que le machine puisse les analyser et les exploiter par la suite. L'apprentissage suppervisé permet de gérer deux problèmes d'extractions de données:
- La classification : Il s'agit d'utilser les informations labellées afin de classifier différents éléments dans différentes catégories. Prenons par exemple un algorithme capable de différencier des images de camions et de voitures. Dans un premier temps nous donnons a la machine une image de camion en présisant pour chacune le vehicule sur l'image; ce sont les données labélisées. Nous donnons en suite une image a analyser. A partir des données labélisées, l'algorithme va chercher des similitudes pour voir a quelle catégorie l'image semble appartenir.

- La régréssion : La régression est utilisée pour comprendre la relation entre les variables dépendantes et indépendantes. Elle est couramment utilisée pour faire des projections, par exemple, sur le chiffre d'affaires d'une entreprise. La régression linéaire, la régression logistique et la régression polynomiale sont des algorithmes de régression couramment utilisés.

### Apprentissage non supervisé
![](https://cdn.discordapp.com/attachments/962004101552545852/1175025667763621939/old-man-working-david.gif?ex=6569baee&is=655745ee&hm=9ce52a67f57ec4e43151d4a4030f69be936b7d312a22fc7e970c4a797054b8e9&)

L'apprentissage non suppervisé permet à la machine d'utiliser des algorithmes d'aprentissage sans l'intervention d'humains. Ces algorithmes permettent de découvrir des similitudes ou des différences dans des jeux de données non etiquetées. L'apprentissage non supervisé est principalement utilisé pour résoudre trois taches :

- La classification : Dans l'apprentissage non supervisé, la classification consiste à regrouper des données non étiquetées en fonctions de leur différences ou similitudes. Il existe plusieurs methodes pour classifier des données. La classification exclusive et chevauchante se base sur l'algorithme de k-means (k-moyenne dans le langage de Molière), cet algorithme ce comporte de la manière suivante :

```python
# Algorithme de k-means :

Ajout des données # un set de données est donné a la machine, on peut les visualiser comme un groupe de points sur un plan.

Initialisation de points de valur k (clusters) # Les clusters sont des points de référence placés aléatoirement ou grâce à une estimation des données.

while(Cluster instable): # Au fil des itérations, la position des cluster va devenir de plus en plus précise.
   
      Assignation des données aux clusters # chaque point est relié au cluster le plus proche.

      Calcul de la nouvelle position du cluster # La nouvelle position est égale à la moyenne des points.

fin (clusters stable) # La position finale et les points qui sont associés aux clusters représentent les données de sortie.

```
- Règles d'association : Methode basée sur des règles qui permet de trouver des relations entre des variables dans un jeu de données. Cette méthode est souvent utilisée pour analyser le panier d'un consommateur pour faire le lien entre différents produits. La section "d'autres utilisateurs ont également acheté" d'Amazon est un bon exemple.

### Apprentissage semi suppervisé

L'apprentissage semi suppervisé est entre l'apprentissage supervisé et l'apprentissage non suppervisé. Il utilise un ensemble de données étiquetées et un ensemble de données non etiquetées. Il a été démontré que l'utilisation de données non étiquetées, en combinaison avec des données étiquetées, permet d'améliorer significativement la qualité de l'apprentissage. Un autre intérêt provient du fait que l'étiquetage de données nécessite souvent l'intervention d'un utilisateur humain. Lorsque les jeux de données deviennent très grands, cette opération peut s'avérer fastidieuse. Dans ce cas, l'apprentissage semi-supervisé, qui ne nécessite que quelques étiquettes, revêt un intérêt pratique évident.

### Apprentissage par renforcement

L’apprentissage par renforcement est un procédé d’apprentissage automatique consistant, pour un système autonome, à apprendre les actions à réaliser, à partir d'expériences, de façon à optimiser une récompense quantitative au cours du temps.

## Exemples d'application

- Objects connectés qui se calent sur le comportement de leurs utlilisateurs
- Détection des fraudes (fiscales par exemple)

![](https://cdn.discordapp.com/attachments/962004101552545852/1175029169747406909/danse-dance.gif)

- Analyses prédictives (statistiques)

## sources comme l'eau (de source)

https://www.ibm.com/fr-fr/topics/supervised-learning

https://fr.wikipedia.org/wiki/Apprentissage_automatique

https://ia-data-analytics.fr/machine-learning/



![](https://media.discordapp.net/attachments/1148600908373053492/1153687794774966323/1142868524419780768.gif)

***Quoicoube*** (on vit dans une saucisse)

