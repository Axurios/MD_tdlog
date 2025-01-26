---
marp : true
theme : uncover
class : invert
math: mathjax
paginate: true
---
<style>
section::after {
    /* Layout of pagination content */
    box-sizing: border-box;
    text-align: center;
    width: 120px;
    height: 120px;
    line-height: 40px;
    padding: 20px;

    /* Triangle background */
    background: linear-gradient(-45deg, rgba(0, 0, 0, 0.05) 50%, transparent 100%);
    background-size: cover;
  }
blockquote {
  background: #ffedcc;
  border-left: 10px solid #d1bf9d;
  margin: 1.5em 10px;
  padding: 0.5em 10px;
  color: #3b3b3b;
}
blockquote:before{
  content: unset;
}
blockquote:after{
  content: unset;
}
</style>

<!--- Welcome to our ReadMe, right-click on this md-file and "Open Preview" to this our presentation of this project --->
TDLOG & projet de département  
# Boltzmanian score matching fine-tuning  
Kenji Chikhaoui, Théodora Gospodaru et A.Dussolle.  
<!-- 27/01/2025 -->
![w:600 h:207](images/enpc.png)

---

#### Sommaire : 
* I -   Objectifs initiaux 
* II -  Choix techniques
* III - Réalisation(s)
* IV -  Difficultés rencontrées
* V -   Perspectives d'extension

---
<style scoped>
section {
    font-size: 25px;
}
</style> 
<!-- si besoin d'écrire beaucoup -->
### I - Objectifs initiaux
*Contexte* : Comment stabiliser les simulations moléculaires faites par GNN ?
* mauvaise corrélation RMSE - time-stability
* meilleure stabilité "by smoothing the loss landscape" 
 \
\
semble arbitraire :  
idéalement Loss qui rapproche les gradients : Divergence de Fisher sur Boltzmann. 

$F(p,  p_{\theta}) = \frac{1}{2} \int p(\mathbf{x}) \left\| \nabla_{\mathbf{x}} \log p(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\theta}(\mathbf{x}) \right\|^2 d\mathbf{x}$

---
D'abord restreint aux modèles linéaires, puis adaptation aux réseaux de neurones.
* Calculer modèle optimisant Fisher.
* Comparer les résultats RMSE/Fisher.  

![w:487 h:291](images/mse_energy.png) ![w:487 h:291](images/fisher_energy.png)

---

### II - Choix techniques
* python Qt plutôt que Web interface, test tkinter, problème interaction graph long terme.
* séparation UI, data, processing etc (check architecture name)
* Object oriented, dataholder etc
* python package for source code (might separate ui, etc in further package ?)
* plt plot into qt ui
<!-- --- 
commentaire  -->
---


```python
def selection_sort(seq):
    i=0
    while i < len(seq) - 1 :
        j_min = i
        j = i+1
        while j < len(seq):
            if seq[j] < seq[j_min]:
                j_min = j
            j += 1
        if j_min != 1 :
            swap(seq, i,j_min)
        i += 1

```

---
### III - Réalisation(s)
on considère **sorted = [seq[0]] trié**
ajoute seq[1] à la bonne place dans "sorted" (inserting)

```python
def insertion_sort(seq):
    i=1
    while i < len(seq) :
        j = i
        while j>0 and seq[j-1] < seq[j] :
            swap(seq, j, j-1)
            j = j-1
        i += 1
```

---
### IV - Difficultés rencontrées
* parcourir tous les éléments de la liste
* permuter indices i et i+1 si pas ordonnés
* répéter tant que "permutation a été nécessaire"
```python
def bubble_sort(seq):
    n = len(seq)
    swapped = True
    while swapped :
        swapped = False
        for i in range(1,n):
            if seq[i-1] > seq[i]:
                swap(seq, i-1, i)
                swapped = True
        n = n-1
```

---
### V - Perspectives d'extension
* séparer en deux parties
* trier chaque partie indépendamment -> récursivité
* fusionner les deux parties en gardant ordre

itère simultanément sur deux sous séquences puis place la valeur la plus faible dans le tableau et fais bouger son pointeur.

---
```python
def merge_sort(seq):
    if len(seq) < 2:
        return seq
    else :
        mid = len(seq)//2
        left = merge_sort(seq[:mid])
        right = merge_sort(seq[mid:])
        return merge(lef, right)
        n = n-1
```
```python
def merge(seq1, seq2):
    # long à écrire mais tranquille
```
représentation en arbre (largeur n, profondeur log(n))

---
### Annexe 


---
en python : list = tableau dynamique (taille variable), d'habitude tableau statique 
(réallocation pour rendre dynamique)
nbr d'éléments != capacité
list.append()
list.insert(index=0, -1)

---
predicting the future by making it, on s'est restraint sur les éléments d'un 
ensemble dénombrable, problème d'applicabilité (mémoire)

> predicting the future by making it, on s'est restraint sur
>
>les éléments d'un ensemble dénombrable, problème d'applicabilité (mémoire)

predicting the future by making it, on s'est restraint sur les éléments d'un ensemble dénombrable, problème d'applicabilité (mémoire)

---
$\mathbb{P}_{\pi \,,\, p}(Y=y\,|\,X=x) = \frac{\pi(y) \, p(x|y)}{\sum_{y' \in Y} \pi(y') \, p(x|y')}$

$$
\begin{aligned}
x & xx \\
y & yy 
\end{aligned}
$$
<https://www.markdownguide.org>
<fake@example.com>

content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
add in pagination style at beginning to make it a fraction

---
<!-- paginate: false -->
| Month    | Savings |
| -------- | ------- |
| January  | $250    |
| February | $80     |
| March    | $420    |


| Item              | In Stock | Price |
| :---------------- | :------: | ----: |
| Python Hat        |   True   | 23.99 |
| SQL Hat           |   True   | 23.99 |
| Codecademy Tee    |  False   | 19.99 |
| Codecademy Hoodie |  False   | 42.99 |

---
![](images/cea.png)