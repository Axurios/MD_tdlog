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
# Boltzmanian score matching fine-tuning
### TDLog & projet de département
Kenji Chikhaoui, Théodora Gospodaru et A.Dussolle.

---

Sommaire :
* I - Objectifs initiaux 
* II - Choix techniques
* III - Réalisation(s)
* IV - Difficultés rencontrées
* V - Perspectives d'extension

---
## I - Objectifs initiaux 
* séquence d'éléments
* accès par indice en O(1)

appartient cursus classique 
récursivité, intéressant d'un point de vue théorique, en pratique réutilise les algo pré-implémentés (i.e entraînement)

---

## Selection sort : $O(n^{2})$
passe en revue tous les éléments, choisis le plus petit, le mets en premier
recommence sur la liste privée du 1er élément. etc

--- 
* correction
* aboutit en temps fini
-> invariant(s)

laisser commentaire  (# [0,...,i] trié)

! lisibilité du code
code plus souvent lu qu'écrit

swap(list,indice_a, indice_b) permute les deux indices dans la liste


<!-- _paginate: skip -->
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
Boucle externe : en *n* itérations
Boucle interne : en *n - i* itérations
-> O($n^{2}$)

---
## Insertion sort : $O(n^{2})$
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
## Bubble sort : $O(n^{2})$
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
## Merge sort : $O(n \, log(n))$
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
## Quick sort : $O(n \, log(n))$
* choix d'un pivot
* partitionner en deux parties selon le pivot
* trier chaque partie
* concaténer les parties
```python
def quick_sort(seq):
    if len(seq) < 2:
        return seq
    else :
        pivot = seq[0]
        left,right = partition(seq[1:], pivot)  
        # compare au pivot, insère la liste correspondante
        return quick_sort(left) + pivot + quick_sort(right)
```

---
## Algorithmes linéaires
### Counting sort : $O(n)$
* compter le nombre d'occurrences des éléments
* parcours table d'occurrence et insère dans une liste
```python
def counting_sort(seq):
    occurrences = [0]*256
    for val in seq :
        occurrences[val] += 1
    i = 0 #pas fini !
```
---
#### Comment est-ce possible ?
on s'est restraint sur les éléments d'un ensemble dénombrable

problème d'applicabilité (mémoire)

---
## Annotations de type
```python
value : int = 1
#(bool,int,float,str,None,object)
def function()
    return seq


x : list[float] = [1.1, 2.4]
x : set[int] = {1,2,3}
x : dict[int,str] = {1:'one'}
x : tuple[int,str,float] = 1, 'one', 3.1
x : Optional[bool] = None
# optional permet d'avoir aussi None dans valeurs
```
None : *absence* de valeur.

---
```python

def function(seq : list[Optional[int]]) -> int:
    res : int = 0
    for item in seq :
        if item is not None:
            res += item
    return res

```
Utilisée comme une forme de documentation
utile pour les classes de données (struct en C++)

---
```python
import dataclasses
@dataclasses.dataclass

class AdataClass:
    an_attribut : int
    another : float

an_instance = AdataClass(an_attribut=1, another=2.3)

an_instance.an_attribut = 2 
#changing the value.
```
in this class, we're going to use "assert"

---
```python
def bucket_sort(seq,n):
    if len(seq) <= 1 :
        return seq
    # besoin rajouter 0 significatifs 



```
in this class, we're going to use "assert"

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