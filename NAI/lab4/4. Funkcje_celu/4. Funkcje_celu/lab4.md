# Lab 4

W bieżącym katalogu jest projekt zawierający pustą funkcję `empty()` oraz gotową funkcję `plot_function()` generuje wykres 3D tej funkcji. 

## Zadanie 1

[]Zaimplementować 5 wybranych funkcji testowych z [listy funkcji testowych z Wikipedii]((https://en.wikipedia.org/wiki/Test_functions_for_optimization) lub z [tego zbioru]((https://www.sfu.ca/~ssurjano/optimization.html). Przynajmniej jedna z nich musi mieć więcej niż jedno minimum (jak np. Rastrigin czy Ackley).)

## Zadanie 2
Dodaj mechanizm wyboru funkcji z wiersza poleceń za pomocą argumentów przekazywanych w `argv`. Jeśli użytkownik nie poda odpowiedniego argumentu, program powinien wyświetlić listę dostępnych funkcji. Program nadal trzeba będzie uruchomić przez PyCharm, ale z zakładki `Terminal` poleceniem `python test_function_plot.py <nazwa_funkcji>`.

## Zadanie domowe - Funkcja wielowymiarowa
Zaimplementuj funkcję, która będzie działała w 𝑛 wymiarach. Liczba wymiarów 𝑛 powinna również być wczytywana z wiersza poleceń – `python test_function_plot.py <nazwa_funkcji> <n>`.

## Zadanie dodatkowe – Funkcja dla problemu NP-kompletnego
Zaimplementuj funkcję optymalizacyjną dla dowolnego problemu NP-kompletnego z [listy problemów NP-kompletnych](https://en.wikipedia.org/wiki/List_of_NP-complete_problems). Napisz odpowiednią funkcję celu, która reprezentuje ten problem. Poniżej lista sugerowanych problemów:
* [Complete coloring](https://en.wikipedia.org/wiki/Complete_coloring)
* [Clique cover](https://en.wikipedia.org/wiki/Clique_cover)
* [Graph coloring](https://en.wikipedia.org/wiki/Graph_coloring)
* [Graph partition](https://en.wikipedia.org/wiki/Graph_partition)
* [3-partition problem](https://en.wikipedia.org/wiki/3-partition_problem)
* [Partition problem](https://en.wikipedia.org/wiki/Partition_problem)
* [Light Up (puzzle)](https://en.wikipedia.org/wiki/Light_Up_(puzzle))
