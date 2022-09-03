## Figure 1, 2, 10, 11
```
python submit_experiment.py --tab1 --tab2 --tab3 --tab4 --seed [seed]
```

## Figure 3-4
```
python submit_experiment.py --tab2-mc2a --tab2-mc3 --seed [seed]
```

## Figure 5
AGMM:
```
python run_agmm.py --seed [seed]
```
IS, IS-X:
```
python xfit.py --seed [seed]
```

## Figure 6
ES, ES-X:
```
python xfit.py --seed [seed]
```

## Figure 7-8
ES-5 fold:
```
python xfit.py --seed [seed] --five-fold
```

ES with 5 nearest neighbors
```
python xfit.py --seed [seed] --two-fold-small-neighbor
```
